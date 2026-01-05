import argparse
import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from jiwer import wer as jiwer_wer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)

# PEFT (LoRA)
from peft import LoraConfig, TaskType, get_peft_model


LOGGER = logging.getLogger("whisper_lora_sagemaker")


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        stream=sys.stdout,
    )
    # Make HF + datasets a bit quieter
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.INFO)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name, default)
    return v if v not in ("", None) else default


def _hf_load_dataset(*args, **kwargs):
    """
    HF datasets auth parameter changed across versions (`token` vs `use_auth_token`).
    This helper tries both, pulling token from env vars only.
    """
    token = _env("HF_TOKEN") or _env("HUGGINGFACEHUB_API_TOKEN")
    if token:
        try:
            return load_dataset(*args, token=token, **kwargs)
        except TypeError:
            return load_dataset(*args, use_auth_token=token, **kwargs)
    return load_dataset(*args, **kwargs)


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _normalize_columns(
    ds: Dataset,
    audio_col: str,
    text_col: str,
    keep_cols: Optional[Sequence[str]] = None,
) -> Dataset:
    keep_cols = set(keep_cols or [])
    required = {audio_col, text_col}
    missing = [c for c in required if c not in ds.column_names]
    if missing:
        raise ValueError(f"Dataset missing columns {missing}. Has: {ds.column_names}")

    # Standardize to ["audio", "text"] for downstream
    if audio_col != "audio":
        ds = ds.rename_column(audio_col, "audio")
    if text_col != "text":
        ds = ds.rename_column(text_col, "text")

    # Drop junk columns unless user asked to keep them
    cols_to_keep = {"audio", "text"} | keep_cols
    drop_cols = [c for c in ds.column_names if c not in cols_to_keep]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds


def _load_one_source(source_cfg: Dict[str, Any]) -> Dataset:
    """
    Supported source types:
      - s3_jsonl: manifests: [s3://.../train.jsonl, ...] (or local paths)
      - hf: name: <dataset_name>, config: <optional>, split: <split string>
    """
    stype = source_cfg["type"]
    if stype == "s3_jsonl":
        manifests = _as_list(source_cfg.get("manifests"))
        if not manifests:
            raise ValueError("s3_jsonl source requires `manifests` (list of jsonl files).")

        # jsonl is handled by datasets' "json" loader; fsspec+s3fs handles s3://...
        ds = _hf_load_dataset(
            "json",
            data_files=manifests,
            split="train",  # JSON loader uses 'train' split by default
        )

        audio_col = source_cfg.get("audio_column", "audio")
        text_col = source_cfg.get("text_column", "text")
        ds = _normalize_columns(ds, audio_col=audio_col, text_col=text_col)

        # Cast audio path/uri column to Audio feature (lazy decode until accessed)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    if stype == "hf":
        name = source_cfg["name"]
        config_name = source_cfg.get("config", None)
        split = source_cfg.get("split", "train")
        ds = _hf_load_dataset(name, config_name, split=split)

        audio_col = source_cfg.get("audio_column", "audio")
        text_col = source_cfg.get("text_column", "text")
        ds = _normalize_columns(ds, audio_col=audio_col, text_col=text_col)

        # If HF dataset audio isn’t already Audio(sampling_rate=16000), cast it.
        if ds.features.get("audio") is None or not isinstance(ds.features["audio"], Audio):
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        else:
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds

    raise ValueError(f"Unsupported source type: {stype}")


def _load_split(split_sources: List[Dict[str, Any]]) -> Dataset:
    parts = []
    for src in split_sources:
        parts.append(_load_one_source(src))
    if not parts:
        raise ValueError("No dataset sources configured for split.")
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor
    padding: str = "longest"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # input_features: List[np.ndarray] (log-mel)
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # Replace padding with -100 to ignore in loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        batch["labels"] = labels
        return batch


def main():
    _setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args, unknown = parser.parse_known_args()
    if unknown:
        # "No overriding": we intentionally ignore extra CLI args.
        LOGGER.info("Ignoring extra CLI args (config is source of truth): %s", unknown)

    cfg = _read_yaml(args.config)

    # SageMaker standard dirs (don’t treat these as hyperparameters)
    sm_model_dir = _env("SM_MODEL_DIR", "./model")
    sm_output_data_dir = _env("SM_OUTPUT_DATA_DIR", "./outputs")

    os.makedirs(sm_model_dir, exist_ok=True)
    os.makedirs(sm_output_data_dir, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    # WANDB: API key via env var, config only determines behavior
    wandb_cfg = cfg.get("wandb", {}) or {}
    wandb_enabled = bool(wandb_cfg.get("enabled", True))
    if wandb_enabled:
        # Let wandb read WANDB_API_KEY from env
        if wandb_cfg.get("project"):
            os.environ["WANDB_PROJECT"] = str(wandb_cfg["project"])
        if wandb_cfg.get("entity"):
            os.environ["WANDB_ENTITY"] = str(wandb_cfg["entity"])
        if wandb_cfg.get("name"):
            os.environ["WANDB_NAME"] = str(wandb_cfg["name"])

    model_cfg = cfg["model"]
    model_name = model_cfg["name_or_path"]

    processor = WhisperProcessor.from_pretrained(model_name)

    language = model_cfg.get("language")
    task = model_cfg.get("task", "transcribe")
    if language:
        processor.tokenizer.set_prefix_tokens(language=language, task=task)

    # Load datasets (no streaming)
    data_cfg = cfg["data"]
    train_sources = data_cfg.get("train", [])
    eval_sources = data_cfg.get("eval", [])

    LOGGER.info("Loading train sources: %d", len(train_sources))
    train_ds = _load_split(train_sources)

    LOGGER.info("Loading eval sources: %d", len(eval_sources))
    eval_ds = _load_split(eval_sources) if eval_sources else None

    # Optional filtering (cheap checks only)
    filters = data_cfg.get("filters", {}) or {}
    drop_empty_text = bool(filters.get("drop_empty_text", True))
    if drop_empty_text:
        train_ds = train_ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)
        if eval_ds is not None:
            eval_ds = eval_ds.filter(lambda x: isinstance(x["text"], str) and len(x["text"].strip()) > 0)

    # Feature prep
    num_proc = int(cfg.get("preprocessing", {}).get("num_proc", 4))
    max_label_length = cfg.get("preprocessing", {}).get("max_label_length", None)

    def prepare_batch(batch):
        audio = batch["audio"]  # dict with array + sampling_rate
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        tokenized = processor.tokenizer(batch["text"])
        labels = tokenized["input_ids"]
        if isinstance(max_label_length, int) and max_label_length > 0:
            labels = labels[:max_label_length]
        batch["labels"] = labels
        return batch

    # Map (non-streaming, cached; Audio decode happens here)
    LOGGER.info("Preprocessing train dataset...")
    train_ds = train_ds.map(
        prepare_batch,
        remove_columns=train_ds.column_names,
        num_proc=num_proc,
        desc="Preparing train features",
    )

    if eval_ds is not None:
        LOGGER.info("Preprocessing eval dataset...")
        eval_ds = eval_ds.map(
            prepare_batch,
            remove_columns=eval_ds.column_names,
            num_proc=num_proc,
            desc="Preparing eval features",
        )

    # Model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)

    # Whisper generation prompt tokens
    if language:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        model.config.forced_decoder_ids = forced_decoder_ids
    else:
        model.config.forced_decoder_ids = None

    # Common training tweaks
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    # LoRA
    lora_cfg = cfg.get("lora", {}) or {}
    target_modules = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(target_modules),
        bias=str(lora_cfg.get("bias", "none")),
    )
    model = get_peft_model(model, lora)

    try:
        model.print_trainable_parameters()
    except Exception:
        LOGGER.info("Trainable parameters printed by PEFT not available in this version.")

    # Training args from YAML only
    tr_cfg = cfg["training"]
    report_to = ["wandb"] if wandb_enabled else []
    # Put checkpoints in output_data_dir; save final adapter+processor into SM_MODEL_DIR
    output_dir = tr_cfg.get("output_dir") or os.path.join(sm_output_data_dir, "checkpoints")

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=eval_ds is not None,
        evaluation_strategy=tr_cfg.get("evaluation_strategy", "steps" if eval_ds is not None else "no"),
        eval_steps=tr_cfg.get("eval_steps", 500),
        save_steps=tr_cfg.get("save_steps", 500),
        logging_steps=tr_cfg.get("logging_steps", 50),
        save_total_limit=int(tr_cfg.get("save_total_limit", 2)),
        per_device_train_batch_size=int(tr_cfg.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(tr_cfg.get("per_device_eval_batch_size", 8)),
        gradient_accumulation_steps=int(tr_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(tr_cfg.get("learning_rate", 1e-4)),
        warmup_steps=int(tr_cfg.get("warmup_steps", 0)),
        num_train_epochs=float(tr_cfg.get("num_train_epochs", 3)),
        fp16=bool(tr_cfg.get("fp16", True)),
        bf16=bool(tr_cfg.get("bf16", False)),
        predict_with_generate=bool(tr_cfg.get("predict_with_generate", True)),
        generation_max_length=int(tr_cfg.get("generation_max_length", 225)),
        dataloader_num_workers=int(tr_cfg.get("dataloader_num_workers", 4)),
        metric_for_best_model=tr_cfg.get("metric_for_best_model", "wer"),
        greater_is_better=bool(tr_cfg.get("greater_is_better", False)),
        load_best_model_at_end=bool(tr_cfg.get("load_best_model_at_end", eval_ds is not None)),
        report_to=report_to,
        run_name=wandb_cfg.get("name"),
        seed=seed,
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Replace -100 in the labels as we can't decode them.
        label_ids = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)

        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # jiwer expects lists of strings
        value = jiwer_wer(label_str, pred_str)
        return {"wer": value}

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
        compute_metrics=compute_metrics if eval_ds is not None else None,
    )

    LOGGER.info("Starting training...")
    train_result = trainer.train()

    LOGGER.info("Training complete. Saving final artifacts to: %s", sm_model_dir)
    # Save LoRA adapter (PEFT) + config
    trainer.save_model(sm_model_dir)
    # Save processor (feature extractor + tokenizer)
    processor.save_pretrained(sm_model_dir)

    # Save a small training summary
    summary = {
        "train_metrics": train_result.metrics,
    }
    if eval_ds is not None:
        eval_metrics = trainer.evaluate(metric_key_prefix="eval")
        summary["eval_metrics"] = eval_metrics

    with open(os.path.join(sm_model_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()
