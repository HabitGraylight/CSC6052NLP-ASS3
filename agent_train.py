import argparse
import importlib.util
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


IGNORE_INDEX = -100
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "Qwen3-0.6B"
DEFAULT_TRAIN_PATH = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "processed" / "medquad_agent_train.json"
DEFAULT_VAL_PATH = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "processed" / "medquad_agent_val.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "qwen3_0.6b_medquad_sft"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tuning for Qwen3-0.6B on processed MedQuad conversations."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local Qwen model path.")
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_PATH, help="Processed train json file.")
    parser.add_argument("--val-file", type=Path, default=DEFAULT_VAL_PATH, help="Processed validation json file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save checkpoints.")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument("--max-train-samples", type=int, default=1024, help="Use only a subset for quick runs.")
    parser.add_argument("--max-eval-samples", type=int, default=128, help="Use only a subset for quick evaluation.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Random seed for subset sampling.")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Per-device eval batch size.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Peak learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of train epochs.")
    parser.add_argument("--logging-steps", type=int, default=10, help="Logging interval.")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint save interval.")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation interval.")
    parser.add_argument("--save-total-limit", type=int, default=2, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--seed", type=int, default=42, help="Training seed.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt prepended before each MedQuad dialogue.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory usage.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading tokenizer/model.",
    )
    parser.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit mode.")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="all-linear",
        help="Comma-separated target modules for LoRA, or 'all-linear'.",
    )
    return parser.parse_args()


def require_dependencies(args: argparse.Namespace) -> None:
    required = ["torch", "transformers", "accelerate"]
    if args.use_lora:
        required.append("peft")
    if args.load_in_4bit:
        required.append("bitsandbytes")

    missing = [module_name for module_name in required if importlib.util.find_spec(module_name) is None]
    if missing:
        missing_text = ", ".join(missing)
        raise ImportError(
            f"Missing required packages: {missing_text}. "
            "Install them first, for example: pip3 install --user torch transformers accelerate peft bitsandbytes"
        )


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, but got {type(data).__name__}.")
    return data


def subset_records(records: List[Dict], max_samples: int, seed: int) -> List[Dict]:
    if max_samples <= 0 or max_samples >= len(records):
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    selected = indices[:max_samples]
    return [records[idx] for idx in selected]


def resolve_system_prompt(record: Dict, default_system_prompt: str) -> str:
    if default_system_prompt:
        return default_system_prompt
    return record.get("system_prompt", "")


def to_chat_messages(record: Dict, system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for turn in record["conversations"]:
        sender = turn["from"]
        if sender == "human":
            role = "user"
        elif sender == "assistant":
            role = "assistant"
        elif sender == "tool":
            role = "tool"
        else:
            raise ValueError(f"Unsupported speaker '{sender}' in record {record.get('id', '<unknown>')}.")
        messages.append({"role": role, "content": turn["value"]})
    return messages


def render_chat(tokenizer, messages: Sequence[Dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def mask_non_assistant_tokens(record: Dict, tokenizer, messages: Sequence[Dict[str, str]], input_ids: List[int]) -> List[int]:
    labels = [IGNORE_INDEX] * len(input_ids)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        prefix_ids = tokenizer.encode(
            render_chat(tokenizer, messages[:index], add_generation_prompt=True),
            add_special_tokens=False,
        )
        assistant_ids = tokenizer.encode(
            render_chat(tokenizer, messages[: index + 1], add_generation_prompt=False),
            add_special_tokens=False,
        )
        start = len(prefix_ids)
        end = len(assistant_ids)
        if end < start:
            raise ValueError(f"Assistant span mismatch in record {record.get('id', '<unknown>')}.")
        labels[start:end] = input_ids[start:end]

    return labels


def encode_supervised_example(record: Dict, tokenizer, max_seq_length: int, system_prompt: str) -> Dict[str, List[int]]:
    record_system_prompt = resolve_system_prompt(record, system_prompt)
    messages = to_chat_messages(record, record_system_prompt)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError(f"Record {record.get('id', '<unknown>')} must end with an assistant turn.")

    full_text = render_chat(tokenizer, messages, add_generation_prompt=False)
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = mask_non_assistant_tokens(record, tokenizer, messages, input_ids)

    if len(input_ids) != len(labels):
        raise ValueError(f"Token-label length mismatch in record {record.get('id', '<unknown>')}.")

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    return {"input_ids": input_ids, "labels": labels}


class SupervisedConversationDataset:
    def __init__(self, records: Sequence[Dict], tokenizer, max_seq_length: int, system_prompt: str) -> None:
        self.examples = [
            encode_supervised_example(record, tokenizer, max_seq_length=max_seq_length, system_prompt=system_prompt)
            for record in records
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.examples[index]


@dataclass
class SupervisedDataCollator:
    tokenizer: object

    def __call__(self, instances: Sequence[Dict[str, List[int]]]) -> Dict[str, object]:
        import torch

        input_ids = [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in instances]
        labels = [torch.tensor(instance["labels"], dtype=torch.long) for instance in instances]

        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = padded_input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": attention_mask,
        }


def resolve_precision_flags(torch_module) -> Dict[str, bool]:
    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def build_model_dtype(torch_module, precision_flags: Dict[str, bool]):
    if precision_flags["bf16"]:
        return torch_module.bfloat16
    if precision_flags["fp16"]:
        return torch_module.float16
    return torch_module.float32


def parse_lora_target_modules(raw_value: str):
    if raw_value.strip() == "all-linear":
        return "all-linear"
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    require_dependencies(args)

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    train_records = subset_records(load_records(args.train_file), args.max_train_samples, args.sample_seed)
    val_records = subset_records(load_records(args.val_file), args.max_eval_samples, args.sample_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = SupervisedConversationDataset(
        records=train_records,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        system_prompt=args.system_prompt,
    )
    val_dataset = SupervisedConversationDataset(
        records=val_records,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        system_prompt=args.system_prompt,
    )

    precision_flags = resolve_precision_flags(torch)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=build_model_dtype(torch, precision_flags),
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=build_model_dtype(torch, precision_flags),
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.use_cache = False if args.gradient_checkpointing else model.config.use_cache

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if args.use_lora:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=parse_lora_target_modules(args.lora_target_modules),
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    steps_per_epoch = max(
        1,
        math.ceil(len(train_dataset) / max(1, args.train_batch_size * args.gradient_accumulation_steps)),
    )
    total_train_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    warmup_steps = max(1, int(round(total_train_steps * args.warmup_ratio))) if args.warmup_ratio > 0 else 0

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        save_total_limit=args.save_total_limit,
        dataloader_pin_memory=torch.cuda.is_available(),
        bf16=precision_flags["bf16"],
        fp16=precision_flags["fp16"],
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="paged_adamw_32bit" if args.load_in_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        seed=args.seed,
        do_train=True,
        do_eval=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=SupervisedDataCollator(tokenizer=tokenizer),
    )

    print(f"Model path: {args.model_path}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Train samples used: {len(train_dataset)}")
    print(f"Val samples used: {len(val_dataset)}")
    print(f"Output dir: {args.output_dir}")
    print(f"Use LoRA: {args.use_lora}")
    print(f"Load in 4-bit: {args.load_in_4bit}")

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
