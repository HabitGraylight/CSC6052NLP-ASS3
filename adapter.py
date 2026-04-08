import argparse
import gc
import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "Qwen3-0.6B"
DEFAULT_ADAPTER_PATH = ROOT_DIR / "outputs" / "qwen3_0.6b_medquad_lora_v2_seq768"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Qwen3-0.6B with an optional LoRA adapter for generation.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local base model path.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH, help="Adapter directory.")
    parser.add_argument("--prompt", type=str, default="", help="Single prompt to run.")
    parser.add_argument("--system-prompt", type=str, default="", help="Optional system prompt.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive chat loop.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit mode.")
    parser.add_argument(
        "--merge-output-dir",
        type=Path,
        default=None,
        help="If provided, merge the adapter into the base model and save the merged model here.",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Ignore the adapter and run the base model only.",
    )
    return parser.parse_args()


def require_dependencies(use_adapter: bool) -> None:
    required = ["torch", "transformers"]
    if use_adapter:
        required.append("peft")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise ImportError(
            f"Missing required packages: {', '.join(missing)}. "
            "Install them first, for example: pip3 install --user torch transformers peft bitsandbytes"
        )


def resolve_precision_flags(torch_module) -> dict:
    if torch_module.cuda.is_available():
        if torch_module.cuda.is_bf16_supported():
            return {"bf16": True, "fp16": False}
        return {"bf16": False, "fp16": True}
    return {"bf16": False, "fp16": False}


def build_model_dtype(torch_module, precision_flags: dict):
    if precision_flags["bf16"]:
        return torch_module.bfloat16
    if precision_flags["fp16"]:
        return torch_module.float16
    return torch_module.float32


def get_tokenizer_source(model_path: Path, adapter_path: Optional[Path]) -> Path:
    if adapter_path is not None and (adapter_path / "tokenizer.json").exists():
        return adapter_path
    return model_path


def load_model_and_tokenizer(
    model_path: Path,
    adapter_path: Optional[Path] = None,
    trust_remote_code: bool = False,
    load_in_4bit: bool = False,
) -> Tuple[object, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(
        get_tokenizer_source(model_path, adapter_path),
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    precision_flags = resolve_precision_flags(torch)
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=build_model_dtype(torch, precision_flags),
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=build_model_dtype(torch, precision_flags),
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def model_input_device(model) -> object:
    return next(model.parameters()).device


def render_messages(tokenizer, messages: List[dict], add_generation_prompt: bool = True) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)


def build_prompt(tokenizer, prompt: str, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return render_messages(tokenizer, messages, add_generation_prompt=True)


def generate_from_messages(
    model,
    tokenizer,
    messages: List[dict],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    import torch

    rendered = render_messages(tokenizer, messages, add_generation_prompt=True)
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    device = model_input_device(model)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generate_answer(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = "",
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return generate_from_messages(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )


def unload_model(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def merge_adapter(
    model_path: Path,
    adapter_path: Path,
    output_dir: Path,
    trust_remote_code: bool = False,
) -> None:
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        trust_remote_code=trust_remote_code,
        load_in_4bit=False,
    )
    merged_model = model.merge_and_unload()
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    unload_model(merged_model)


def interactive_loop(model, tokenizer, args: argparse.Namespace) -> None:
    while True:
        try:
            prompt = input("\nUser> ").strip()
        except EOFError:
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nAssistant> {answer}")


def main() -> None:
    args = parse_args()
    adapter_path = None if args.base_only else args.adapter_path
    require_dependencies(use_adapter=adapter_path is not None)

    if args.merge_output_dir is not None:
        if args.base_only:
            raise ValueError("--merge-output-dir requires an adapter; remove --base-only.")
        if args.load_in_4bit:
            raise ValueError("Merging is only supported without --load-in-4bit.")
        merge_adapter(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            output_dir=args.merge_output_dir,
            trust_remote_code=args.trust_remote_code,
        )
        print(f"Merged adapter saved to: {args.merge_output_dir}")
        return

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=adapter_path,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )

    if args.prompt:
        answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(answer)
    elif args.interactive:
        interactive_loop(model, tokenizer, args)
    else:
        raise ValueError("Provide --prompt, --interactive, or --merge-output-dir.")


if __name__ == "__main__":
    main()
