import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from adapter import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL_PATH,
    generate_answer,
    load_model_and_tokenizer,
    unload_model,
)


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_FILE = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "processed" / "medquad_agent_val.json"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "outputs" / "medquad_eval_base_vs_adapter.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base Qwen vs Qwen+adapter on MedQuad validation data.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Local base model path.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER_PATH, help="Adapter directory.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Validation set JSON file.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Where to save eval results.")
    parser.add_argument("--max-samples", type=int, default=32, help="Number of validation examples to evaluate.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generation length.")
    parser.add_argument("--system-prompt", type=str, default="", help="Optional system prompt.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Evaluate with 4-bit base model loading.")
    return parser.parse_args()


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


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def word_tokens(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", normalize_text(text))


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = word_tokens(prediction)
    ref_tokens = word_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def lcs_length(tokens_a: List[str], tokens_b: List[str]) -> int:
    if not tokens_a or not tokens_b:
        return 0
    if len(tokens_a) < len(tokens_b):
        tokens_a, tokens_b = tokens_b, tokens_a

    previous = [0] * (len(tokens_b) + 1)
    for token_a in tokens_a:
        current = [0]
        for idx_b, token_b in enumerate(tokens_b, start=1):
            if token_a == token_b:
                current.append(previous[idx_b - 1] + 1)
            else:
                current.append(max(previous[idx_b], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    pred_tokens = word_tokens(prediction)
    ref_tokens = word_tokens(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "rouge_l_f1": rouge_l_f1(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "exact_match": exact_match(prediction, reference),
        "prediction_tokens": float(len(word_tokens(prediction))),
        "reference_tokens": float(len(word_tokens(reference))),
    }


def average_metrics(metric_rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_rows:
        return {}
    keys = metric_rows[0].keys()
    return {key: sum(row[key] for row in metric_rows) / len(metric_rows) for key in keys}


def evaluate_variant(
    variant_name: str,
    records: List[Dict],
    model_path: Path,
    adapter_path: Path | None,
    args: argparse.Namespace,
) -> List[Dict]:
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        adapter_path=adapter_path,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )

    rows: List[Dict] = []
    total = len(records)
    for idx, record in enumerate(records, start=1):
        question = record["conversations"][0]["value"]
        reference = record["conversations"][1]["value"]
        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
        metrics = compute_metrics(prediction, reference)
        rows.append(
            {
                "id": record["id"],
                "qtype": record.get("qtype", "unknown"),
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "metrics": metrics,
            }
        )
        print(
            f"[{variant_name}] {idx}/{total} "
            f"ROUGE-L={metrics['rouge_l_f1']:.4f} "
            f"Token-F1={metrics['token_f1']:.4f}"
        )

    unload_model(model)
    return rows


def summarize_by_qtype(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[row["qtype"]].append(row["metrics"])
    return {qtype: average_metrics(metric_rows) for qtype, metric_rows in sorted(grouped.items())}


def build_comparison(base_rows: List[Dict], adapter_rows: List[Dict]) -> List[Dict]:
    comparison: List[Dict] = []
    for base_row, adapter_row in zip(base_rows, adapter_rows):
        comparison.append(
            {
                "id": base_row["id"],
                "qtype": base_row["qtype"],
                "question": base_row["question"],
                "reference": base_row["reference"],
                "base_prediction": base_row["prediction"],
                "adapter_prediction": adapter_row["prediction"],
                "base_metrics": base_row["metrics"],
                "adapter_metrics": adapter_row["metrics"],
            }
        )
    return comparison


def metric_delta(base_summary: Dict[str, float], adapter_summary: Dict[str, float]) -> Dict[str, float]:
    return {key: adapter_summary[key] - base_summary[key] for key in base_summary.keys()}


def main() -> None:
    args = parse_args()
    records = subset_records(load_records(args.eval_file), args.max_samples, args.sample_seed)

    base_rows = evaluate_variant(
        variant_name="base",
        records=records,
        model_path=args.model_path,
        adapter_path=None,
        args=args,
    )
    adapter_rows = evaluate_variant(
        variant_name="adapter",
        records=records,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        args=args,
    )

    base_summary = average_metrics([row["metrics"] for row in base_rows])
    adapter_summary = average_metrics([row["metrics"] for row in adapter_rows])

    results = {
        "config": {
            "model_path": str(args.model_path),
            "adapter_path": str(args.adapter_path),
            "eval_file": str(args.eval_file),
            "max_samples": len(records),
            "max_new_tokens": args.max_new_tokens,
            "system_prompt": args.system_prompt,
            "load_in_4bit": args.load_in_4bit,
        },
        "metric_note": {
            "primary_metric": "rouge_l_f1",
            "reason": "MedQuad is long-form generative QA, so reference-based overlap metrics are more suitable than accuracy.",
        },
        "summary": {
            "base": base_summary,
            "adapter": adapter_summary,
            "adapter_minus_base": metric_delta(base_summary, adapter_summary),
        },
        "by_qtype": {
            "base": summarize_by_qtype(base_rows),
            "adapter": summarize_by_qtype(adapter_rows),
        },
        "examples": build_comparison(base_rows, adapter_rows),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print("\nSummary")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))
    print(f"\nSaved evaluation results to: {args.output_file}")


if __name__ == "__main__":
    main()
