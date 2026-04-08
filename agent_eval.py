import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from agent_runner import extract_final_answer, run_agent_loop
from adapter import DEFAULT_MODEL_PATH, load_model_and_tokenizer, unload_model


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_FILE = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "agent_posttrain_v2_react" / "agent_posttrain_val.json"
DEFAULT_V1_ADAPTER = ROOT_DIR / "outputs" / "qwen3_0.6b_agent_lora_v1"
DEFAULT_V2_ADAPTER = ROOT_DIR / "outputs" / "qwen3_0.6b_agent_lora_v2_react"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "outputs" / "agent_eval_v1_vs_v2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate legacy vs real-react agent adapters.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Base model path.")
    parser.add_argument("--v1-adapter-path", type=Path, default=DEFAULT_V1_ADAPTER, help="Legacy agent adapter path.")
    parser.add_argument("--v2-adapter-path", type=Path, default=DEFAULT_V2_ADAPTER, help="React multi-turn adapter path.")
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE, help="Agent validation file.")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="Where to save evaluation results.")
    parser.add_argument("--max-samples", type=int, default=-1, help="How many eval samples to use. -1 means all.")
    parser.add_argument("--sample-seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--max-steps", type=int, default=2, help="Maximum tool-use iterations per sample.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum tokens per generation step.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load models in 4-bit mode.")
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


def average_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def extract_reference_answer(record: Dict) -> str:
    assistant_turns = [turn["value"] for turn in record["conversations"] if turn["from"] == "assistant"]
    if not assistant_turns:
        return ""
    final = extract_final_answer(assistant_turns[-1])
    return final if final is not None else assistant_turns[-1].strip()


def summarize_tool_metrics(rows: List[Dict]) -> Dict[str, float]:
    total = len(rows)
    if total == 0:
        return {}

    tool_called = sum(1 for row in rows if row["tool_called"])
    execution_ok = sum(1 for row in rows if row["tool_execution_success"])
    tool_correct = sum(1 for row in rows if row["tool_choice_correct"])
    completed = sum(1 for row in rows if row["completed"])
    no_error = sum(1 for row in rows if not row["error"])

    return {
        "tool_call_rate": tool_called / total,
        "tool_execution_success_rate": execution_ok / total,
        "tool_choice_accuracy": tool_correct / total,
        "completion_rate": completed / total,
        "no_error_rate": no_error / total,
    }


def summarize_answer_metrics(rows: List[Dict]) -> Dict[str, float]:
    metrics = []
    for row in rows:
        metrics.append(
            {
                "rouge_l_f1": rouge_l_f1(row["final_answer"], row["reference_answer"]),
                "token_f1": token_f1(row["final_answer"], row["reference_answer"]),
                "exact_match": exact_match(row["final_answer"], row["reference_answer"]),
                "prediction_tokens": float(len(word_tokens(row["final_answer"]))),
                "reference_tokens": float(len(word_tokens(row["reference_answer"]))),
            }
        )
    return average_dict(metrics)


def summarize_by_tool_type(rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in rows:
        grouped[row["gold_tool_type"]].append(row)
    return {
        tool_type: {
            **summarize_tool_metrics(group_rows),
            **summarize_answer_metrics(group_rows),
        }
        for tool_type, group_rows in sorted(grouped.items())
    }


def evaluate_variant(
    variant_name: str,
    adapter_path: Path,
    records: List[Dict],
    args: argparse.Namespace,
) -> List[Dict]:
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=adapter_path,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )

    rows: List[Dict] = []
    total = len(records)
    for idx, record in enumerate(records, start=1):
        query = next(turn["value"] for turn in record["conversations"] if turn["from"] == "human")
        result = run_agent_loop(
            query=query,
            model=model,
            tokenizer=tokenizer,
            system_prompt=record.get("system_prompt", ""),
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
        )

        first_step = result["steps"][0] if result["steps"] else {}
        observation = first_step.get("observation", "")
        row = {
            "id": record["id"],
            "query": query,
            "gold_tool_type": record.get("tool_type", "unknown"),
            "reference_answer": extract_reference_answer(record),
            "tool_called": bool(result["steps"]),
            "tool_execution_success": bool(result["steps"]) and not str(observation).startswith("Error"),
            "predicted_tool_type": first_step.get("tool_name", ""),
            "tool_choice_correct": first_step.get("tool_name", "") == record.get("tool_type", ""),
            "completed": result.get("completed", False),
            "error": result.get("error", ""),
            "final_answer": result.get("final_answer", ""),
            "num_steps": len(result["steps"]),
            "steps": result["steps"],
            "last_generation": result.get("last_generation", ""),
        }
        rows.append(row)

        print(
            f"[{variant_name}] {idx}/{total} "
            f"tool_called={int(row['tool_called'])} "
            f"tool_ok={int(row['tool_execution_success'])} "
            f"tool_correct={int(row['tool_choice_correct'])} "
            f"completed={int(row['completed'])}"
        )

    unload_model(model)
    return rows


def summarize_variant(rows: List[Dict]) -> Dict:
    return {
        "tool_metrics": summarize_tool_metrics(rows),
        "answer_metrics": summarize_answer_metrics(rows),
        "by_tool_type": summarize_by_tool_type(rows),
    }


def metric_delta(lhs: Dict[str, float], rhs: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(set(lhs.keys()) & set(rhs.keys()))
    return {key: rhs[key] - lhs[key] for key in keys}


def build_examples(v1_rows: List[Dict], v2_rows: List[Dict]) -> List[Dict]:
    examples: List[Dict] = []
    for legacy_row, react_row in zip(v1_rows, v2_rows):
        examples.append(
            {
                "id": legacy_row["id"],
                "query": legacy_row["query"],
                "gold_tool_type": legacy_row["gold_tool_type"],
                "reference_answer": legacy_row["reference_answer"],
                "v1_legacy": {
                    "predicted_tool_type": legacy_row["predicted_tool_type"],
                    "tool_called": legacy_row["tool_called"],
                    "tool_execution_success": legacy_row["tool_execution_success"],
                    "tool_choice_correct": legacy_row["tool_choice_correct"],
                    "completed": legacy_row["completed"],
                    "error": legacy_row["error"],
                    "final_answer": legacy_row["final_answer"],
                },
                "v2_react": {
                    "predicted_tool_type": react_row["predicted_tool_type"],
                    "tool_called": react_row["tool_called"],
                    "tool_execution_success": react_row["tool_execution_success"],
                    "tool_choice_correct": react_row["tool_choice_correct"],
                    "completed": react_row["completed"],
                    "error": react_row["error"],
                    "final_answer": react_row["final_answer"],
                },
            }
        )
    return examples


def main() -> None:
    args = parse_args()
    records = subset_records(load_records(args.eval_file), args.max_samples, args.sample_seed)

    v1_rows = evaluate_variant("v1_legacy", args.v1_adapter_path, records, args)
    v2_rows = evaluate_variant("v2_react", args.v2_adapter_path, records, args)

    v1_summary = summarize_variant(v1_rows)
    v2_summary = summarize_variant(v2_rows)

    results = {
        "config": {
            "model_path": str(args.model_path),
            "v1_adapter_path": str(args.v1_adapter_path),
            "v2_adapter_path": str(args.v2_adapter_path),
            "eval_file": str(args.eval_file),
            "num_samples": len(records),
            "max_steps": args.max_steps,
            "max_new_tokens": args.max_new_tokens,
            "load_in_4bit": args.load_in_4bit,
        },
        "metric_note": {
            "tool_call_rate": "Fraction of samples where the model emitted at least one executable tool action.",
            "tool_execution_success_rate": "Fraction of samples where the first tool call executed without an error string.",
            "tool_choice_accuracy": "Fraction of samples where the first predicted tool matches the gold tool type.",
            "completion_rate": "Fraction of samples where the runner received a Final Answer before max_steps.",
            "answer_metrics": "Reference-based overlap between final answer and gold final answer.",
        },
        "summary": {
            "v1_legacy": v1_summary,
            "v2_react": v2_summary,
            "v2_minus_v1": {
                "tool_metrics": metric_delta(v1_summary["tool_metrics"], v2_summary["tool_metrics"]),
                "answer_metrics": metric_delta(v1_summary["answer_metrics"], v2_summary["answer_metrics"]),
            },
        },
        "examples": build_examples(v1_rows, v2_rows),
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    print(f"Saved evaluation to: {args.output_file}")
    print(json.dumps(results["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
