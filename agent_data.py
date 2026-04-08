import argparse
import json
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from agent_config import AGENT_SYSTEM_PROMPT
from tools import calculator, retrieve_from_kb


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATH = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "processed" / "medquad_agent_train.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "agent_posttrain_v2_react"


RETRIEVE_THOUGHTS = [
    "This is a factual medical question, so I should retrieve the most relevant local knowledge first.",
    "I should ground the answer in the local MedQuad knowledge base before responding.",
    "The safest next step is to retrieve the most relevant medical passage from the local knowledge base.",
]

RETRIEVE_FOLLOWUPS = [
    "The retrieved passage contains the answer, so I can now answer the user directly.",
    "The observation gives me enough grounded information to answer accurately.",
    "I can use the retrieved evidence to produce the final answer.",
]

CALCULATOR_THOUGHTS = [
    "This requires arithmetic, so I should use the calculator instead of estimating.",
    "I should compute the value explicitly with the calculator.",
    "The task is numerical, so I will calculate it before answering.",
]

CALCULATOR_FOLLOWUPS = [
    "Now that I have the computed result, I can answer clearly.",
    "The calculation is complete, so I can provide the final answer.",
    "I can now convert the calculation result into a concise final answer.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate local agent post-train data for tool use.")
    parser.add_argument("--source-file", type=Path, default=DEFAULT_SOURCE_PATH, help="Processed MedQuad train set.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for generated agent data.")
    parser.add_argument("--max-retrieve-samples", type=int, default=1500, help="Number of retrieve-based samples to build.")
    parser.add_argument("--num-calculator-samples", type=int, default=200, help="Number of synthetic calculator samples.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--format-version",
        type=str,
        default="react_multiturn",
        choices=["legacy_trace", "react_multiturn"],
        help="Emit either the old single-turn trajectories or the new multi-turn agent format.",
    )
    return parser.parse_args()


def load_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, but got {type(data).__name__}.")
    return data


def retrieve_action(query: str, top_k: int = 2) -> str:
    return f"Action: retrieve_from_kb({json.dumps({'query': query, 'top_k': top_k}, ensure_ascii=False)})"


def calculator_action(expression: str) -> str:
    return f"Action: calculator({json.dumps(expression, ensure_ascii=False)})"


def build_legacy_retrieve_sample(record: Dict, rng: random.Random, index: int) -> Dict:
    question = record["conversations"][0]["value"]
    answer = record["conversations"][1]["value"]
    qtype = record.get("qtype", "unknown")
    query = f"{qtype} {question}"
    observation = retrieve_from_kb(query=query, top_k=2)

    trace = "\n".join(
        [
            f"Thought: {rng.choice(RETRIEVE_THOUGHTS)}",
            retrieve_action(query=query, top_k=2),
            f"Observation: {observation}",
            f"Thought: {rng.choice(RETRIEVE_FOLLOWUPS)}",
            f"Final Answer: {answer}",
        ]
    )

    return {
        "id": f"agent-retrieve-{index:05d}",
        "dataset": "MedQuad-Agent",
        "source_record_id": record["id"],
        "tool_type": "retrieve_from_kb",
        "format_version": "legacy_trace",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {"from": "assistant", "value": trace},
        ],
    }


def build_multiturn_retrieve_sample(record: Dict, rng: random.Random, index: int) -> Dict:
    question = record["conversations"][0]["value"]
    answer = record["conversations"][1]["value"]
    qtype = record.get("qtype", "unknown")
    query = f"{qtype} {question}"
    observation = retrieve_from_kb(query=query, top_k=2)

    return {
        "id": f"agent-retrieve-{index:05d}",
        "dataset": "MedQuad-Agent",
        "source_record_id": record["id"],
        "tool_type": "retrieve_from_kb",
        "format_version": "react_multiturn",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(RETRIEVE_THOUGHTS)}",
                        retrieve_action(query=query, top_k=2),
                    ]
                ),
            },
            {"from": "tool", "value": observation},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(RETRIEVE_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def build_legacy_weight_based_dose_sample(rng: random.Random, index: int) -> Dict:
    weight = rng.randint(8, 80)
    dose_per_kg = rng.choice([5, 7.5, 10, 12.5, 15])
    expression = f"{weight} * {dose_per_kg}"
    result = calculator(expression)
    question = f"A patient weighs {weight} kg and a medicine is prescribed at {dose_per_kg} mg/kg. What is the single-dose amount in mg?"
    answer = f"At {dose_per_kg} mg/kg for {weight} kg, the single dose is {result} mg."
    return {
        "id": f"agent-calculator-{index:05d}",
        "dataset": "Synthetic-MedCalc",
        "tool_type": "calculator",
        "format_version": "legacy_trace",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_THOUGHTS)}",
                        calculator_action(expression),
                        f"Observation: {result}",
                        f"Thought: {rng.choice(CALCULATOR_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def build_multiturn_weight_based_dose_sample(rng: random.Random, index: int) -> Dict:
    weight = rng.randint(8, 80)
    dose_per_kg = rng.choice([5, 7.5, 10, 12.5, 15])
    expression = f"{weight} * {dose_per_kg}"
    result = calculator(expression)
    question = f"A patient weighs {weight} kg and a medicine is prescribed at {dose_per_kg} mg/kg. What is the single-dose amount in mg?"
    answer = f"At {dose_per_kg} mg/kg for {weight} kg, the single dose is {result} mg."
    return {
        "id": f"agent-calculator-{index:05d}",
        "dataset": "Synthetic-MedCalc",
        "tool_type": "calculator",
        "format_version": "react_multiturn",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_THOUGHTS)}",
                        calculator_action(expression),
                    ]
                ),
            },
            {"from": "tool", "value": str(result)},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def build_legacy_divided_daily_dose_sample(rng: random.Random, index: int) -> Dict:
    weight = rng.randint(10, 70)
    daily_mg_per_kg = rng.choice([10, 15, 20, 25, 30])
    doses_per_day = rng.choice([2, 3, 4])
    expression = f"{weight} * {daily_mg_per_kg} / {doses_per_day}"
    result = calculator(expression)
    question = (
        f"A {weight} kg patient needs {daily_mg_per_kg} mg/kg/day of a medication divided into "
        f"{doses_per_day} equal doses. How many mg should each dose contain?"
    )
    answer = (
        f"The total daily dose is {weight * daily_mg_per_kg} mg, so each of the {doses_per_day} doses should "
        f"contain {result} mg."
    )
    return {
        "id": f"agent-calculator-{index:05d}",
        "dataset": "Synthetic-MedCalc",
        "tool_type": "calculator",
        "format_version": "legacy_trace",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_THOUGHTS)}",
                        calculator_action(expression),
                        f"Observation: {result}",
                        f"Thought: {rng.choice(CALCULATOR_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def build_multiturn_divided_daily_dose_sample(rng: random.Random, index: int) -> Dict:
    weight = rng.randint(10, 70)
    daily_mg_per_kg = rng.choice([10, 15, 20, 25, 30])
    doses_per_day = rng.choice([2, 3, 4])
    expression = f"{weight} * {daily_mg_per_kg} / {doses_per_day}"
    result = calculator(expression)
    question = (
        f"A {weight} kg patient needs {daily_mg_per_kg} mg/kg/day of a medication divided into "
        f"{doses_per_day} equal doses. How many mg should each dose contain?"
    )
    answer = (
        f"The total daily dose is {weight * daily_mg_per_kg} mg, so each of the {doses_per_day} doses should "
        f"contain {result} mg."
    )
    return {
        "id": f"agent-calculator-{index:05d}",
        "dataset": "Synthetic-MedCalc",
        "tool_type": "calculator",
        "format_version": "react_multiturn",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_THOUGHTS)}",
                        calculator_action(expression),
                    ]
                ),
            },
            {"from": "tool", "value": str(result)},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def build_legacy_bmi_sample(rng: random.Random, index: int) -> Dict:
    weight = rng.randint(45, 110)
    height = rng.choice([1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90])
    expression = f"round({weight} / ({height} ** 2), 1)"
    result = calculator(expression)
    question = f"A patient weighs {weight} kg and is {height:.2f} m tall. What is the BMI?"
    answer = f"The BMI is {result} kg/m^2."
    return {
        "id": f"agent-calculator-{index:05d}",
        "dataset": "Synthetic-MedCalc",
        "tool_type": "calculator",
        "format_version": "legacy_trace",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_THOUGHTS)}",
                        calculator_action(expression),
                        f"Observation: {result}",
                        f"Thought: {rng.choice(CALCULATOR_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def build_multiturn_bmi_sample(rng: random.Random, index: int) -> Dict:
    weight = rng.randint(45, 110)
    height = rng.choice([1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90])
    expression = f"round({weight} / ({height} ** 2), 1)"
    result = calculator(expression)
    question = f"A patient weighs {weight} kg and is {height:.2f} m tall. What is the BMI?"
    answer = f"The BMI is {result} kg/m^2."
    return {
        "id": f"agent-calculator-{index:05d}",
        "dataset": "Synthetic-MedCalc",
        "tool_type": "calculator",
        "format_version": "react_multiturn",
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "conversations": [
            {"from": "human", "value": question},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_THOUGHTS)}",
                        calculator_action(expression),
                    ]
                ),
            },
            {"from": "tool", "value": str(result)},
            {
                "from": "assistant",
                "value": "\n".join(
                    [
                        f"Thought: {rng.choice(CALCULATOR_FOLLOWUPS)}",
                        f"Final Answer: {answer}",
                    ]
                ),
            },
        ],
    }


def make_calculator_samples(num_samples: int, seed: int, format_version: str) -> List[Dict]:
    rng = random.Random(seed)
    if format_version == "react_multiturn":
        builders = [
            build_multiturn_weight_based_dose_sample,
            build_multiturn_divided_daily_dose_sample,
            build_multiturn_bmi_sample,
        ]
    else:
        builders = [
            build_legacy_weight_based_dose_sample,
            build_legacy_divided_daily_dose_sample,
            build_legacy_bmi_sample,
        ]

    samples: List[Dict] = []
    for index in range(num_samples):
        builder = builders[index % len(builders)]
        samples.append(builder(rng, index))
    return samples


def split_records(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        val_size = max(1, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def save_json(path: Path, data: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, data: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in data:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    records = load_records(args.source_file)

    rng = random.Random(args.seed)
    selected_records = records[:]
    rng.shuffle(selected_records)
    selected_records = selected_records[: min(args.max_retrieve_samples, len(selected_records))]

    retrieve_builder: Callable[[Dict, random.Random, int], Dict]
    if args.format_version == "react_multiturn":
        retrieve_builder = build_multiturn_retrieve_sample
    else:
        retrieve_builder = build_legacy_retrieve_sample

    retrieve_samples = [
        retrieve_builder(record=record, rng=rng, index=index)
        for index, record in enumerate(selected_records)
    ]
    calculator_samples = make_calculator_samples(
        args.num_calculator_samples,
        seed=args.seed,
        format_version=args.format_version,
    )

    all_samples = retrieve_samples + calculator_samples
    train_samples, val_samples = split_records(all_samples, val_ratio=args.val_ratio, seed=args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "agent_posttrain_all.json"
    train_path = args.output_dir / "agent_posttrain_train.json"
    val_path = args.output_dir / "agent_posttrain_val.json"
    all_jsonl_path = args.output_dir / "agent_posttrain_all.jsonl"
    train_jsonl_path = args.output_dir / "agent_posttrain_train.jsonl"
    val_jsonl_path = args.output_dir / "agent_posttrain_val.jsonl"
    stats_path = args.output_dir / "agent_posttrain_stats.json"

    save_json(all_path, all_samples)
    save_json(train_path, train_samples)
    save_json(val_path, val_samples)
    save_jsonl(all_jsonl_path, all_samples)
    save_jsonl(train_jsonl_path, train_samples)
    save_jsonl(val_jsonl_path, val_samples)

    stats = {
        "source_file": str(args.source_file),
        "format_version": args.format_version,
        "used_external_llm_api": False,
        "note": "Template-generated locally. The react_multiturn format keeps tool observations outside assistant turns so training matches the runtime protocol.",
        "num_retrieve_samples": len(retrieve_samples),
        "num_calculator_samples": len(calculator_samples),
        "total_samples": len(all_samples),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "system_prompt": AGENT_SYSTEM_PROMPT,
        "output_files": {
            "all": str(all_path),
            "train": str(train_path),
            "val": str(val_path),
            "all_jsonl": str(all_jsonl_path),
            "train_jsonl": str(train_jsonl_path),
            "val_jsonl": str(val_jsonl_path),
        },
    }
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    print(f"Format version: {args.format_version}")
    print(f"Retrieve samples: {len(retrieve_samples)}")
    print(f"Calculator samples: {len(calculator_samples)}")
    print(f"Train size: {len(train_samples)}")
    print(f"Val size: {len(val_samples)}")
    print(f"Saved train file to: {train_path}")
    print(f"Saved val file to: {val_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
