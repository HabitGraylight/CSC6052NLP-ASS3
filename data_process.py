import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "medquad.jsonl"
OUTPUT_DIR = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "processed"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([?.!,;:])", r"\1", text)
    return text.strip()


def load_medquad(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            question = normalize_text(row["Question"])
            answer = normalize_text(row["Answer"])
            qtype = normalize_text(row.get("qtype", "unknown"))
            if not question or not answer:
                continue

            rows.append(
                {
                    "split": row.get("split", "train"),
                    "qtype": qtype,
                    "question": question,
                    "answer": answer,
                }
            )
    return rows


def build_conversation_record(row: Dict, index: int) -> Dict:
    return {
        "id": f"medquad-{index:05d}",
        "dataset": "MedQuad",
        "task": "medical_qa",
        "qtype": row["qtype"],
        "conversations": [
            {"from": "human", "value": row["question"]},
            {"from": "assistant", "value": row["answer"]},
        ],
    }


def split_records(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    shuffled = records[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    val_size = max(1, int(len(shuffled) * val_ratio))
    if val_size >= len(shuffled):
        val_size = max(1, len(shuffled) - 1)

    val_records = shuffled[:val_size]
    train_records = shuffled[val_size:]
    return train_records, val_records


def qtype_counts(records: Iterable[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        qtype = record["qtype"]
        counts[qtype] = counts.get(qtype, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def save_json(path: Path, data: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def save_jsonl(path: Path, data: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in data:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert MedQuad into SFT-style conversations.")
    parser.add_argument("--input", type=Path, default=RAW_DATA_PATH, help="Path to the raw MedQuad JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for processed outputs.")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_rows = load_medquad(args.input)
    unique_rows: List[Dict] = []
    seen = set()
    duplicate_count = 0

    for row in raw_rows:
        key = (row["question"], row["answer"])
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        unique_rows.append(row)

    records = [build_conversation_record(row, idx) for idx, row in enumerate(unique_rows)]
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "medquad_agent_all.json"
    train_path = args.output_dir / "medquad_agent_train.json"
    val_path = args.output_dir / "medquad_agent_val.json"
    all_jsonl_path = args.output_dir / "medquad_agent_all.jsonl"
    train_jsonl_path = args.output_dir / "medquad_agent_train.jsonl"
    val_jsonl_path = args.output_dir / "medquad_agent_val.jsonl"
    stats_path = args.output_dir / "medquad_agent_stats.json"

    save_json(all_path, records)
    save_json(train_path, train_records)
    save_json(val_path, val_records)
    save_jsonl(all_jsonl_path, records)
    save_jsonl(train_jsonl_path, train_records)
    save_jsonl(val_jsonl_path, val_records)

    stats = {
        "input_path": str(args.input),
        "total_raw_rows": len(raw_rows),
        "total_unique_rows": len(unique_rows),
        "duplicate_rows_removed": duplicate_count,
        "train_size": len(train_records),
        "val_size": len(val_records),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "top_qtypes": dict(list(qtype_counts(unique_rows).items())[:20]),
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

    print(f"Loaded raw rows: {len(raw_rows)}")
    print(f"Removed duplicates: {duplicate_count}")
    print(f"Saved all records to: {all_path}")
    print(f"Saved train records to: {train_path}")
    print(f"Saved val records to: {val_path}")
    print(f"Saved JSONL records to: {all_jsonl_path}")
    print(f"Saved stats to: {stats_path}")


if __name__ == "__main__":
    main()
