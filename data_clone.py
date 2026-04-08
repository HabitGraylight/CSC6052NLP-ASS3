import json
import os
from pathlib import Path

# Configure the mirror before importing Hugging Face libraries.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

from datasets import DatasetDict, load_dataset


REPO_ID = "keivalya/MedQuad-MedicalQnADataset"
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset"
JSONL_PATH = DATA_DIR / "medquad.jsonl"
def save_dataset(dataset: DatasetDict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(DATA_DIR / "hf_dataset"))

    with JSONL_PATH.open("w", encoding="utf-8") as fh:
        for split_name, split in dataset.items():
            for row in split:
                item = {"split": split_name, **row}
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    dataset = load_dataset(REPO_ID)
    save_dataset(dataset)
    print(f"Downloaded {REPO_ID}")
    print(f"Saved Arrow dataset to: {DATA_DIR / 'hf_dataset'}")
    print(f"Saved JSONL export to: {JSONL_PATH}")


if __name__ == "__main__":
    main()
