import os
from pathlib import Path

# Configure the mirror before importing Hugging Face libraries.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HUGGINGFACE_HUB_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download


REPO_ID = "Qwen/Qwen3-0.6B"
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models" / "Qwen3-0.6B"


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    local_path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=str(MODEL_DIR),
    )
    print(f"Downloaded {REPO_ID}")
    print(f"Saved model to: {local_path}")


if __name__ == "__main__":
    main()
