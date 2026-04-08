import argparse
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Dict, List


ROOT_DIR = Path(__file__).resolve().parent
SOURCE_DATA_PATH = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "processed" / "medquad_agent_all.json"
KB_DIR = ROOT_DIR / "knowledge_base"
KB_DOCS_PATH = KB_DIR / "medquad_docs.jsonl"


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def load_source_records(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}, but got {type(data).__name__}.")
    return data


def build_kb_documents(records: List[Dict]) -> List[Dict]:
    documents: List[Dict] = []
    for record in records:
        question = record["conversations"][0]["value"]
        answer = record["conversations"][1]["value"]
        doc = {
            "id": record["id"],
            "qtype": record.get("qtype", "unknown"),
            "question": question,
            "answer": answer,
            "text": f"Question: {question}\nAnswer: {answer}",
        }
        documents.append(doc)
    return documents


def save_documents(path: Path, documents: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")


def build_knowledge_base(source_path: Path = SOURCE_DATA_PATH, output_path: Path = KB_DOCS_PATH) -> Path:
    records = load_source_records(source_path)
    documents = build_kb_documents(records)
    save_documents(output_path, documents)
    load_documents.cache_clear()
    build_runtime_index.cache_clear()
    return output_path


def ensure_knowledge_base(source_path: Path = SOURCE_DATA_PATH, output_path: Path = KB_DOCS_PATH) -> Path:
    if not output_path.exists():
        return build_knowledge_base(source_path=source_path, output_path=output_path)
    return output_path


@lru_cache(maxsize=1)
def load_documents(path: str = str(KB_DOCS_PATH)) -> List[Dict]:
    kb_path = ensure_knowledge_base(output_path=Path(path))
    documents: List[Dict] = []
    with kb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            documents.append(json.loads(line))
    return documents


@lru_cache(maxsize=1)
def build_runtime_index(path: str = str(KB_DOCS_PATH)) -> Dict:
    documents = load_documents(path)
    term_frequencies: List[Counter] = []
    document_frequencies: Counter = Counter()
    document_lengths: List[int] = []

    for doc in documents:
        tokens = tokenize(doc["text"])
        frequencies = Counter(tokens)
        term_frequencies.append(frequencies)
        document_lengths.append(len(tokens))
        document_frequencies.update(frequencies.keys())

    num_docs = len(documents)
    avg_doc_len = sum(document_lengths) / max(num_docs, 1)
    idf = {
        term: math.log(1 + (num_docs - freq + 0.5) / (freq + 0.5))
        for term, freq in document_frequencies.items()
    }
    return {
        "documents": documents,
        "term_frequencies": term_frequencies,
        "document_lengths": document_lengths,
        "avg_doc_len": avg_doc_len,
        "idf": idf,
    }


def bm25_score(query_tokens: List[str], doc_freqs: Counter, doc_len: int, avg_doc_len: float, idf: Dict[str, float]) -> float:
    k1 = 1.5
    b = 0.75
    score = 0.0
    for token in query_tokens:
        if token not in doc_freqs or token not in idf:
            continue
        tf = doc_freqs[token]
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1e-8)))
        score += idf[token] * numerator / denominator
    return score


def search_knowledge_base(query: str, top_k: int = 3, docs_path: Path = KB_DOCS_PATH) -> List[Dict]:
    index = build_runtime_index(str(docs_path))
    query_tokens = tokenize(query)
    scored: List[Dict] = []

    for doc, freqs, doc_len in zip(index["documents"], index["term_frequencies"], index["document_lengths"]):
        score = bm25_score(query_tokens, freqs, doc_len, index["avg_doc_len"], index["idf"])
        if score <= 0:
            continue
        scored.append({"score": score, **doc})

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def format_retrieval_results(results: List[Dict], max_chars_per_doc: int = 500) -> str:
    if not results:
        return "[Retrieved] No relevant documents found in the local knowledge base."

    blocks = []
    for idx, result in enumerate(results, start=1):
        answer = result["answer"][:max_chars_per_doc].strip()
        blocks.append(
            f"[Retrieved {idx}] id={result['id']} qtype={result['qtype']}\n"
            f"Question: {result['question']}\n"
            f"Answer: {answer}"
        )
    return "\n\n".join(blocks)


def retrieve(query: str, top_k: int = 3, docs_path: Path = KB_DOCS_PATH) -> str:
    return format_retrieval_results(search_knowledge_base(query=query, top_k=top_k, docs_path=docs_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple local RAG over MedQuad using BM25-style retrieval.")
    parser.add_argument("--build", action="store_true", help="Build the local knowledge base.")
    parser.add_argument("--query", type=str, default="", help="Query the knowledge base.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to return.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build:
        output_path = build_knowledge_base()
        print(f"Knowledge base saved to: {output_path}")
        return

    if args.query:
        ensure_knowledge_base()
        print(retrieve(args.query, top_k=args.top_k))
        return

    raise ValueError("Provide --build or --query.")


if __name__ == "__main__":
    main()
