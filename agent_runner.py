import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from adapter import generate_from_messages, load_model_and_tokenizer, unload_model
from agent_config import AGENT_SYSTEM_PROMPT
from tools import execute_tool, parse_action


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "Qwen3-0.6B"
DEFAULT_BASE_ADAPTER_PATH = ROOT_DIR / "outputs" / "qwen3_0.6b_medquad_lora_v2_seq768"
DEFAULT_AGENT_DATA_PATH = ROOT_DIR / "data" / "MedQuad-MedicalQnADataset" / "agent_posttrain_v1" / "agent_posttrain_val.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or replay local tool-calling agent trajectories.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Base model path.")
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_BASE_ADAPTER_PATH,
        help="Optional adapter path for the model used to generate actions.",
    )
    parser.add_argument("--query", type=str, default="", help="User query for agent loop.")
    parser.add_argument("--system-prompt", type=str, default=AGENT_SYSTEM_PROMPT, help="Agent system prompt.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum Thought/Action/Observation iterations.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per step.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load model in 4-bit mode.")
    parser.add_argument("--base-only", action="store_true", help="Ignore the adapter when generating.")
    parser.add_argument(
        "--replay-sample-id",
        type=str,
        default="",
        help="Replay a gold tool trajectory from the generated agent dataset.",
    )
    parser.add_argument(
        "--agent-data-file",
        type=Path,
        default=DEFAULT_AGENT_DATA_PATH,
        help="Agent dataset file used for replay mode.",
    )
    return parser.parse_args()


def extract_action_prefix(text: str) -> Optional[str]:
    match = re.search(r"(Thought:.*?Action:\s*[^\n]+)", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def extract_final_answer(text: str) -> Optional[str]:
    match = re.search(r"Final Answer:\s*(.*)", text, flags=re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def run_agent_loop(
    query: str,
    model,
    tokenizer,
    system_prompt: str,
    max_steps: int,
    max_new_tokens: int,
) -> Dict:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query.strip()})
    steps: List[Dict] = []

    for step_index in range(max_steps):
        generation = generate_from_messages(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        final_answer = extract_final_answer(generation)
        if final_answer:
            return {
                "completed": True,
                "steps": steps,
                "final_answer": final_answer,
                "last_generation": generation,
            }

        action_prefix = extract_action_prefix(generation)
        if not action_prefix:
            return {
                "completed": False,
                "steps": steps,
                "final_answer": "",
                "last_generation": generation,
                "error": "Model did not emit a parsable Action block.",
            }

        action_line_match = re.search(r"Action:\s*[^\n]+", action_prefix)
        action_line = action_line_match.group(0) if action_line_match else ""
        tool_name, tool_args = parse_action(action_line)
        if not tool_name:
            return {
                "completed": False,
                "steps": steps,
                "final_answer": "",
                "last_generation": generation,
                "error": "Failed to parse Action line.",
            }

        observation = execute_tool(tool_name, tool_args)
        steps.append(
            {
                "step": step_index + 1,
                "generation": generation,
                "action_prefix": action_prefix,
                "action_line": action_line,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "observation": observation,
            }
        )
        messages.append({"role": "assistant", "content": action_prefix})
        messages.append({"role": "tool", "content": observation})

    return {
        "completed": False,
        "steps": steps,
        "final_answer": "",
        "last_generation": steps[-1]["generation"] if steps else "",
        "error": f"Reached max_steps={max_steps} without Final Answer.",
    }


def load_replay_record(path: Path, sample_id: str) -> Dict:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    for record in data:
        if record["id"] == sample_id:
            return record
    raise ValueError(f"Could not find sample id '{sample_id}' in {path}.")


def replay_sample(record: Dict) -> Dict:
    assistant_text = record["conversations"][1]["value"]
    action_lines = re.findall(r"Action:\s*.*", assistant_text)
    replay_steps: List[Dict] = []
    for idx, line in enumerate(action_lines, start=1):
        tool_name, tool_args = parse_action(line)
        observation = execute_tool(tool_name, tool_args) if tool_name else "Error: could not parse action"
        replay_steps.append(
            {
                "step": idx,
                "action": line,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "observation": observation,
            }
        )

    final_answer = extract_final_answer(assistant_text) or ""
    return {
        "query": record["conversations"][0]["value"],
        "tool_type": record.get("tool_type", "unknown"),
        "steps": replay_steps,
        "final_answer": final_answer,
        "completed": True,
    }


def main() -> None:
    args = parse_args()

    if args.replay_sample_id:
        result = replay_sample(load_replay_record(args.agent_data_file, args.replay_sample_id))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not args.query:
        raise ValueError("Provide --query for agent generation mode, or use --replay-sample-id.")

    adapter_path = None if args.base_only else args.adapter_path
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        adapter_path=adapter_path,
        trust_remote_code=args.trust_remote_code,
        load_in_4bit=args.load_in_4bit,
    )
    result = run_agent_loop(
        query=args.query,
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
    )
    unload_model(model)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
