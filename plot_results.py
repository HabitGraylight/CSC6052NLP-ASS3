import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "NLP_course_Assignment_3_Template" / "figures"


def load_json(path: Path):
    return json.loads(path.read_text())


def ensure_dir() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def annotate_bars(ax, fmt: str = "{:.3f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if height is None:
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points",
        )


def plot_medquad_sft_metrics() -> None:
    data = load_json(ROOT / "outputs" / "medquad_eval_base_vs_adapter_v2_32.json")["summary"]
    metrics = ["rouge_l_f1", "token_f1"]
    labels = ["ROUGE-L F1", "Token F1"]
    base = [data["base"][m] for m in metrics]
    adapter = [data["adapter"][m] for m in metrics]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = range(len(metrics))
    width = 0.34
    ax.bar([i - width / 2 for i in x], base, width=width, label="Base Qwen3-0.6B", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], adapter, width=width, label="MedQuad SFT Adapter", color="#F58518")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(base + adapter) * 1.35)
    ax.set_title("Medical QA Adaptation Improves ROUGE-L")
    ax.legend(frameon=True)
    annotate_bars(ax)
    save(fig, "medquad_sft_metrics")


def plot_agent_tool_metrics() -> None:
    data = load_json(ROOT / "outputs" / "agent_eval_v1_vs_v2.json")["summary"]
    metrics = [
        "tool_call_rate",
        "tool_execution_success_rate",
        "tool_choice_accuracy",
        "completion_rate",
    ]
    labels = ["Tool Call", "Exec Success", "Tool Choice", "Completion"]
    v1 = [data["v1_legacy"]["tool_metrics"][m] for m in metrics]
    v2 = [data["v2_react"]["tool_metrics"][m] for m in metrics]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = range(len(metrics))
    width = 0.34
    ax.bar([i - width / 2 for i in x], v1, width=width, label="Legacy Trace Agent", color="#72B7B2")
    ax.bar([i + width / 2 for i in x], v2, width=width, label="React Multi-turn Agent", color="#E45756")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1.08)
    ax.set_title("Real Multi-turn Training Increases Tool-use Behavior")
    ax.legend(frameon=True, loc="upper left")
    annotate_bars(ax, "{:.2f}")
    save(fig, "agent_tool_metrics")


def _eval_points(trainer_state_path: Path):
    data = load_json(trainer_state_path)
    rows = [row for row in data["log_history"] if "eval_loss" in row]
    return [row["step"] for row in rows], [row["eval_loss"] for row in rows]


def plot_training_curves() -> None:
    medquad_steps, medquad_loss = _eval_points(
        ROOT / "outputs" / "qwen3_0.6b_medquad_lora_v2_seq768" / "checkpoint-192" / "trainer_state.json"
    )
    agent_v1_steps, agent_v1_loss = _eval_points(
        ROOT / "outputs" / "qwen3_0.6b_agent_lora_v1" / "checkpoint-89" / "trainer_state.json"
    )
    agent_v2_steps, agent_v2_loss = _eval_points(
        ROOT / "outputs" / "qwen3_0.6b_agent_lora_v2_react" / "checkpoint-89" / "trainer_state.json"
    )

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(medquad_steps, medquad_loss, marker="o", linewidth=2.2, color="#4C78A8", label="MedQuad SFT")
    ax.plot(agent_v1_steps, agent_v1_loss, marker="o", linewidth=2.2, color="#54A24B", label="Agent v1 Legacy")
    ax.plot(agent_v2_steps, agent_v2_loss, marker="o", linewidth=2.2, color="#E45756", label="Agent v2 React")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss Curves Across Training Stages")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.35)
    save(fig, "training_curves")


def plot_agent_length_imbalance() -> None:
    stats = {
        "Action Turn\n(mean)": 45.86402266288952,
        "Tool Obs.\n(mean)": 220.58215297450425,
        "Final Turn\n(mean)": 245.06090651558074,
        "Action Turn\n(p50)": 48,
        "Tool Obs.\n(p50)": 265,
        "Final Turn\n(p50)": 158,
    }

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    labels = list(stats.keys())
    values = list(stats.values())
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#4C78A8", "#72B7B2", "#F58518"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Tokens")
    ax.set_title("Action Supervision Is Much Shorter Than Final-answer Supervision")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    annotate_bars(ax, "{:.0f}")
    save(fig, "agent_length_imbalance")


def main() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    ensure_dir()
    plot_medquad_sft_metrics()
    plot_agent_tool_metrics()
    plot_training_curves()
    plot_agent_length_imbalance()
    print(f"Saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()
