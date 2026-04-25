import json
from pathlib import Path

import matplotlib.pyplot as plt


def extract_overall(data):
    if "overall" in data:
        return data["overall"]
    return {
        "naive": data.get("naive", {}),
        "rule": data.get("rule", {}),
    }


def plot_metrics():
    try:
        input_path = Path("outputs/eval_results.json")
        with open(input_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        overall = extract_overall(data)
        naive_avg = overall["naive"].get("avg_reward", data.get("naive_avg", 0.0))
        rule_avg = overall["rule"].get("avg_reward", data.get("rule_avg", 0.0))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].bar(["Naive", "Rule"], [naive_avg, rule_avg], color=["#d64545", "#2f6db0"])
        axes[0].set_ylabel("Average Reward")
        axes[0].set_title("Overall Reward")

        seen_safe = data.get("seen", {}).get("rule", {}).get("safe_ship_rate")
        unseen_safe = data.get("unseen", {}).get("rule", {}).get("safe_ship_rate")
        if seen_safe is None or unseen_safe is None:
            seen_safe = overall["rule"].get("safe_ship_rate", 0.0)
            unseen_safe = overall["naive"].get("safe_ship_rate", 0.0)

        axes[1].bar(["Seen", "Unseen"], [seen_safe, unseen_safe], color=["#4f8f4f", "#c58f3a"])
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_ylabel("Safe Ship Rate")
        axes[1].set_title("Rule Baseline Slice Safety")

        fig.tight_layout()
        output_path = Path("outputs/eval_chart.png")
        fig.savefig(output_path)
        print(f"Chart saved to {output_path}")

    except Exception as exc:
        print(f"Error plotting metrics: {exc}")


if __name__ == "__main__":
    plot_metrics()
