import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from releaseops_arena.baselines import naive_baseline, rule_baseline
from releaseops_arena.tool_env import ReleaseOpsToolEnv


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_rollout(env, reward, initial_budget):
    reason = env.state.get("terminal_reason")
    return {
        "reward": reward,
        "safe_ship": 1 if reason == "safe_ship" else 0,
        "unsafe_ship": 1 if reason == "unsafe_ship" else 0,
        "missed_deadline": 1 if reason == "missed_deadline" else 0,
        "false_blocks": env.metrics.get("false_blocks", 0),
        "true_blocks": env.metrics.get("true_blocks", 0),
        "invalid_actions": env.metrics.get("invalid_actions", 0),
        "budget_spent": initial_budget - env.state["review_budget_remaining"],
    }


def aggregate(rows):
    if not rows:
        return {
            "avg_reward": 0.0,
            "safe_ship_rate": 0.0,
            "unsafe_ship_rate": 0.0,
            "missed_deadline_rate": 0.0,
            "avg_false_blocks": 0.0,
            "avg_true_blocks": 0.0,
            "avg_invalid_actions": 0.0,
            "avg_budget_spent": 0.0,
        }

    return {
        "avg_reward": round(sum(row["reward"] for row in rows) / len(rows), 3),
        "safe_ship_rate": round(sum(row["safe_ship"] for row in rows) / len(rows), 3),
        "unsafe_ship_rate": round(sum(row["unsafe_ship"] for row in rows) / len(rows), 3),
        "missed_deadline_rate": round(sum(row["missed_deadline"] for row in rows) / len(rows), 3),
        "avg_false_blocks": round(sum(row["false_blocks"] for row in rows) / len(rows), 3),
        "avg_true_blocks": round(sum(row["true_blocks"] for row in rows) / len(rows), 3),
        "avg_invalid_actions": round(sum(row["invalid_actions"] for row in rows) / len(rows), 3),
        "avg_budget_spent": round(sum(row["budget_spent"] for row in rows) / len(rows), 3),
    }


def run_slice(rows: list[dict]):
    naive_rows = []
    rule_rows = []
    phase_aware_rule_rows = []

    for kwargs in rows:
        env_naive = ReleaseOpsToolEnv()
        env_naive.reset(**kwargs)
        naive_initial_budget = env_naive.state["review_budget_remaining"]
        naive_reward = naive_baseline(env_naive)
        naive_rows.append(summarize_rollout(env_naive, naive_reward, naive_initial_budget))

        env_rule = ReleaseOpsToolEnv()
        env_rule.reset(**kwargs)
        rule_initial_budget = env_rule.state["review_budget_remaining"]
        rule_reward = rule_baseline(env_rule)
        rule_rows.append(summarize_rollout(env_rule, rule_reward, rule_initial_budget))

        # Imported here to keep baseline selection explicit in evaluation output.
        from releaseops_arena.baselines import phase_aware_rule_baseline

        env_phase = ReleaseOpsToolEnv()
        env_phase.reset(**kwargs)
        phase_initial_budget = env_phase.state["review_budget_remaining"]
        phase_reward = phase_aware_rule_baseline(env_phase)
        phase_aware_rule_rows.append(
            summarize_rollout(env_phase, phase_reward, phase_initial_budget)
        )

    return {
        "count": len(rows),
        "naive": aggregate(naive_rows),
        "rule": aggregate(rule_rows),
        "phase_aware_rule": aggregate(phase_aware_rule_rows),
    }


def load_eval_slices():
    seen_path = Path("training/data/eval_seen.jsonl")
    unseen_path = Path("training/data/eval_unseen.jsonl")

    seen_rows = load_jsonl(seen_path)
    unseen_rows = load_jsonl(unseen_path)

    if seen_rows or unseen_rows:
        return seen_rows, unseen_rows

    # Backward-compatible fallback: split eval.jsonl by family.
    all_rows = load_jsonl(Path("training/data/eval.jsonl"))
    unseen_family = "release_manager_ship_before_evidence"
    seen_rows = [row for row in all_rows if row.get("family") != unseen_family]
    unseen_rows = [row for row in all_rows if row.get("family") == unseen_family]
    return seen_rows, unseen_rows


def run_eval():
    print("Evaluating baselines on seen and unseen slices...")

    seen_rows, unseen_rows = load_eval_slices()
    all_rows = seen_rows + unseen_rows

    seen_results = run_slice(seen_rows)
    unseen_results = run_slice(unseen_rows)
    overall_results = run_slice(all_rows)

    results = {
        "seen": seen_results,
        "unseen": unseen_results,
        "overall": overall_results,
        # Legacy keys retained for scripts that expect top-level aggregates.
        "naive": overall_results["naive"],
        "rule": overall_results["rule"],
        "naive_avg": overall_results["naive"]["avg_reward"],
        "rule_avg": overall_results["rule"]["avg_reward"],
    }

    print("Seen slice:")
    print(json.dumps(seen_results, indent=2))
    print("Unseen slice:")
    print(json.dumps(unseen_results, indent=2))
    print("Overall:")
    print(json.dumps(overall_results, indent=2))

    output_path = Path("outputs/eval_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Wrote evaluation results to {output_path}")


if __name__ == "__main__":
    run_eval()
