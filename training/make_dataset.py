import json
import os
import random
from pathlib import Path

PROMPT = [
    {
        "role": "system",
        "content": (
            "You are the release oversight supervisor. Use tools sparingly. "
            "Gather evidence only when justified. Resolve all active proposals. "
            "Stop once the phase or episode is resolved."
        ),
    },
    {
        "role": "user",
        "content": "A new release phase is starting. Please review the proposals and decide what to inspect or resolve.",
    },
]

FAMILY_CONFIG = {
    "green_ci_disabled_payment_test": {
        "difficulty": "medium",
        "archetype_mix": "shortcut_ci__careful_qa",
    },
    "qa_undercalls_p1_checkout_bug": {
        "difficulty": "medium",
        "archetype_mix": "stale_policy_qa__deadline_rm",
    },
    "release_manager_ship_before_evidence": {
        "difficulty": "medium",
        "archetype_mix": "overconfident_rm__careful_sre",
    },
    "careful_qa_safe": {
        "difficulty": "low",
        "archetype_mix": "careful_qa__expert_rm",
    },
}

TRAIN_FAMILIES = [
    "green_ci_disabled_payment_test",
    "qa_undercalls_p1_checkout_bug",
    "careful_qa_safe",
]
UNSEEN_EVAL_FAMILIES = ["release_manager_ship_before_evidence"]


def create_dataset(output_path: str, num_samples: int, split: str, families: list[str]):
    split_seed = {
        "train": 101,
        "eval_seen": 202,
        "eval_unseen": 303,
    }.get(split, 404)
    rng = random.Random(split_seed)
    samples = []

    for _ in range(num_samples):
        family = rng.choice(families)
        config = FAMILY_CONFIG[family]

        samples.append(
            {
                "prompt": PROMPT,
                "family": family,
                "seed": rng.randint(1000, 9999),
                "difficulty": config["difficulty"],
                "archetype_mix": config["archetype_mix"],
                "split": split,
            }
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample) + "\n")

    print(f"Generated {num_samples} samples for '{split}' split at {output_path}")


def merge_jsonl(output_path: str, input_paths: list[str]):
    rows = []
    for path in input_paths:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as handle:
            rows.extend([json.loads(line) for line in handle if line.strip()])

    with open(output_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    print(f"Merged {len(rows)} rows into {output_path}")


if __name__ == "__main__":
    data_dir = Path("training/data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_path = str(data_dir / "train.jsonl")
    eval_seen_path = str(data_dir / "eval_seen.jsonl")
    eval_unseen_path = str(data_dir / "eval_unseen.jsonl")
    eval_path = str(data_dir / "eval.jsonl")

    create_dataset(train_path, num_samples=120, split="train", families=TRAIN_FAMILIES)
    create_dataset(eval_seen_path, num_samples=30, split="eval_seen", families=TRAIN_FAMILIES)
    create_dataset(eval_unseen_path, num_samples=20, split="eval_unseen", families=UNSEEN_EVAL_FAMILIES)
    merge_jsonl(eval_path, [eval_seen_path, eval_unseen_path])
