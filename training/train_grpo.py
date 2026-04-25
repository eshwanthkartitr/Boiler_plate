"""
GRPO training entrypoint for ReleaseOps Arena.
Supports quick smoke runs plus configurable full runs.
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from releaseops_arena.tool_env import ReleaseOpsToolEnv


class ReleaseOpsGRPOEnv:
    """Training wrapper that exposes only model-facing ReleaseOps tools to TRL."""

    def __init__(self):
        self.env = ReleaseOpsToolEnv()
        self.reward = 0.0

    def _sync_reward(self):
        self.reward = self.env.reward

    def reset(self, **kwargs) -> str:
        observation = self.env.reset(**kwargs)
        self._sync_reward()
        return observation

    def inspect_pr_diff(self, pr_id: str) -> str:
        """
        Inspect a pull request diff for risky edits, disabled tests, or safety-rule violations.

        Args:
            pr_id: Pull request id from the current observation refs, such as pr_482.

        Returns:
            Compact evidence about changed files, risky edits, and disabled tests.
        """
        result = self.env.inspect_pr_diff(pr_id)
        self._sync_reward()
        return result

    def inspect_ci_run(self, run_id: str) -> str:
        """
        Inspect CI output for skipped tests, failing checks, or deploy gate status.

        Args:
            run_id: CI run id from the current observation refs, such as ci_run_991.

        Returns:
            Compact evidence about CI status, skipped tests, and deploy checks.
        """
        result = self.env.inspect_ci_run(run_id)
        self._sync_reward()
        return result

    def inspect_ticket(self, ticket_id: str) -> str:
        """
        Inspect a ticket for issue severity, scope, and release risk.

        Args:
            ticket_id: Ticket id from the current observation refs, such as ticket_317.

        Returns:
            Compact evidence about issue severity, status, and release relevance.
        """
        result = self.env.inspect_ticket(ticket_id)
        self._sync_reward()
        return result

    def ask_worker(
        self,
        worker_id: str,
        question_type: Literal[
            "evidence_basis", "policy_basis", "confidence_basis", "risk_summary"
        ],
    ) -> str:
        """
        Ask a specialist worker for a compact explanation of evidence, policy, confidence, or risk.

        Args:
            worker_id: Worker id from the current observation worker_stats.
            question_type: One of evidence_basis, policy_basis, confidence_basis, or risk_summary.

        Returns:
            The worker's answer as compact JSON.
        """
        result = self.env.ask_worker(worker_id, question_type)
        self._sync_reward()
        return result

    def approve_proposal(self, proposal_id: str) -> str:
        """
        Approve an unresolved proposal when its evidence is sufficient and safety rules are not violated.

        Args:
            proposal_id: Proposal id from the current observation proposals, such as p17.

        Returns:
            JSON describing the approval and whether the episode ended.
        """
        result = self.env.approve_proposal(proposal_id)
        self._sync_reward()
        return result

    def block_proposal(
        self,
        proposal_id: str,
        rule_id: Literal[
            "S1",
            "S2",
            "S3",
            "no_p1_open",
            "payments_tests_must_pass",
            "deploy_checks_before_ship",
        ],
    ) -> str:
        """
        Block an unresolved proposal that violates or likely violates a release safety rule.

        Args:
            proposal_id: Proposal id from the current observation proposals, such as p17.
            rule_id: Safety rule that justifies the block. Use S1, S2, or S3 when possible.

        Returns:
            JSON describing the block and whether the episode ended.
        """
        result = self.env.block_proposal(proposal_id, rule_id)
        self._sync_reward()
        return result

    def hold_release(
        self,
        reason_code: Literal[
            "awaiting_fix", "risk_too_high", "policy_unclear", "deploy_checks_pending"
        ],
    ) -> str:
        """
        Delay the release when remaining unresolved risk is too high or required checks are unclear.

        Args:
            reason_code: Reason for holding the release gate.

        Returns:
            JSON containing the updated observation or terminal deadline status.
        """
        result = self.env.hold_release(reason_code)
        self._sync_reward()
        return result


def reward_func(environments, **kwargs) -> list[float]:
    """Extract reward directly from each environment instance."""
    return [env.reward for env in environments]


def compatibility_reward_func(prompts, completions, **kwargs) -> list[float]:
    """Fallback reward for legacy TRL versions without OpenEnv integration."""
    rewards = []
    tool_tokens = {
        "inspect_pr_diff",
        "inspect_ci_run",
        "inspect_ticket",
        "ask_worker",
        "approve_proposal",
        "block_proposal",
        "hold_release",
    }

    for completion in completions:
        if isinstance(completion, list):
            text = " ".join(part.get("content", "") for part in completion if isinstance(part, dict))
        else:
            text = str(completion)

        lower = text.lower()
        reward = 0.0
        if any(token in lower for token in tool_tokens):
            reward += 0.25
        if "inspect_" in lower:
            reward += 0.1
        if "invalid" in lower:
            reward -= 0.2
        if "\"thought\"" in lower:
            reward -= 0.1
        if len(text) > 1200:
            reward -= 0.1

        rewards.append(reward)

    return rewards


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReleaseOps supervisor with GRPO.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", default="training/data/train.jsonl")
    parser.add_argument("--output-dir", default="outputs/releaseops-grpo")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Apply tiny run defaults suitable for quick validation.",
    )
    parser.add_argument(
        "--metrics-json",
        default="outputs/grpo_smoke_metrics.json",
        help="Path to save trainer log history JSON.",
    )
    parser.add_argument(
        "--allow-compatibility-reward",
        action="store_true",
        help=(
            "Allow legacy text-only reward training when installed TRL does not support "
            "environment_factory. Do not use this for final OpenEnv GRPO results."
        ),
    )
    return parser.parse_args()


def build_config(args) -> GRPOConfig:
    if args.smoke:
        max_steps = min(args.max_steps, 8)
        num_generations = min(args.num_generations, 2)
        gradient_accumulation_steps = min(args.gradient_accumulation_steps, 2)
        logging_steps = 1
        output_dir = f"{args.output_dir}-smoke"
    else:
        max_steps = args.max_steps
        num_generations = args.num_generations
        gradient_accumulation_steps = args.gradient_accumulation_steps
        logging_steps = args.logging_steps
        output_dir = args.output_dir

    config_kwargs = dict(
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_generations=num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=max_steps,
        logging_steps=logging_steps,
        bf16=False,
        seed=args.seed,
    )

    if "env_kwargs_keys" in inspect.signature(GRPOConfig.__init__).parameters:
        config_kwargs["env_kwargs_keys"] = ["family", "seed", "difficulty", "archetype_mix"]

    return GRPOConfig(**config_kwargs)


def ensure_dataset(train_file: str):
    if os.path.exists(train_file):
        return

    print(f"Dataset not found at {train_file}. Generating...", flush=True)
    python_executable = Path(sys.executable)
    os.system(f"{python_executable} training/make_dataset.py")


def save_metrics(log_history, metrics_json: str):
    output_path = Path(metrics_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(log_history, handle, indent=2)
    print(f"Saved trainer log history to {output_path}", flush=True)


def summarize_log_history(log_history):
    reward_rows = [row for row in log_history if "reward" in row]
    if not reward_rows:
        return {
            "reward_points": 0,
            "reward_first": None,
            "reward_last": None,
            "reward_delta": None,
        }

    reward_first = reward_rows[0]["reward"]
    reward_last = reward_rows[-1]["reward"]
    return {
        "reward_points": len(reward_rows),
        "reward_first": reward_first,
        "reward_last": reward_last,
        "reward_delta": reward_last - reward_first,
    }


def main():
    args = parse_args()
    ensure_dataset(args.train_file)

    trainer_signature = inspect.signature(GRPOTrainer.__init__).parameters
    supports_env_factory = "environment_factory" in trainer_signature
    supports_env_kwargs = "env_kwargs_keys" in inspect.signature(GRPOConfig.__init__).parameters

    if not supports_env_factory and not args.allow_compatibility_reward:
        raise RuntimeError(
            "Installed TRL does not support OpenEnv GRPO: GRPOTrainer.__init__ has no "
            "'environment_factory' parameter. Install a TRL version with OpenEnv support "
            '(for example the version documented at https://huggingface.co/docs/trl/openenv), '
            "or pass --allow-compatibility-reward for a text-only smoke test that must not "
            "be reported as environment RL."
        )

    if supports_env_factory and not supports_env_kwargs:
        print(
            "Warning: installed TRL supports environment_factory but GRPOConfig does not expose "
            "env_kwargs_keys; dataset scenario columns may not be passed into reset(**kwargs).",
            flush=True,
        )

    dataset = load_dataset("json", data_files={"train": args.train_file})
    training_args = build_config(args)

    print("Initializing GRPO Trainer...", flush=True)
    trainer_kwargs = {
        "model": args.model_name,
        "args": training_args,
        "train_dataset": dataset["train"],
    }

    if supports_env_factory:
        print("Using OpenEnv GRPO mode (environment_factory available).", flush=True)
        trainer_kwargs["reward_funcs"] = [reward_func]
        trainer_kwargs["environment_factory"] = ReleaseOpsGRPOEnv
    else:
        print(
            "OpenEnv GRPO mode unavailable in installed TRL; running compatibility smoke mode.",
            flush=True,
        )
        trainer_kwargs["reward_funcs"] = [compatibility_reward_func]

    trainer = GRPOTrainer(**trainer_kwargs)

    print("Starting GRPO training run...", flush=True)
    trainer.train()

    log_history = trainer.state.log_history
    save_metrics(log_history, args.metrics_json)

    summary = summarize_log_history(log_history)
    print("Training reward trend summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
