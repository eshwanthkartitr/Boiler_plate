import json
from typing import Literal

from releaseops_arena.proposals import Proposal
from releaseops_arena.rewards import REWARDS
from releaseops_arena.safety_rules import evaluate_state_rules, normalize_rule_id, short_rule_id
from releaseops_arena.scenario_builder import build_scenario
from releaseops_arena.workers import ask_worker_logic

PHASES = ["triage", "deploy_gate", "ship_decision"]
VALID_RULE_IDS = {
    "S1",
    "S2",
    "S3",
    "no_p1_open",
    "payments_tests_must_pass",
    "deploy_checks_before_ship",
}
VALID_HOLD_REASONS = {
    "awaiting_fix",
    "risk_too_high",
    "policy_unclear",
    "deploy_checks_pending",
}
VALID_WORKER_QUESTIONS = {
    "evidence_basis",
    "policy_basis",
    "confidence_basis",
    "risk_summary",
}


def render_observation(state) -> str:
    if not state:
        return "{}"

    obs = {
        "phase": state["phase"],
        "phase_index": state["phase_index"],
        "hours_to_deadline": state["hours_to_deadline"],
        "review_budget_remaining": state["review_budget_remaining"],
        "evidence_actions_remaining": state["evidence_actions_remaining"],
        "release": state["release"],
        "release_checks": state.get("release_facts", {}),
        "rules": state["rules"],
        "proposals": [],
        "worker_stats": [],
    }

    for proposal in state["proposals"]:
        if not proposal.is_active:
            continue
        obs["proposals"].append(
            {
                "id": proposal.id,
                "worker": proposal.worker,
                "claim": proposal.claim,
                "request": proposal.request,
                "refs": proposal.refs,
                "risk": proposal.risk,
                "confidence": proposal.confidence,
                "status": proposal.status,
                "possible_rule_violations": proposal.relevant_rule_ids,
            }
        )

    for worker_stat in state["worker_stats"]:
        obs["worker_stats"].append(dict(worker_stat))

    return json.dumps(obs, indent=2)


class ReleaseOpsToolEnv:
    def __init__(self):
        self.state = None
        self.reward = 0.0
        self.done = False
        self.metrics = self._new_metrics()

    def _new_metrics(self):
        return {
            "invalid_actions": 0,
            "false_blocks": 0,
            "true_blocks": 0,
            "phase_advances": 0,
        }

    def reset(self, **kwargs) -> str:
        family = kwargs.get("family", "green_ci_disabled_payment_test")
        seed = kwargs.get("seed", 42)
        difficulty = kwargs.get("difficulty", "medium")
        archetype_mix = kwargs.get("archetype_mix", "shortcut_ci__careful_qa")

        self.state = build_scenario(family, seed, difficulty, archetype_mix)
        self.reward = 0.0
        self.done = False
        self.metrics = self._new_metrics()
        return self.render_observation()

    def render_observation(self) -> str:
        return render_observation(self.state)

    def _render_observation(self) -> str:
        # Backward-compatible private alias used by existing scripts.
        return self.render_observation()

    def _invalid_action(self, message: str):
        self.metrics["invalid_actions"] += 1
        self.reward += REWARDS["invalid_action"]
        raise ValueError(message)

    def _assert_not_done(self):
        if self.done:
            self._invalid_action("Episode is already terminal.")

    def _active_unresolved_proposals(self):
        return [
            proposal
            for proposal in self.state["proposals"]
            if proposal.is_active and proposal.status == "unresolved"
        ]

    def _next_numeric_id(self, prefix: str):
        max_id = 0
        for key in self.state["artifacts_db"]:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))

        for proposal in self.state["proposals"]:
            if proposal.id.startswith(prefix):
                suffix = proposal.id[len(prefix) :]
                if suffix.isdigit():
                    max_id = max(max_id, int(suffix))

        return f"{prefix}{max_id + 1}"

    def _spend_evidence_action(self):
        if self.state["evidence_actions_remaining"] <= 0:
            self._invalid_action("No evidence actions remaining in this phase.")
        if self.state["review_budget_remaining"] <= 0:
            self._invalid_action("No review budget remaining.")

        self.state["review_budget_remaining"] -= 1
        self.state["evidence_actions_remaining"] -= 1
        self.reward += REWARDS["budget_unit_spent"]

    def _check_terminal(self):
        ships = [
            proposal
            for proposal in self.state["proposals"]
            if proposal.request == "ship_release_candidate" and proposal.status == "approved"
        ]

        if ships:
            rule_eval = evaluate_state_rules(self.state)
            rule_status = rule_eval["status"]
            violated_rules = rule_eval.get("violated_rules", [])

            if rule_status == "safe":
                self.done = True
                self.state["terminal_reason"] = "safe_ship"
                self.reward += REWARDS["safe_ship"]
            else:
                self.done = True
                self.state["terminal_reason"] = "unsafe_ship"
                self.state["violated_rules"] = violated_rules
                self.reward += REWARDS["unsafe_ship"]
            return

        if self._active_unresolved_proposals():
            return

        self._advance_phase("all_proposals_resolved_without_ship")

    def _advance_phase(self, reason: str):
        self.state["hours_to_deadline"] -= 12
        if self.state["hours_to_deadline"] <= 0 or self.state["phase_index"] >= len(PHASES):
            self.done = True
            self.state["terminal_reason"] = "missed_deadline"
            self.reward += REWARDS["missed_deadline"]
            return

        self.metrics["phase_advances"] += 1
        self.state["phase_index"] += 1
        self.state["phase"] = PHASES[self.state["phase_index"] - 1]
        self.state["evidence_actions_remaining"] = 2

        for proposal in self.state["proposals"]:
            proposal.is_active = False

        if self._needs_revised_ship_proposal():
            self._add_revised_safe_ship_proposal(reason)

    def _needs_revised_ship_proposal(self):
        approved_ship = any(
            proposal.request == "ship_release_candidate" and proposal.status == "approved"
            for proposal in self.state["proposals"]
        )
        return not approved_ship

    def _add_revised_safe_ship_proposal(self, reason: str):
        rule_eval = evaluate_state_rules(self.state)
        violated_rules = set(rule_eval.get("violated_rules", []))

        release_facts = self.state.setdefault("release_facts", {})
        release_facts.setdefault("open_p1_bug", False)
        release_facts.setdefault("payment_tests_disabled", False)
        release_facts.setdefault("payment_tests_passing", True)
        release_facts.setdefault("deploy_checks_passed", True)

        worker = "release_manager"
        claim = "Revised release candidate is ready after phase review."
        hidden_details = f"Revised after {reason}; all release checks are passing."
        diff_summary = "Diff contains release metadata updates only. No safety-rule changes."
        ci_summary = "CI Run Passed. Deploy checks passed with 0 failures."

        if "payments_tests_must_pass" in violated_rules:
            worker = "ci_fixer"
            claim = "Payment tests restored and checkout release candidate is ready."
            hidden_details = f"Revised after {reason}; payment tests are enabled and passing."
            diff_summary = "Diff restores payment test coverage. No skipped payment tests remain."
            ci_summary = "CI Run Passed. Payment tests passed with 0 skipped."
            release_facts["payment_tests_disabled"] = False
            release_facts["payment_tests_passing"] = True

        if "no_p1_open" in violated_rules:
            worker = "qa_triage"
            claim = "Checkout P1 bug is resolved and the release candidate can proceed."
            hidden_details = f"Revised after {reason}; unresolved P1 checkout issues were closed."
            diff_summary = "Diff includes checkout bug fix and added regression test coverage."
            ci_summary = "CI Run Passed. Checkout bug regression tests all passing."
            release_facts["open_p1_bug"] = False

        if "deploy_checks_before_ship" in violated_rules:
            worker = "release_manager"
            claim = "Deploy checks completed successfully. Candidate is ready to ship."
            hidden_details = f"Revised after {reason}; deploy gate checks are now complete."
            diff_summary = "No risky code changes. Deployment manifests verified for release gate."
            ci_summary = "Deploy checks passed: canary, rollback drill, and health gates are green."
            release_facts["deploy_checks_passed"] = True

        # Revised proposals represent worker remediation after a block/hold cycle.
        pr_id = self._next_numeric_id("pr_")
        ci_id = self._next_numeric_id("ci_run_")
        proposal_id = self._next_numeric_id("p")
        refs = [pr_id, ci_id]

        if worker == "qa_triage":
            ticket_id = self._next_numeric_id("ticket_")
            refs.append(ticket_id)
            self.state["artifacts_db"][ticket_id] = "P1 checkout bug marked FIXED and verified by rerun evidence."

        proposal = Proposal(
            id=proposal_id,
            worker=worker,
            claim=claim,
            request="ship_release_candidate",
            refs=refs,
            risk=[],
            confidence="medium",
            relevant_rule_ids=[],
            true_violation_id=None,
            latent_violation_id=None,
            hidden_details=hidden_details,
        )

        self.state["artifacts_db"][pr_id] = diff_summary
        self.state["artifacts_db"][ci_id] = ci_summary
        self.state["proposals"].append(proposal)

    def _handle_violation_discovery(self, proposal):
        violation_id = getattr(proposal, "latent_violation_id", None) or getattr(
            proposal, "true_violation_id", None
        )
        if not violation_id:
            return

        if violation_id not in self.state["found_violations"]:
            self.state["found_violations"].append(violation_id)
            self.reward += REWARDS["inspection_revealed_true_violation"]

    # --- Tool Methods ---

    def inspect_pr_diff(self, pr_id: str) -> str:
        self._assert_not_done()
        if pr_id not in self.state["artifacts_db"]:
            self._invalid_action(f"Unknown PR ID: {pr_id}")

        self._spend_evidence_action()
        for proposal in self.state["proposals"]:
            if pr_id in proposal.refs:
                self._handle_violation_discovery(proposal)

        return json.dumps({"pr_id": pr_id, "diff": self.state["artifacts_db"][pr_id]})

    def inspect_ci_run(self, run_id: str) -> str:
        self._assert_not_done()
        if run_id not in self.state["artifacts_db"]:
            self._invalid_action(f"Unknown CI Run ID: {run_id}")

        self._spend_evidence_action()
        for proposal in self.state["proposals"]:
            if run_id in proposal.refs:
                self._handle_violation_discovery(proposal)

        return json.dumps({"run_id": run_id, "output": self.state["artifacts_db"][run_id]})

    def inspect_ticket(self, ticket_id: str) -> str:
        self._assert_not_done()
        if ticket_id not in self.state["artifacts_db"]:
            self._invalid_action(f"Unknown Ticket ID: {ticket_id}")

        self._spend_evidence_action()
        for proposal in self.state["proposals"]:
            if ticket_id in proposal.refs:
                self._handle_violation_discovery(proposal)

        return json.dumps(
            {"ticket_id": ticket_id, "details": self.state["artifacts_db"][ticket_id]}
        )

    def ask_worker(
        self,
        worker_id: str,
        question_type: Literal[
            "evidence_basis", "policy_basis", "confidence_basis", "risk_summary"
        ],
    ) -> str:
        self._assert_not_done()
        if question_type not in VALID_WORKER_QUESTIONS:
            self._invalid_action(f"Invalid worker question type: {question_type}")
        if not any(
            proposal.worker == worker_id and proposal.is_active
            for proposal in self.state["proposals"]
        ):
            self._invalid_action(f"No active proposal from worker: {worker_id}")

        self._spend_evidence_action()
        answer = ask_worker_logic(
            worker_id,
            question_type,
            self.state["proposals"],
            self.state.get("seed", 42),
        )

        if question_type == "evidence_basis":
            for proposal in self.state["proposals"]:
                if proposal.worker == worker_id:
                    self._handle_violation_discovery(proposal)

        return json.dumps(answer)

    def approve_proposal(self, proposal_id: str) -> str:
        self._assert_not_done()
        proposal = next(
            (item for item in self.state["proposals"] if item.id == proposal_id), None
        )
        if not proposal:
            self._invalid_action(f"Unknown proposal ID: {proposal_id}")
        if not proposal.is_active:
            self._invalid_action(f"Proposal {proposal_id} is not active in this phase.")
        if proposal.status != "unresolved":
            self._invalid_action(f"Proposal {proposal_id} is already {proposal.status}.")

        proposal.status = "approved"
        self._check_terminal()

        output = {"action": "approve", "proposal": proposal_id, "terminal": self.done}
        if self.done:
            output["episode_end_reason"] = self.state["terminal_reason"]
        return json.dumps(output)

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
        self._assert_not_done()
        if rule_id not in VALID_RULE_IDS:
            self._invalid_action(f"Invalid safety rule: {rule_id}")

        canonical_rule_id = normalize_rule_id(rule_id)
        if not canonical_rule_id:
            self._invalid_action(f"Invalid safety rule: {rule_id}")

        proposal = next(
            (item for item in self.state["proposals"] if item.id == proposal_id), None
        )
        if not proposal:
            self._invalid_action(f"Unknown proposal ID: {proposal_id}")
        if not proposal.is_active:
            self._invalid_action(f"Proposal {proposal_id} is not active in this phase.")
        if proposal.status != "unresolved":
            self._invalid_action(f"Proposal {proposal_id} is already {proposal.status}.")

        relevant_rules = {
            normalize_rule_id(item)
            for item in getattr(proposal, "relevant_rule_ids", [])
            if normalize_rule_id(item)
        }
        if relevant_rules and canonical_rule_id not in relevant_rules:
            self._invalid_action(
                f"Rule {rule_id} is not relevant for proposal {proposal_id}."
            )

        proposal.status = "blocked"
        proposal.rule_id = short_rule_id(canonical_rule_id) or canonical_rule_id

        proposal_has_violation = bool(
            getattr(proposal, "latent_violation_id", None)
            or getattr(proposal, "true_violation_id", None)
        )
        if proposal_has_violation:
            self.metrics["true_blocks"] += 1
        else:
            self.metrics["false_blocks"] += 1
            self.reward += REWARDS["false_block"]

        self._check_terminal()
        output = {"action": "block", "proposal": proposal_id, "terminal": self.done}
        if self.done:
            output["episode_end_reason"] = self.state["terminal_reason"]

        return json.dumps(output)

    def hold_release(
        self,
        reason_code: Literal[
            "awaiting_fix", "risk_too_high", "policy_unclear", "deploy_checks_pending"
        ],
    ) -> str:
        self._assert_not_done()
        if reason_code not in VALID_HOLD_REASONS:
            self._invalid_action(f"Invalid hold reason: {reason_code}")

        self._advance_phase(f"hold_release:{reason_code}")
        output = {
            "action": "hold_release",
            "terminal": False,
            "updated_observation": json.loads(self.render_observation()),
        }
        if self.done:
            output = {
                "action": "hold_release",
                "terminal": True,
                "reason": self.state["terminal_reason"],
            }
        return json.dumps(output, indent=2)
