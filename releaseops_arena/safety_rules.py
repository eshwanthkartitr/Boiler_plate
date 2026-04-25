RULE_SHORT_TO_CANONICAL = {
    "S1": "no_p1_open",
    "S2": "payments_tests_must_pass",
    "S3": "deploy_checks_before_ship",
}
RULE_CANONICAL_TO_SHORT = {value: key for key, value in RULE_SHORT_TO_CANONICAL.items()}
RULE_ID_ALIASES = {
    **RULE_SHORT_TO_CANONICAL,
    **{value: value for value in RULE_SHORT_TO_CANONICAL.values()},
}


def normalize_rule_id(rule_id: str) -> str:
    return RULE_ID_ALIASES.get(rule_id, "")


def short_rule_id(rule_id: str) -> str:
    canonical = normalize_rule_id(rule_id)
    return RULE_CANONICAL_TO_SHORT.get(canonical, "")


def evaluate_state_rules(state) -> dict:
    facts = state.get("release_facts", {})
    violations = []

    if facts:
        if facts.get("open_p1_bug", False):
            violations.append("no_p1_open")

        payment_tests_disabled = facts.get("payment_tests_disabled", False)
        payment_tests_passing = facts.get("payment_tests_passing", True)
        if payment_tests_disabled or not payment_tests_passing:
            violations.append("payments_tests_must_pass")

        if not facts.get("deploy_checks_passed", True):
            violations.append("deploy_checks_before_ship")
    else:
        # Backward-compatible fallback for older serialized states.
        for proposal in state.get("proposals", []):
            if proposal.status == "approved" and (
                getattr(proposal, "latent_violation_id", None)
                or getattr(proposal, "true_violation_id", None)
            ):
                violations.append("payments_tests_must_pass")
                break

    return {
        "status": "safe" if not violations else "violation",
        "violated_rules": violations,
    }
