import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from releaseops_arena.tool_env import ReleaseOpsToolEnv


def first_proposal(env, **criteria):
    for proposal in env.state["proposals"]:
        if all(getattr(proposal, key) == value for key, value in criteria.items()):
            return proposal
    raise AssertionError(f"No proposal matched {criteria}")


def active_unresolved(env):
    return [
        proposal
        for proposal in env.state["proposals"]
        if proposal.is_active and proposal.status == "unresolved"
    ]


def test_blocking_disabled_payment_test_creates_revised_safe_ship():
    env = ReleaseOpsToolEnv()
    env.reset(
        family="green_ci_disabled_payment_test",
        seed=8491,
        difficulty="medium",
        archetype_mix="shortcut_ci__careful_qa",
    )

    bad_ship = first_proposal(env, request="ship_release_candidate")
    pr_id = next(ref for ref in bad_ship.refs if ref.startswith("pr_"))

    env.inspect_pr_diff(pr_id)
    env.block_proposal(bad_ship.id, "S2")
    for proposal in list(active_unresolved(env)):
        env.approve_proposal(proposal.id)

    assert not env.done
    assert env.state["phase"] == "deploy_gate"
    assert env.metrics["true_blocks"] == 1

    revised = active_unresolved(env)
    assert len(revised) == 1
    assert revised[0].request == "ship_release_candidate"
    assert revised[0].true_violation_id is None

    env.approve_proposal(revised[0].id)
    assert env.done
    assert env.state["terminal_reason"] == "safe_ship"


def test_false_block_is_penalized_and_counted():
    env = ReleaseOpsToolEnv()
    env.reset(
        family="careful_qa_safe",
        seed=8528,
        difficulty="low",
        archetype_mix="careful_qa__expert_rm",
    )

    safe_triage = next(
        proposal
        for proposal in active_unresolved(env)
        if proposal.request == "approve_triage"
    )
    starting_reward = env.reward

    env.block_proposal(safe_triage.id, "S2")

    assert env.metrics["false_blocks"] == 1
    assert env.reward < starting_reward


def test_block_proposal_rejects_already_resolved_proposal():
    env = ReleaseOpsToolEnv()
    env.reset(
        family="green_ci_disabled_payment_test",
        seed=8491,
        difficulty="medium",
        archetype_mix="shortcut_ci__careful_qa",
    )

    proposal = next(
        item
        for item in active_unresolved(env)
        if item.request != "ship_release_candidate"
    )
    env.approve_proposal(proposal.id)

    with pytest.raises(ValueError, match="already approved"):
        env.block_proposal(proposal.id, "S2")

    assert env.metrics["invalid_actions"] == 1


def test_observation_hides_inactive_previous_phase_proposals():
    env = ReleaseOpsToolEnv()
    env.reset(
        family="green_ci_disabled_payment_test",
        seed=8491,
        difficulty="medium",
        archetype_mix="shortcut_ci__careful_qa",
    )

    bad_ship = first_proposal(env, request="ship_release_candidate")
    env.block_proposal(bad_ship.id, "S2")
    for proposal in list(active_unresolved(env)):
        env.approve_proposal(proposal.id)

    obs = json.loads(env._render_observation())
    assert obs["phase"] == "deploy_gate"
    assert len(obs["proposals"]) == 1
    assert obs["proposals"][0]["status"] == "unresolved"


def test_block_rejects_non_relevant_rule_for_true_violation():
    env = ReleaseOpsToolEnv()
    env.reset(
        family="green_ci_disabled_payment_test",
        seed=8491,
        difficulty="medium",
        archetype_mix="shortcut_ci__careful_qa",
    )

    bad_ship = first_proposal(env, worker="ci_fixer", request="ship_release_candidate")
    with pytest.raises(ValueError, match="not relevant"):
        env.block_proposal(bad_ship.id, "S1")

    assert env.metrics["invalid_actions"] == 1


def test_approving_open_p1_ship_results_in_unsafe_ship():
    env = ReleaseOpsToolEnv()
    env.reset(
        family="qa_undercalls_p1_checkout_bug",
        seed=9102,
        difficulty="medium",
        archetype_mix="stale_policy_qa__deadline_rm",
    )

    risky_ship = first_proposal(env, request="ship_release_candidate")
    env.approve_proposal(risky_ship.id)

    assert env.done
    assert env.state["terminal_reason"] == "unsafe_ship"
    assert "no_p1_open" in env.state.get("violated_rules", [])
