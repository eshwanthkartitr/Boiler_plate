import json
from typing import Optional, Tuple, Dict, Any, List
from releaseops_arena.tool_env import ReleaseOpsToolEnv

def naive_baseline(env: ReleaseOpsToolEnv) -> float:
    # Always approves active proposals
    while not env.done:
        obs = json.loads(env._render_observation())
        active_props = [p for p in obs["proposals"] if p["status"] == "unresolved"]
        if not active_props:
            # Nothing to do, ship or hold
            env.hold_release("awaiting_fix")
        else:
            for p in active_props:
                try:
                    env.approve_proposal(p["id"])
                except Exception:
                    pass
                if env.done:
                    break
    return env.reward

def rule_baseline(env: ReleaseOpsToolEnv) -> float:
    # Takes evidence if possible, blocks based on explicit danger, else approves
    while not env.done:
        obs = json.loads(env._render_observation())
        active_props = [p for p in obs["proposals"] if p["status"] == "unresolved"]
        if not active_props:
            env.hold_release("awaiting_fix")
        else:
            for p in active_props:
                # If refs available, inspect first
                if p["refs"] and env.state["evidence_actions_remaining"] > 0:
                    for ref in p["refs"]:
                        if ref.startswith("pr_"):
                            try: env.inspect_pr_diff(ref)
                            except Exception: pass
                            break
                        elif ref.startswith("ci_"):
                            try: env.inspect_ci_run(ref)
                            except Exception: pass
                            break
                # Rule logic: approve everything unless it has 'payments' risk
                if "payments" in p["risk"]:
                    try: env.block_proposal(p["id"], "S2")
                    except Exception: pass
                else:
                    try: env.approve_proposal(p["id"])
                    except Exception: pass
                    
                if env.done:
                    break
    return env.reward


def _pick_best_ref(refs: List[str], prefixes: Tuple[str, ...]) -> Optional[str]:
    for prefix in prefixes:
        for ref in refs:
            if ref.startswith(prefix):
                return ref
    return None


def _pick_target_proposal(active_props: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    ships = [p for p in active_props if p.get("request") == "ship_release_candidate"]
    if not ships:
        return None

    def score(p: Dict[str, Any]) -> int:
        risk = set(p.get("risk") or [])
        # Lower score = higher priority
        if "payments" in risk:
            return 0
        if "p1" in risk:
            return 1
        if "deploy" in risk:
            return 2
        return 3

    return sorted(ships, key=score)[0]


def phase_aware_rule_baseline(env: ReleaseOpsToolEnv) -> float:
    """
    A stronger scripted baseline aligned with ref.md:
    - Spend evidence actions only on high-risk ship proposals (payments/p1/deploy).
    - Then resolve proposals efficiently (block when rule-relevant, otherwise approve).
    - Use hold_release when high-risk uncertainty remains but evidence is exhausted.
    """
    while not env.done:
        obs = json.loads(env._render_observation())
        active_props = [p for p in obs["proposals"] if p["status"] == "unresolved"]
        if not active_props:
            env.hold_release("awaiting_fix")
            continue

        target = _pick_target_proposal(active_props)
        if target and env.state["evidence_actions_remaining"] > 0 and target.get("refs"):
            risk = set(target.get("risk") or [])
            if "payments" in risk:
                ref = _pick_best_ref(target["refs"], ("pr_", "ci_run_", "ticket_"))
            elif "p1" in risk:
                ref = _pick_best_ref(target["refs"], ("ticket_", "pr_", "ci_run_"))
            elif "deploy" in risk:
                ref = _pick_best_ref(target["refs"], ("ci_run_", "ticket_", "pr_"))
            else:
                ref = None

            try:
                if ref:
                    if ref.startswith("pr_"):
                        env.inspect_pr_diff(ref)
                    elif ref.startswith("ci_run_"):
                        env.inspect_ci_run(ref)
                    elif ref.startswith("ticket_"):
                        env.inspect_ticket(ref)
            except Exception:
                pass

        # If evidence is exhausted and a high-risk ship proposal remains, prefer holding the phase.
        # This matches the intended “hold vs ship” tradeoff without inspecting everything.
        if env.state["evidence_actions_remaining"] <= 0:
            active_props = [p for p in json.loads(env._render_observation())["proposals"] if p["status"] == "unresolved"]
            high_risk_ship = None
            for p in active_props:
                if p.get("request") == "ship_release_candidate":
                    risk = set(p.get("risk") or [])
                    if {"payments", "p1", "deploy"} & risk:
                        high_risk_ship = p
                        break
            if high_risk_ship:
                env.hold_release("awaiting_fix")
                continue

        # Resolution step: block rule-relevant proposals; otherwise approve.
        for p in active_props:
            possible = p.get("possible_rule_violations") or []
            if possible:
                # Prefer the first relevant rule id, allow env aliases (S1/S2/S3).
                try:
                    env.block_proposal(p["id"], possible[0])
                except Exception:
                    pass
            else:
                try:
                    env.approve_proposal(p["id"])
                except Exception:
                    pass

            if env.done:
                break

    return env.reward
