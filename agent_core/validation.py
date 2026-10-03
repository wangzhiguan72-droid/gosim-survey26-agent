"""Validate a decision_response before it goes to stdout, and provide a
deterministic fallback for when the planner errors out or produces something
invalid. The protocol ends a run as `agent_error` for things like a fibre used
twice, an unknown target, a value out of range, or an unknown field -- this
module's whole job is to make sure none of those ever reach stdout.
"""
from __future__ import annotations

from typing import Optional

PROGRAMS = {"DARK", "BRIGHT", "BACKUP"}


class ActionRejected(ValueError):
    pass


def fallback_action(reason: str = "fallback") -> dict:
    """Always-valid, always-safe action: wait one slot. Works even before the
    agent has seen `initialize`, since it needs no state at all."""
    return {"action": "wait", "duration_seconds": 900, "reason": reason, "decision_source": "deterministic"}


def validate_action(action: dict, state, consecutive_reports: int = 0) -> dict:
    """Return a sanitized copy of `action` containing only protocol-legal fields,
    or raise ActionRejected. `state` may be None (before initialize)."""
    if not isinstance(action, dict):
        raise ActionRejected("action is not a dict")
    kind = action.get("action")
    if kind == "observe":
        return _validate_observe(action, state)
    if kind == "wait":
        return _validate_wait(action, state)
    if kind == "report":
        limit = state.max_consecutive_reports if state is not None else 32
        if consecutive_reports >= limit:
            raise ActionRejected(f"consecutive report limit reached ({limit})")
        return {"action": "report", "reason": action.get("reason"), "decision_source": action.get("decision_source")}
    if kind == "finish":
        return {"action": "finish", "reason": action.get("reason"), "decision_source": action.get("decision_source")}
    raise ActionRejected(f"unknown action kind {kind!r}")


def _validate_wait(action: dict, state) -> dict:
    duration = action.get("duration_seconds")
    until = action.get("until_utc")
    out = {"action": "wait", "reason": action.get("reason"), "decision_source": action.get("decision_source")}
    if until is not None:
        if not isinstance(until, str) or not until.endswith("Z"):
            raise ActionRejected(f"until_utc must be a UTC string ending in Z, got {until!r}")
        out["until_utc"] = until
        return out
    if duration is None:
        raise ActionRejected("wait needs duration_seconds or until_utc")
    duration = int(duration)
    lo = state.min_exposure if state else 60
    hi = state.max_exposure if state else 3600
    if not lo <= duration <= hi:
        raise ActionRejected(f"wait duration_seconds out of range: {duration}")
    out["duration_seconds"] = duration
    return out


def _validate_observe(action: dict, state) -> dict:
    pointing = action.get("pointing") or {}
    alt = pointing.get("alt_deg")
    az = pointing.get("az_deg")
    if not isinstance(alt, (int, float)) or not 0.0 <= alt <= 90.0:
        raise ActionRejected(f"pointing.alt_deg out of range: {alt!r}")
    if not isinstance(az, (int, float)) or not 0.0 <= az < 360.0:
        raise ActionRejected(f"pointing.az_deg out of range: {az!r}")

    assignments = action.get("assignments") or {}
    if not assignments:
        raise ActionRejected("observe needs at least one fibre assignment")
    seen_fibers: set[int] = set()
    seen_targets: set[str] = set()
    clean_assignments: dict[str, str] = {}
    for fiber_key, target_id in assignments.items():
        fiber_id = int(fiber_key)
        if not 0 <= fiber_id < (state.fiber_grid.n if state else 16):
            raise ActionRejected(f"fibre id out of range: {fiber_key!r}")
        if fiber_id in seen_fibers:
            raise ActionRejected(f"fibre {fiber_id} used twice")
        if target_id in seen_targets:
            raise ActionRejected(f"target {target_id} assigned twice")
        if state is not None and target_id not in state.index_of:
            raise ActionRejected(f"unknown target id {target_id!r}")
        seen_fibers.add(fiber_id)
        seen_targets.add(target_id)
        clean_assignments[str(fiber_id)] = target_id

    duration = action.get("duration_seconds")
    if duration is None:
        raise ActionRejected("observe needs duration_seconds")
    duration = int(duration)
    lo = state.min_exposure if state else 60
    hi = state.max_exposure if state else 3600
    if not lo <= duration <= hi:
        raise ActionRejected(f"duration_seconds out of range: {duration}")

    program = action.get("program", "BACKUP")
    if program not in PROGRAMS:
        raise ActionRejected(f"unknown program {program!r}")

    return {
        "action": "observe",
        "pointing": {"alt_deg": float(alt), "az_deg": float(az)},
        "assignments": clean_assignments,
        "duration_seconds": duration,
        "program": program,
        "reason": action.get("reason"),
        "decision_source": action.get("decision_source"),
    }
