#!/usr/bin/env python3
"""Rule-only entry point: the example Planner with its LLM calls disabled.

Same decision loop as agent.py, but skips require_api_key() and replaces the
LLM client with a stub whose ask_json() always returns None, so every LLM-advised
step (night advice, report confirmation) takes its rule-based fallback path
immediately without spending wall-clock time on HTTP attempts.
"""
from __future__ import annotations

import sys

if sys.version_info < (3, 9):
    sys.stderr.write("agent: Python 3.9 or newer is required\n")
    raise SystemExit(3)

from agent_core.planner import Planner
from agent_core.protocol import log, read_messages, send_response
from agent_core.state import SurveyState
from agent_core.validation import ActionRejected, fallback_action, validate_action


class NullLLM:
    """Stand-in for LLMClient: never calls anything, always falls back to rules."""

    calls_made = 0

    def ask_json(self, system_prompt, user_payload, wallclock_remaining_seconds):
        return None


def main() -> int:
    state = None
    planner = None
    for message in read_messages(sys.stdin):
        kind = message.get("message_type")

        if kind == "initialize":
            try:
                state = SurveyState(message["payload"])
                planner = Planner(state, log=log)
                planner.llm = NullLLM()
            except Exception as exc:  # noqa: BLE001 - never crash on a malformed initialize
                log(f"agent: failed to initialize ({type(exc).__name__}: {exc}); will fall back on every decision")
                state = None
                planner = None

        elif kind == "decision_request":
            sequence = message["decision_sequence"]
            consecutive_reports = planner.consecutive_reports if planner is not None else 0
            try:
                action = planner.decide(message["payload"]) if planner is not None else fallback_action("not initialized")
                action = validate_action(action, state, consecutive_reports)
            except ActionRejected as exc:
                log(f"agent: planner produced an invalid action ({exc}); falling back")
                action = fallback_action("validation-rejected")
            except Exception as exc:  # noqa: BLE001 - a strategy bug must not end the run
                log(f"agent: planner error ({type(exc).__name__}: {exc}); falling back")
                action = fallback_action("planner-exception")
            if planner is not None:
                planner.note_action(action)
            send_response(sequence, action)

        elif kind == "finish":
            if planner is not None:
                try:
                    planner.on_finish(message.get("payload", {}))
                except Exception as exc:  # noqa: BLE001 - finish must not raise after the score is fixed
                    log(f"agent: error during finish logging ({type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
