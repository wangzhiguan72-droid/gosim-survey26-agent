"""An optional, best-effort decision trace log.

Night-to-night and run-wide memory the planner actually leans on (learned sky
scale, per-target factor/misses/attempts, night advice, report cooldowns) lives on
`SurveyState` and `Planner` themselves -- it is genuinely part of what each of them
tracks, not a separate concern. What belongs here is purely diagnostic: an opt-in
JSONL trace of planning events (night advice, weather-event interpretation, the
finish summary), written only if `AGENT_TRACE_PATH` is set. Writing to it never
raises: a read-only sandbox or a bad path must not break the agent. Protocol logs
still only ever go to stderr (see protocol.log); this is a separate artifact.
"""
from __future__ import annotations

import json
import os
import time


class TraceLog:
    def __init__(self, log=lambda text: None):
        self.log = log
        self.path = os.environ.get("AGENT_TRACE_PATH", "").strip()
        self._handle = None
        if self.path:
            try:
                self._handle = open(self.path, "a", encoding="utf-8")
            except OSError as exc:
                self.log(f"memory: could not open AGENT_TRACE_PATH ({exc}); tracing disabled")
                self.path = ""

    def write(self, record: dict) -> None:
        if not self._handle:
            return
        try:
            record = {"ts": time.time(), **record}
            self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            self._handle.flush()
        except OSError:
            pass  # a trace write must never interrupt the decision loop

    def close(self) -> None:
        if self._handle:
            try:
                self._handle.close()
            except OSError:
                pass
