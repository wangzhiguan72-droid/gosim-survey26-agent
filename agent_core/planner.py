"""Decision logic: pick a pointing, fill the 16 fibres, choose exposure length and
program -- or wait / report / finish.

1. Rank visible, not-yet-done targets. Required targets that are not done yet get a
   bonus (missing one costs real points at the end). Targets that set soon, or that
   have few nights left, rank higher.
2. For the best few "anchor" candidates, try each fibre as the pointing centre; fill
   every fibre with the best-value neighbour that lands on its glass; keep the best
   pointing.
3. Pick the exposure length with the best expected score per second, and the program
   (DARK / BRIGHT / BACKUP) most assigned targets will match.

Everything here uses only the public catalogue, the public scoring config, and the
agent's own past hits (SurveyState) -- never hidden weather truth. Once per night, two
LLM calls run and their answers are merged (see `_night_advice` below): one reads the
forecast/bulletin notices for tonight, the other reads tonight's live bulletin text and
the agent's own hit rate so far. A call that keeps failing falls back to its rule-based
answer for that night; the next night's calls still run normally.

This mirrors the anchor-search algorithm of this project's companion TypeScript
example target-for-target, so both examples solve the problem the same way.
"""
from __future__ import annotations

from datetime import timedelta

from .geometry import (
    Moon,
    SIDEREAL_DEG_PER_SECOND,
    altaz_to_radec,
    format_utc,
    local_sidereal_deg,
    lunar_factor,
    max_hour_angle_deg,
    parse_utc,
    radec_to_altaz,
    shift_altaz,
    tangent_offsets,
    wrap180,
)
from .llm_client import LLMClient
from .memory import TraceLog
from .state import PendingPrediction

import math
import os
from collections import deque

REQUIRED_BONUS = float(os.environ.get("SAC_REQ_BONUS", "60"))
REQUEST_BONUS_CAP = float(os.environ.get("SAC_REQ_CAP", "6"))
REQUEST_URGENCY_HOURS = float(os.environ.get("SAC_REQ_URGENCY", "14"))
LLM_DUR_LO = float(os.environ.get("SAC_LLM_DUR_LO", "0.9"))
LLM_DUR_HI = float(os.environ.get("SAC_LLM_DUR_HI", "1.1"))
# Weather advice may only ever nudge: at most this many directions, and the avoid
# factor is a mild value discount (not the 0.35 hard dodging the bulletins get) so
# one bad model answer cannot fence off half the sky.
LLM_AVOID = os.environ.get("SAC_LLM_AVOID", "1") != "0"
LLM_AVOID_MAX_DIRECTIONS = int(os.environ.get("SAC_LLM_AVOID_MAX", "2"))
LLM_AVOID_FACTOR = float(os.environ.get("SAC_LLM_AVOID_FACTOR", "0.75"))
LLM_VETO = os.environ.get("SAC_LLM_VETO", "0") != "0"
RESCUE_DAMP = os.environ.get("SAC_RESCUE_DAMP", "0") != "0"
RESCUE_DUR = os.environ.get("SAC_RESCUE_DUR", "1") != "0"
DONE_FACTOR = 0.95
PLAN_FACTOR_SAFETY = float(os.environ.get("SAC_SAFETY", "0.9"))
EDGE_MARGIN_DEG = 0.08
DURATIONS = tuple(int(x) for x in os.environ.get(
    "SAC_DURS", "300,450,600,900,1200,1500,1800,2400,3000,3600").split(","))
MIN_VISIBLE_SECONDS = 600
NEIGHBOUR_RADIUS_DEG = 2.1
ANCHORS = int(os.environ.get("SAC_ANCHORS", "6"))
ANCHOR_POOL = int(os.environ.get("SAC_ANCHOR_POOL", "300"))
CLOSED_KINDS = {"rain", "storm"}
BLOCKING_KINDS = {"terrain_obstruction", "rocket_launch"}
WEATHER_EXPLAINS = {"rain", "storm", "overcast", "haze", "cold_snap"}
DIRECTION_AZ = {"N": 0.0, "NE": 45.0, "E": 90.0, "SE": 135.0, "S": 180.0,
                "SW": 225.0, "W": 270.0, "NW": 315.0}

REPORT_DROP_FIRST = float(os.environ.get("SAC_REPORT_DROP", "0.78"))
REPORT_DROP_LATER = float(os.environ.get("SAC_REPORT_DROP2", "0.72"))
REPORT_CONFIRMATIONS = 2
REPORT_SPACING_HOURS = 2.5
FALSE_SUPPRESS_HOURS = 20.0
QUAKE_GUARD_HOURS = float(os.environ.get("SAC_QUAKE_GUARD", "30"))
MAX_REPORTS = int(os.environ.get("SAC_MAX_REPORTS", "6"))

# Dedicated completion mode (one-exposure threshold crossings)
DEDICATED_SAFETY = float(os.environ.get("SAC_DED_SAFETY", "0.85"))
DEDICATED_MAX_ANCHORS = int(os.environ.get("SAC_DED_ANCHORS", "4"))
# Under decision starvation (pace level 2) the dedicated pass used to shut off
# entirely -- exactly when faint required targets and request windows needed it.
# It now survives in a one-anchor, near-miss/request-only form (SAC_L2_DEDICATED=0
# restores the old shutoff).
L2_DEDICATED = os.environ.get("SAC_L2_DEDICATED", "1") != "0"
RESCUE_RETRY_HOURS = float(os.environ.get("SAC_RESCUE_RETRY", "12"))
RESCUE_MAX_ATTEMPTS = int(os.environ.get("SAC_RESCUE_MAX_ATTEMPTS", "3"))
BIG_SPECIAL = 1.0e5
DEDICATED_PER_NIGHT = int(os.environ.get("SAC_DED_PER_NIGHT", "2"))
RESCUE_MAX_T_NEED = float(os.environ.get("SAC_RESCUE_MAX_T", "1800"))
# Near-miss retries: a required target already at factor 0.3-0.5 only needs a
# slightly longer exposure to cross; dropping it after two attempts wasted
# ~20 platform-card misses worth of penalty in exactly this band.
NEAR_MISS_LO = 0.30
NEAR_MISS_MAX_ATTEMPTS = int(os.environ.get("SAC_NEAR_MAX", "5"))
NEAR_MISS_RETRY_HOURS = float(os.environ.get("SAC_NEAR_RETRY", "10"))
NEAR_MISS_ENDGAME_NIGHTS = int(os.environ.get("SAC_NEAR_ENDGAME", "8"))
# Mid-survey, ANY perturbation of the dedicated pass gets amplified into
# thousands of points of chaos (formal B lost 5k to the unmasked v8 retries).
# The retry machinery therefore only unlocks in the survey's own last
# SURVEY_TAIL_NIGHTS nights, where displaced fields have no future value.
SURVEY_TAIL_NIGHTS = int(os.environ.get("SAC_SURVEY_TAIL", "12"))
REQ_ALT_MARGIN = float(os.environ.get("SAC_REQ_ALT_MARGIN", "1.5"))
NEAR_MISS_SAFETY = float(os.environ.get("SAC_NEAR_SAFETY", "0.58"))
NEAR_MISS_AIM = float(os.environ.get("SAC_NEAR_AIM", "0.56"))
REQUIRED_AIM_MULT = float(os.environ.get("SAC_REQ_AIM", "1.0"))
# Fault reporting: after the free false allowance is burnt, only a deep drop
# (near-certain fault) is worth the -150 risk; and a recent ALL-sky weather
# notice explains quality drops for a day after it clears.
REPORT_DROP_BURNED = float(os.environ.get("SAC_REPORT_DROP3", "0.45"))
WEATHER_LOOKBACK_HOURS = float(os.environ.get("SAC_WX_LOOKBACK", "0"))


def _az_distance(a: float, b: float) -> float:
    return abs(wrap180(a - b))


def _bulletin_text(notices: list) -> str:
    """A human-readable rendering of a bulletin's notices, for the LLM call that reads
    "live bulletin text" rather than structured JSON."""
    if not notices:
        return "clear (no active notices)"
    return "; ".join(f"{n.get('event_kind')} {n.get('direction')}" for n in notices)


def _notice_json(notice) -> str:
    import json as _json
    try:
        return _json.dumps(notice, sort_keys=True)
    except (TypeError, ValueError):
        return str(notice)


class Planner:
    def __init__(self, state, log=lambda text: None):
        self.state = state
        self.log = log
        self.grid = state.fiber_grid
        self.llm = LLMClient(log=log)
        self.trace = TraceLog(log=log)

        self.observe_count = 0
        self.reports = 0
        self.correct_reports = 0
        self.false_since_correct = 0
        self.last_false_hours = float("-inf")
        self.last_all_weather_hours = float("-inf")
        self.last_report_hours = float("-inf")
        self.suspicion_hours: list[float] = []
        self.suspicion_nights: list[int] = []
        self._decide_t0 = 0.0
        self._decide_seconds = 0.0
        self._decide_count = 0
        self._t_report = self._t_dedicated = self._t_plan = self._t_onresult = 0.0
        self._recent_decide_durs: list[float] = []
        self._last_remaining: float | None = None
        self._last_sim = None
        self._cycle_costs: deque = deque(maxlen=30)   # wall seconds per full decision cycle
        self._sim_advances: deque = deque(maxlen=30)  # sim seconds a cycle advances
        self._value_cache: dict[int, float] = {}
        self._dirty_values: set[int] = set()
        self._request_sig: tuple | None = None
        self._seen_resync = -1
        self.night_index_seen: int | None = None
        self.consecutive_reports = 0
        self._last_forecast_notices: list = []
        self._notice_events: list = []     # (hours, sorted notices, hours since last quake or None)
        self._notice_sig: tuple | None = None
        self._advice_sig: tuple | None = None
        self._report_log: list = []        # (hours, correct|None) -- None = emitted, outcome pending
        self.total_assigned = 0
        self.total_hit = 0
        self.request_bonus: dict[int, float] = {}
        self.request_bonus_full: dict[int, float] = {}
        self.request_threshold: dict[int, float] = {}
        self.active_reqs: list[dict] = []
        self.rescue_last_try: dict[int, float] = {}
        self._dedicated_tonight: dict[int, int] = {}
        # A required target is rescuable only if a max-length exposure in near-ideal
        # conditions can plausibly cross the required threshold; fainter ones must
        # stay buried by the attempts damp or they eat the schedule for nothing.
        f0t0 = state.scoring.f0t0
        self._req_rescuable = [state.flux[i] * state.max_exposure * 1.2 >= state.scoring.required_threshold * f0t0
                               for i in range(len(state.ids))]

        log(f"planner: {len(state.ids)} targets ({sum(state.required)} required), "
            f"{len(state.nights)} nights, llm model={self.llm.model} base_url={self.llm.base_url}")

    # -- top-level decision ----------------------------------------------------

    def decide(self, payload: dict) -> dict:
        import time as _time
        self._decide_t0 = _time.monotonic()
        try:
            return self._decide(payload)
        finally:
            dur = _time.monotonic() - self._decide_t0
            self._decide_seconds += dur
            self._decide_count += 1
            self._recent_decide_durs.append(dur)
            del self._recent_decide_durs[:-24]

    def _decide(self, payload: dict) -> dict:
        import time as _t
        state = self.state
        now = parse_utc(payload["now_utc"])
        hours = (now - state.survey_start).total_seconds() / 3600.0

        for message in payload.get("new_messages", []):
            if message.get("record_type") == "forecast":
                self._last_forecast_notices = message.get("notices", [])
        state.on_messages(payload.get("new_messages", []), payload.get("latest_bulletin"))
        if any(key.partition("|")[0] in WEATHER_EXPLAINS and key.partition("|")[2] == "ALL"
               for key in state.notices):
            self.last_all_weather_hours = hours
        quake_hours = (None if state.last_quake_at is None
                       else (now - state.last_quake_at).total_seconds() / 3600.0)
        sig = (tuple(sorted(state.notices)), quake_hours is not None and quake_hours < 1.0)
        if sig != self._notice_sig:
            self._notice_events.append((round(hours, 3), sorted(state.notices),
                                        round(quake_hours, 2) if quake_hours is not None else None))
            self._notice_sig = sig
        t0 = _t.monotonic()
        # Targets about to be updated by on_result: their value() inputs move.
        touched = {self.state.index_of.get(tid) for tid in self.state.pending}
        touched.discard(None)
        state.on_result(payload.get("last_result"), hours)
        self._dirty_values |= touched
        if state.resync_generation != self._seen_resync:
            self._seen_resync = state.resync_generation
            self._dirty_values.clear()
            self._value_cache.clear()
        self._t_onresult += _t.monotonic() - t0
        last_result = payload.get("last_result")
        if last_result and last_result.get("action") == "observe":
            self.total_assigned += int(last_result.get("assigned_count", 0))
            self.total_hit += int(last_result.get("hit_count", 0))
        elif last_result and last_result.get("action") == "report":
            self._report_log.append((round(hours, 3), bool(last_result.get("correct"))))
            if last_result.get("correct"):
                self.correct_reports += 1
                self.false_since_correct = 0
                # A correct report repairs the instrument: the pre-repair quality
                # baseline is poisoned now, so start the learning history over.
                state.forget_quality_history()
                self.log(f"planner: report CORRECT (+{last_result.get('score_delta')}) at {payload.get('now_utc')}")
            else:
                # A false report changes nothing about the sky: keep the history
                # (wiping it would blind the detector for two nights).
                self.false_since_correct += 1
                self.last_false_hours = hours
                self.log(f"planner: report false at {payload.get('now_utc')} "
                         f"(false_since_correct={self.false_since_correct})")
        self._pace(payload, now)
        self._update_requests(payload.get("active_requests") or [], now)

        night = state.current_night(now)
        if night is None:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "no observing night left"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "daytime: sleep until the next night"}
        night_index, night_start, night_end = night

        if self.night_index_seen != night_index:
            self.night_index_seen = night_index
            self._night_advice(night_start, payload)

        if (night_end - now).total_seconds() < state.min_exposure:
            nxt = state.next_night_start(now)
            if nxt is None:
                return {"action": "finish", "reason": "survey over"}
            return {"action": "wait", "until_utc": format_utc(nxt), "reason": "night ending"}

        if state.site_closed():
            return {"action": "wait", "duration_seconds": self._idle_wait(now, night_start, night_end),
                    "reason": "bulletin: rain/storm over the whole sky"}

        t0 = _t.monotonic()
        report = self._maybe_report(hours, payload)
        self._t_report += _t.monotonic() - t0
        if report is not None:
            return report

        t0 = _t.monotonic()
        dedicated = self._dedicated_plan(now, night_end, night_index, hours)
        self._t_dedicated += _t.monotonic() - t0
        if dedicated is not None:
            self.observe_count += 1
            dedicated["reason"] = f"dedicated: {dedicated.get('goal', '?')} ({len(dedicated['assignments'])} fibres, program {dedicated['program']})"
            return dedicated

        t0 = _t.monotonic()
        action = self.plan(now, night_end, night_index, hours)
        self._t_plan += _t.monotonic() - t0
        if action is None:
            return {"action": "wait", "duration_seconds": self._idle_wait(now, night_start, night_end),
                    "reason": "nothing useful is up"}
        self.observe_count += 1
        action["reason"] = f"{len(action['assignments'])} fibres, program {action['program']}"
        return action

    def on_finish(self, payload: dict) -> None:
        self.trace.write({"event": "finish", **payload})
        self.trace.close()
        dump_path = os.environ.get("SAC_DUMP_QUALITY")
        if dump_path:
            import json as _json
            state = self.state
            by_night: dict[int, list[float]] = {}
            for _h, night, ratio, _clean, _az in state.quality_log:
                by_night.setdefault(night, []).append(ratio)
            nightly = {str(k): sorted(v)[len(v) // 2] for k, v in sorted(by_night.items())}
            with open(dump_path, "w", encoding="utf-8") as fh:
                _json.dump({
                    "samples": [(round(h, 3), n, round(r, 4), c, az)
                                for h, n, r, c, az in state.quality_log],
                    "nightly_medians": nightly,
                    "notice_events": self._notice_events,
                    "report_log": self._report_log,
                    "band_checks": list(state._band_checks),
                    "scale_final": round(state.scale, 4),
                }, fh)
            self.log(f"planner: quality dump written to {dump_path}")
        self.log(f"planner: finished termination_reason={payload.get('termination_reason')} "
                 f"observes={self.observe_count} reports={self.reports} llm_calls={self.llm.calls_made}")
        if os.environ.get("SAC_TIMING"):
            self.log(f"planner: timing seconds decide={self._decide_seconds:.1f} "
                     f"on_result={self._t_onresult:.1f} report={self._t_report:.1f} "
                     f"dedicated={self._t_dedicated:.1f} plan={self._t_plan:.1f} "
                     f"over {self._decide_count} decisions")

    def note_action(self, action: dict) -> None:
        """Called by agent.py right after an action is validated, so the consecutive-report
        counter (enforced by validation.py) stays correct even when a fallback replaced it."""
        self.consecutive_reports = self.consecutive_reports + 1 if action.get("action") == "report" else 0

    def _to_next_slot(self, now, night_start) -> int:
        slot = self.state.slot_seconds
        into = (now - night_start).total_seconds() % slot
        return int(max(60, min(3600, slot - into if into else slot)))

    def _idle_wait(self, now, night_start, night_end) -> int:
        """Weather-closure / nothing-up waiting.

        Every wait burns one full decision cycle (~0.15 s of wall clock on a
        formal card), and the old one-slot waits repeated up to ~40x per closed
        night. Under pace pressure, land on a slot boundary several slots out
        instead of the very next one: the sim time lost to a slower weather
        re-check is cheap next to the wall clock each extra round trip costs.
        Slot alignment is kept (level 0 waits exactly one slot, as before) --
        request windows and bulletin updates are slot-shaped."""
        slot = self.state.slot_seconds
        into = (now - night_start).total_seconds() % slot
        step = 1 if self.state.fast_level == 0 else 3
        target = slot * step - into if into else slot * step
        cap = (night_end - now).total_seconds()
        return int(max(60, min(3600, target, cap)))

    def _pace(self, payload: dict, now) -> None:
        """Do less work per decision when the wall clock is short for the nights still to come.

        The unit of cost is one full decision CYCLE -- our planning time plus the
        transport plus the engine's own work -- measured as the drop in
        `wallclock.remaining_seconds` between consecutive requests. That is what
        the 900 s budget actually bills, and on formal-scale cards it runs about
        twice the agent's own decide() time (FB: ~0.15 s/cycle vs ~0.07 s of
        planning). Projection: cycles still owed (remaining night seconds over
        the measured sim-seconds a cycle advances) times a conservative cycle
        cost, versus the wall clock left."""
        state = self.state
        remaining = float((payload.get("wallclock") or {}).get("remaining_seconds", 1e9))
        if self._last_remaining is not None and 0.0 < self._last_remaining - remaining < 600.0:
            self._cycle_costs.append(self._last_remaining - remaining)
        if self._last_sim is not None and 0.0 < (now - self._last_sim).total_seconds() < 6 * 3600:
            self._sim_advances.append((now - self._last_sim).total_seconds())
        self._last_remaining = remaining
        self._last_sim = now

        night_seconds = sum(max(0.0, (end - max(start, now)).total_seconds()) for start, end in state.nights if end > now)
        advances = sorted(self._sim_advances)
        sim_per_cycle = advances[len(advances) // 2] if len(advances) >= 8 else 700.0
        decisions_left = max(1.0, night_seconds / max(1.0, sim_per_cycle))
        cycles = sorted(self._cycle_costs)
        if len(cycles) >= 12:
            # 75th percentile: one slow cycle (a state_resync over 50k targets, a
            # slow model call) must not clamp the run into survival mode, but the
            # typical cost alone has been too optimistic on formal cards.
            cycle_cost = cycles[int(0.75 * (len(cycles) - 1))]
            if cycle_cost > 0.85 * remaining / decisions_left:
                level = 2
            elif cycle_cost > 0.55 * remaining / decisions_left:
                level = 1
            else:
                level = 0
        else:
            # No cycle track record yet: the v8 nominal budget (absolute
            # ms-per-decision thresholds). A ratio against `remaining` would
            # self-normalize to survival mode on the very first decisions.
            per_decision = remaining / decisions_left
            level = 0 if per_decision > 0.12 else 1 if per_decision > 0.04 else 2
        if level != state.fast_level:
            self.log(f"planner: pace level {level} ({remaining:.0f}s wall left, "
                     f"{decisions_left:.0f} cycles owed, "
                     f"measured median {cycles[len(cycles) // 2] * 1000 if cycles else 0:.0f} ms)")
            state.fast_level = level

        if os.environ.get("SAC_TIMING") and self._decide_count % 400 == 0:
            # Periodic, because the platform kills timed-out agents without a
            # finish message -- at-finish-only timing logs never survive a FB run.
            self.log(f"planner: timing t={self._decide_seconds:.1f}s over {self._decide_count} decisions "
                     f"(on_result={self._t_onresult:.1f} report={self._t_report:.1f} "
                     f"dedicated={self._t_dedicated:.1f} plan={self._t_plan:.1f}) "
                     f"wall_left={remaining:.0f}s level={level}")

    # -- LLM: weather advice, asked only when the sky's story changes -------------

    def _night_advice(self, night_start, payload: dict) -> None:
        """Event-triggered weather advice.

        The old version paid two model calls every night while its answers were
        clamped to no-ops (avoid off, duration scale pinned at 1). Now: one call,
        only when tonight's forecast notices or the live bulletin differ from the
        skies the last advice already covered -- identical skies reuse the cached
        answer for free. Its effect is deliberately mild and always applied: at
        most LLM_AVOID_MAX_DIRECTIONS extra-avoid directions (a 0.75 value factor,
        not the 0.35 hard dodging that bulletin notices get) and a duration scale
        inside [LLM_DUR_LO, LLM_DUR_HI] = [0.9, 1.1]. A failed or missing call
        just leaves the rule-based defaults (no avoid, scale 1.0)."""
        state = self.state
        night_date = (night_start - timedelta(hours=12)).date().isoformat()

        forecast_tonight = [n for n in self._last_forecast_notices if night_date in (n.get("nights") or [])]
        bulletin_notices = (payload.get("latest_bulletin") or {}).get("notices", [])
        # No night_date on purpose: a stable multi-night weather stretch should
        # cost ONE call, not one per night. Any forecast or bulletin change
        # re-triggers immediately.
        sig = (tuple(sorted(_notice_json(n) for n in forecast_tonight)),
               tuple(sorted(f"{n.get('event_kind')}|{n.get('direction')}" for n in bulletin_notices)))
        if sig == self._advice_sig:
            return  # same night, same forecast, same bulletin: cached advice stands

        left = float((payload.get("wallclock") or {}).get("remaining_seconds", 0))
        hit_rate = (self.total_hit / self.total_assigned) if self.total_assigned > 0 else 1.0
        answer = self.llm.ask_json(
            "You help schedule a telescope survey. Reply with one JSON object only: "
            '{"avoid_directions": [compass codes among N,NE,E,SE,S,SW,W,NW], "duration_scale": '
            f"number {LLM_DUR_LO}-{LLM_DUR_HI}}}. Avoid directions with bad weather tonight (at most "
            f"{LLM_AVOID_MAX_DIRECTIONS} codes, worst first), going by the forecast, the current bulletin and the "
            "agent's own hit rate; use a larger duration_scale when the sky looks poor, a smaller one when it "
            "looks pristine. Be conservative: empty avoid_directions and duration_scale 1.0 are fine answers.",
            {"night": night_date, "forecast_notices_for_tonight": forecast_tonight,
             "current_bulletin_text": _bulletin_text(bulletin_notices),
             "hit_rate_so_far": round(hit_rate, 3)},
            left,
        )
        avoid: set[str] = set()
        scale = 1.0
        if answer:
            if LLM_AVOID:
                ordered: list[str] = []
                for d in (answer.get("avoid_directions") or []):
                    d = str(d).upper()
                    if d in DIRECTION_AZ and d not in ordered:
                        ordered.append(d)
                avoid = set(ordered[:LLM_AVOID_MAX_DIRECTIONS])  # model order = worst first
            try:
                scale = min(LLM_DUR_HI, max(LLM_DUR_LO, float(answer.get("duration_scale", 1.0))))
            except (TypeError, ValueError):
                pass
        state.extra_avoid = avoid
        state.duration_scale = scale
        self._advice_sig = sig
        self.log(f"planner: night {night_date} llm advice ({'ok' if answer else 'fell back'}) "
                 f"avoid={sorted(avoid)} duration x{scale:.2f}")
        self.trace.write({"event": "night_advice", "night_date": night_date, "avoid": sorted(avoid),
                          "scale": scale, "call_ok": bool(answer)})

    # -- instrument fault reporting (deterministic rules + LLM confirmation) -----

    def _maybe_report(self, hours: float, payload: dict):
        """Instrument-fault reporting.

        Report aggressively once the learned quality level drops well below its own
        baseline: the first `false_report_free_allowance` false reports after each
        correct one are free, so probing costs nothing until the budget runs out.
        Two guards keep the budget from being wasted on lookalikes:
          * earthquakes are announced in bulletins and their damage decays nightly
            -- recovering nightly medians mean quake, not fault;
          * a real fault never changes program-band matching (the band formula
            excludes efficiency), while bad weather does -- a collapsing DARK
            match rate means weather.
        """
        state = self.state
        state.force_program = None
        if self.reports >= MAX_REPORTS or hours - self.last_report_hours < 12.0:
            return None
        allowance = max(1, state.false_report_free_allowance)
        deep_only = self.false_since_correct >= allowance
        if hours - self.last_false_hours < FALSE_SUPPRESS_HOURS:
            return None
        evidence = state.fault_evidence()
        dbg = os.environ.get("SAC_DEBUG_REPORT")
        if evidence is not None and dbg:
            self.log(f"planner: fault evidence {evidence}")
        threshold = REPORT_DROP_FIRST if self.correct_reports == 0 else REPORT_DROP_LATER
        if deep_only:
            # Allowance burnt (e.g. by a seasonal weather stretch): one more try
            # only in certain-fault territory -- flat, no recovery trend, and far
            # deeper than any weather-only dip observed so far.
            threshold = min(threshold, REPORT_DROP_BURNED)
        if evidence is None or evidence.drop >= threshold:
            self.suspicion_hours = []
            return None
        if state.last_quake_at is not None:
            quake_hours = (parse_utc(payload["now_utc"]) - state.last_quake_at).total_seconds() / 3600.0
            if quake_hours < QUAKE_GUARD_HOURS and state.quality_recovering():
                if dbg:
                    self.log(f"planner: report veto quake-guard (quake {quake_hours:.0f}h ago, recovering)")
                return None
        # An active ALL-sky weather bulletin explains a global quality drop;
        # instrument faults are never announced. The same holds for the day
        # after one clears: the 2-night evidence window still holds its dip.
        if hours - self.last_all_weather_hours < WEATHER_LOOKBACK_HOURS:
            if dbg:
                self.log(f"planner: report veto weather-lookback ({hours - self.last_all_weather_hours:.0f}h since ALL notice)")
            self.suspicion_hours = []
            return None
        if any(key.partition("|")[0] in WEATHER_EXPLAINS and key.partition("|")[2] == "ALL"
               for key in state.notices):
            self.suspicion_hours = []
            return None
        # A sustained nightly climb in the quality ratio is quake damage
        # decaying or weather clearing -- a stuck fault never climbs. Vetoing
        # here is what keeps the free-false allowance for the real thing.
        if state.night_median_trend() == "rising":
            if dbg:
                self.log("planner: report veto rising-trend")
            return None
        if self.suspicion_hours and hours - self.suspicion_hours[-1] < REPORT_SPACING_HOURS:
            return None
        night = state.current_night(parse_utc(payload["now_utc"]))
        self.suspicion_hours.append(hours)
        self.suspicion_nights.append(night[0] if night else -1)
        if len(self.suspicion_hours) > 6:  # a stale same-night chain proves nothing
            self.suspicion_hours = self.suspicion_hours[-6:]
            self.suspicion_nights = self.suspicion_nights[-6:]
        # Confirmations must land on two different nights: a same-night pair
        # fires on weather dips that look deep for a few hours.
        if not (len(self.suspicion_hours) >= REPORT_CONFIRMATIONS
                and self.suspicion_nights[-1] != self.suspicion_nights[-2]):
            if dbg:
                self.log(f"planner: suspicion holds (nights {self.suspicion_nights[-3:]})")
            return None
        self.suspicion_hours = []
        self.suspicion_nights = []
        verdict_answer = self.llm.ask_json(
            "You check telescope data quality. A false instrument-fault report costs points, "
            'a correct one earns points. Reply with one JSON object only: {"report": true|false}.',
            evidence._asdict(), float((payload.get("wallclock") or {}).get("remaining_seconds", 0)),
        )
        verdict = verdict_answer.get("report") if isinstance(verdict_answer, dict) and \
            isinstance(verdict_answer.get("report"), bool) else None
        if verdict is False:
            # The model's veto is advisory-only by default (SAC_LLM_VETO=1 restores the
            # veto): on the practice cards a veto cost a real +100 repair reward.
            self.log(f"planner: model would veto the report at {payload.get('now_utc')} ({evidence}); "
                     f"{'veto applied' if LLM_VETO else 'advisory only, reporting anyway'}")
            if LLM_VETO:
                self.last_report_hours = hours
                return None
        self.reports += 1
        self.last_report_hours = hours
        self._report_log.append((round(hours, 3), "emit"))
        self.log(f"planner: reporting instrument fault at {payload.get('now_utc')} evidence={evidence}")
        return {"action": "report", "reason": f"quality dropped to {evidence.drop:.0%} of the earlier level",
                "decision_source": "llm-confirmed" if verdict else "rule"}

    # -- observation requests ---------------------------------------------------

    def _update_requests(self, active: list, now) -> None:
        """Turn active_requests into a per-target planning bonus, capped so a request
        can win ties and attract pointings without hijacking the whole schedule the
        way an uncapped reward/remaining split did (it emptied fibre fills and cost
        far more science than the +100 reward was worth)."""
        state = self.state
        if active:
            self.log(f"planner: {len(active)} active request(s) at {now}: "
                     + "; ".join(f"{r.get('request_id')} need {r.get('remaining_count', '?')} reward {r.get('completion_reward')}" for r in active))
        bonus: dict[int, float] = {}
        bonus_full: dict[int, float] = {}
        thresholds: dict[int, float] = {}
        self.active_reqs = []
        sig_parts: list = []
        for req in active:
            remaining = req.get("remaining_count")
            if remaining is None:
                remaining = int(req.get("minimum_completed", 1)) - int(req.get("completed_count", 0))
            remaining = int(remaining)
            reward = float(req.get("completion_reward", 0.0))
            completed = {str(t) for t in (req.get("completed_target_ids") or [])}
            targets_left = [state.index_of[str(t)] for t in (req.get("target_ids") or [])
                            if str(t) not in completed and str(t) in state.index_of]
            deadline = parse_utc(req["deadline_utc"]) if req.get("deadline_utc") else None
            urgency = 1.0
            if deadline is not None:
                hours_left = (deadline - now).total_seconds() / 3600.0
                urgency = 2.0 if hours_left < REQUEST_URGENCY_HOURS else 1.0
            sig_parts.append((req.get("request_id"), remaining, req.get("deadline_utc"), urgency))
            if remaining > 0 and targets_left:
                self.active_reqs.append({
                    "id": req.get("request_id"), "targets_left": targets_left,
                    "remaining": remaining, "reward": reward, "deadline": deadline,
                    "threshold": float(req.get("completion_factor_threshold", 0.5)),
                })
            if remaining <= 0:
                continue
            full = reward / max(1, remaining) * urgency
            per = min(full, REQUEST_BONUS_CAP)
            for tid in req.get("target_ids") or []:
                tid = str(tid)
                if tid in completed:
                    continue
                i = state.index_of.get(tid)
                if i is not None:
                    bonus[i] = bonus.get(i, 0.0) + per
                    bonus_full[i] = bonus_full.get(i, 0.0) + full
                    thresholds[i] = max(thresholds.get(i, 0.0),
                                        float(req.get("completion_factor_threshold", 0.5)))
        # Only a change in the request set (or in how much of it is left) moves
        # per-target bonuses; until then the cached planning values stay valid.
        sig = tuple(sig_parts)
        if sig != self._request_sig:
            self._dirty_values |= bonus.keys() | self.request_bonus.keys()
            self._request_sig = sig
        self.request_bonus = bonus
        self.request_bonus_full = bonus_full
        self.request_threshold = thresholds

    # -- dedicated completion mode ----------------------------------------------
    # Both observation requests and the required threshold pay a lump sum the
    # moment ONE exposure crosses g=0.5. The normal scheduler optimizes gain per
    # second, so a faint target that needs a 1500 s+ exposure to cross never wins
    # a pointing -- that is exactly why requests went 0/2 and faint required
    # targets kept missing by a hair. This pass picks the pointing and the
    # exposure length for THOSE targets, fills the rest of the field normally,
    # and hands back an ordinary observe action.

    def _fill_value(self, j: int) -> float:
        """Lightweight fill value for non-special neighbours in a dedicated field."""
        state = self.state
        if state.factor[j] >= DONE_FACTOR:
            return self.request_bonus.get(j, 0.0)
        damp = (0.6 ** state.misses[j]) * (0.7 ** state.attempts[j])
        f = state.factor[j]
        base = state.weight[j] * (1.0 - f * f)
        if state.required[j] and f < state.scoring.required_threshold:
            base += REQUIRED_BONUS
        return base * damp + self.request_bonus.get(j, 0.0)

    def _dedicated_plan(self, now, night_end, night_index: int, hours: float):
        state = self.state
        level = state.fast_level
        if level >= 2 and not L2_DEDICATED:
            return None
        horizon = min(night_end, state.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < state.min_exposure + 60:
            return None
        lst = local_sidereal_deg(now, state.lon)
        moon = Moon(now + timedelta(seconds=450), lst, state.lat)
        scoring = state.scoring
        f0t0 = scoring.f0t0

        special: dict[int, dict] = {}

        def consider(i: int, threshold: float, kind: str, prio: float, deadline=None, max_t: float | None = None, safety: float | None = None, t_override: float | None = None) -> None:
            # Hour-angle gate first: pure arithmetic, no trig. Roughly half the
            # required targets fail it on any given decision, and the altitude
            # check below correlates almost perfectly with it (hmax IS the hour
            # angle at which the target sits at the altitude floor).
            ha = wrap180(lst - state.ra[i])
            h = state.hmax[i]
            if h < 180 and not (-h <= ha <= h):
                return
            alt, az = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
            # Request windows are short: a smaller altitude cushion for their
            # anchors than the survey-wide 1.5 deg (SAC_REQ_ALT_MARGIN).
            if alt < state.min_alt + (REQ_ALT_MARGIN if kind == "request" else 1.5):
                return
            up = (h - ha) / SIDEREAL_DEG_PER_SECOND if h < 180 else 1e9
            if up < state.min_exposure:
                return
            lunar = lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            if model <= 0.0:
                return
            if t_override is not None:
                t_need = max(float(state.min_exposure), t_override)
            else:
                q_est = model * max(0.05, state.scale)
                t_need = threshold * f0t0 / max(1e-9, state.flux[i] * q_est * (DEDICATED_SAFETY if safety is None else safety))
            t_need = max(float(state.min_exposure), t_need)
            if max_t is not None and t_need > max_t:
                return
            cap = min(float(state.max_exposure), up, seconds_left)
            if deadline is not None:
                cap = min(cap, (deadline - now).total_seconds())
            if t_need > cap:
                return
            prev = special.get(i)
            if prev is not None:
                prev["t_need"] = max(prev["t_need"], t_need)
                if prio > prev["prio"]:
                    prev["prio"] = prio
                if deadline is not None and (prev["deadline"] is None or deadline < prev["deadline"]):
                    prev["deadline"] = deadline
                return
            special[i] = {"t_need": t_need, "prio": prio, "kind": kind,
                          "alt": alt, "az": az, "model": model, "up": up, "deadline": deadline}

        for req in self.active_reqs:
            share = req["reward"] / max(1, req["remaining"])
            for i in req["targets_left"]:
                consider(i, req["threshold"], "request", 1000.0 + share, req["deadline"])
        for i in state.required_undone:
            if level >= 2 and not (NEAR_MISS_LO <= state.factor[i] < scoring.required_threshold):
                # Survival pace keeps only the near-miss band: targets one
                # well-sized exposure away from crossing. Wide-open rescues of
                # faint targets are exactly the schedule poison the endgame
                # gates below exist to prevent.
                continue
            if not self._req_rescuable[i]:
                continue
            nights_left = max(1, state.last_night[i] - night_index + 1)
            near = NEAR_MISS_LO <= state.factor[i] < scoring.required_threshold
            # A near miss (factor already 0.3-0.5) is one right-sized exposure
            # away from crossing. Mid-survey that retry displaces multi-special
            # fields and pollutes the schedule (measured -4600 summed over the
            # bench cards), so the extra machinery only arms in the endgame,
            # where opportunity cost is nil. Everything before that stays on
            # the exact v6b code path.
            endgame = nights_left <= NEAR_MISS_ENDGAME_NIGHTS
            survey_tail = len(state.nights) - night_index <= SURVEY_TAIL_NIGHTS
            if not (near and endgame and survey_tail):
                # Two failures without crossing means the flux/quality estimate
                # was optimistic; a third try only makes sense when nights run out.
                if state.attempts[i] >= 2 and nights_left > 2:
                    continue
                last = self.rescue_last_try.get(i)
                if last is not None and hours - last < RESCUE_RETRY_HOURS and nights_left > 2:
                    continue
                prio = 500.0 + scoring.required_penalty + state.weight[i] + 30.0 / nights_left
                # Early on, only cheap rescues are worth the quality dilution; when
                # nights run out, any physically possible attempt is +50 upside.
                max_t = RESCUE_MAX_T_NEED if nights_left > 8 else float(state.max_exposure)
                consider(i, scoring.required_threshold, "required", prio, None, max_t)
                continue
            if state.attempts[i] >= NEAR_MISS_MAX_ATTEMPTS:
                continue
            last = self.rescue_last_try.get(i)
            if last is not None and hours - last < NEAR_MISS_RETRY_HOURS:
                continue
            prio = 500.0 + scoring.required_penalty + state.weight[i] + 30.0 / nights_left
            if last is not None:
                # Starvation guard: the anchor pick is a static priority sort, so
                # a near miss stuck just below the top few anchors would wait
                # forever while its window closes (six platform-card near misses
                # at factor 0.42-0.50 were never retried for exactly this reason).
                prio += min(15.0, 0.02 * (hours - last))
            max_t = float(state.max_exposure)
            if state.best_dur[i] > 0:
                # Realized-data retry: scale the exposure that produced the
                # current best factor. Precise where the model estimate is not.
                t_retry = state.best_dur[i] * (NEAR_MISS_AIM / max(1e-9, state.factor[i])) * 1.15
                # A retry that mathematically exceeds the cap is still worth the
                # capped exposure: the next night's quality may be better than
                # the night behind the best factor. Rejecting it outright left a
                # 0.479-at-3000s target untried for its last 40 nights while the
                # very same night at 3600s would have crossed the threshold.
                t_retry = min(t_retry, float(state.max_exposure))
                consider(i, scoring.required_threshold, "required", prio, None, max_t, t_override=t_retry)
            else:
                # No duration history: aim above the line with a cushion --
                # platform evidence showed first tries landing at 0.47-0.50.
                consider(i, NEAR_MISS_AIM, "required", prio, None, max_t, safety=NEAR_MISS_SAFETY)
        if not special:
            return None

        # Dedicated exposures are quality-diluting by design (they stretch into
        # worse slots for the lump sum). Cap how many run per night so the survey
        # schedule stays science-dominated; a night with an open request is exempt
        # (there are only ~2 requests per card, each a guaranteed +100).
        has_request = any(sp["kind"] == "request" for sp in special.values())
        if not has_request and self._dedicated_tonight.get(night_index, 0) >= DEDICATED_PER_NIGHT:
            return None

        # A field whose longest need eats most of an hour is only worth it when
        # the targets are running out of nights; early on, better conditions come.
        if not any(sp["t_need"] <= 2700.0 or sp["kind"] == "request"
                   or (state.last_night[i] - night_index + 1) <= 5
                   for i, sp in special.items()):
            return None

        ordered = sorted(special.items(), key=lambda kv: (-kv[1]["prio"], kv[1]["t_need"]))
        # The dedicated anchor search costs O(anchors x fibres x neighbours) and
        # runs before every plan pass; on 100-fibre year-long cards it alone can
        # eat a third of the wall clock, so pace pressure narrows it too.
        n_anchors = (1 if level >= 2 else DEDICATED_MAX_ANCHORS // 2) if level >= 1 else DEDICATED_MAX_ANCHORS
        fibers = (range(self.grid.n) if level < 2
                  else tuple(range(0, self.grid.n, max(1, self.grid.n // 4))) or (0,))
        best = None  # (key, c_alt, c_az, chosen)
        for anchor, spec in ordered[:n_anchors]:
            a_alt, a_az = spec["alt"], spec["az"]
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], NEIGHBOUR_RADIUS_DEG)
                    if j in special or state.factor[j] < DONE_FACTOR or self.request_bonus.get(j, 0.0) > 0.0]
            if level >= 1 and len(near) > 48:
                near.sort(key=lambda j: -self._fill_value(j))
                del near[48:]
            for fiber in fibers:
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not (state.min_alt + 1.5 <= c_alt <= 89.0):
                    continue
                c_alt = round(c_alt, 4)
                c_az = round(c_az, 4) % 360.0
                chosen: dict[int, tuple[float, int, float]] = {}
                for j in near:
                    v = BIG_SPECIAL + special[j]["prio"] if j in special else self._fill_value(j)
                    if v <= 0.0:
                        continue
                    alt, az = radec_to_altaz(state.ra[j], state.dec[j], lst, state.lat)
                    if alt < state.min_alt + 0.3:
                        continue
                    offsets = tangent_offsets(alt, az, c_alt, c_az)
                    if offsets is None:
                        continue
                    fib, margin = self.grid.classify(*offsets)
                    if fib is None:
                        continue
                    score = v if j in special else v * (1.0 if margin >= EDGE_MARGIN_DEG * (1 + 1.5 * state.misses[j]) else 0.4)
                    existing = chosen.get(fib)
                    if existing is None or score > existing[0]:
                        chosen[fib] = (score, j, margin)
                n_special = sum(1 for _, j, _ in chosen.values() if j in special)
                if n_special == 0:
                    continue
                key = (n_special, sum(s for s, _, _ in chosen.values()))
                if best is None or key > best[0]:
                    best = (key, c_alt, c_az, chosen)
        if best is None:
            return None
        _, c_alt, c_az, chosen = best

        t_need = 0.0
        for _, j, _ in chosen.values():
            sp = special.get(j)
            if sp is not None:
                t_need = max(t_need, sp["t_need"])
        duration = int(math.ceil(t_need / 30.0) * 30)
        duration = int(max(state.min_exposure, min(state.max_exposure, duration)))
        duration = min(duration, int(seconds_left))
        if duration < state.min_exposure:
            return None
        for fiber in [f for f, (_, j, _) in chosen.items()
                      if j in special and (special[j]["up"] < duration
                                           or (special[j]["deadline"] is not None
                                               and (now + timedelta(seconds=duration)) > special[j]["deadline"]))]:
            del chosen[fiber]
        if not any(j in special for _, j, _ in chosen.values()):
            return None

        for _, j, _ in chosen.values():
            if j in special and special[j]["kind"] == "required":
                self.rescue_last_try[j] = hours
        self._dedicated_tonight[night_index] = self._dedicated_tonight.get(night_index, 0) + 1
        return self._finish_dedicated(now, lst, c_alt, c_az, chosen, duration, moon, night_index, special)

    def _finish_dedicated(self, now, lst, c_alt, c_az, chosen, duration, moon, night_index, special):
        state = self.state
        scoring = state.scoring
        info: dict[int, dict] = {}
        for fiber, (_, j, _margin) in chosen.items():
            sp = special.get(j)
            if sp is not None:
                alt, az, model, up = sp["alt"], sp["az"], sp["model"], sp["up"]
            else:
                alt, az = radec_to_altaz(state.ra[j], state.dec[j], lst, state.lat)
                lunar = lunar_factor(moon, state.ra[j], state.dec[j], scoring.lunar_model)
                model = scoring.quality_model(alt, lunar) or 0.0
                ha = wrap180(lst - state.ra[j])
                up = (state.hmax[j] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[j] < 180 else 1e9
            k = (state.flux[j] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            info[fiber] = {"i": j, "alt": alt, "az": az, "model": model, "up": up, "k": k}

        assignments: dict[str, str] = {}
        for fiber, item in info.items():
            if item["up"] >= duration:
                assignments[str(fiber)] = state.ids[item["i"]]
        if not assignments:
            return None
        if not any(item["i"] in special for fiber, item in info.items() if str(fiber) in assignments):
            return None

        band_scale = (state.scale / 0.95) * state.band_bias
        votes = {"DARK": 0.0, "BRIGHT": 0.0, "BACKUP": 0.0}
        for fiber, item in info.items():
            if str(fiber) not in assignments:
                continue
            band = scoring.program_band(item["model"] * band_scale)
            votes[band] += state.weight[item["i"]] * min(1.0, item["k"] * duration) + \
                (REQUIRED_BONUS * 0.02 if state.required[item["i"]] else 0.0)
        program, best_score = "BACKUP", float("-inf")
        for name in ("DARK", "BRIGHT", "BACKUP"):
            matched = votes[name] * scoring.program_multipliers.get(name, 1.0)
            mismatched = (votes["DARK"] + votes["BRIGHT"] + votes["BACKUP"] - votes[name]) * scoring.mismatch_multiplier
            score = matched + mismatched
            if score > best_score:
                best_score, program = score, name

        clean = not state.all_sky_notice()
        state.pending.clear()
        for fiber, item in info.items():
            if str(fiber) in assignments:
                state.pending[state.ids[item["i"]]] = PendingPrediction(
                    model=item["model"], band_model=item["model"] / 0.95, alt=item["alt"], az=item["az"],
                    clean=clean and self._direction_factor(item["alt"], item["az"]) >= 1.0,
                    scale=state.scale,
                )
        state.pending_program = program
        state.pending_duration = duration
        state.pending_night = night_index

        kinds = {special[j]["kind"] for _, j, _ in chosen.values() if j in special}
        goal = "+".join(sorted(kinds)) if kinds else "fill"
        self.log(f"planner: dedicated {goal} exposure at alt={c_alt:.1f} az={c_az:.1f} "
                 f"for {duration}s covering {sum(1 for _, j, _ in chosen.values() if j in special)} special target(s)")

        return {
            "action": "observe",
            "pointing": {"alt_deg": c_alt, "az_deg": c_az},
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
            "goal": goal,
        }

    # -- planning value / achievability -----------------------------------------

    def _direction_factor(self, alt: float, az: float) -> float:
        state = self.state
        for direction in state.terrain:
            if direction in DIRECTION_AZ and alt < 50.0 and _az_distance(az, DIRECTION_AZ[direction]) <= 60.0:
                return 0.0
        factor = 1.0
        for key in state.notices:
            kind, _, direction = key.partition("|")
            if direction not in DIRECTION_AZ:
                continue
            near = _az_distance(az, DIRECTION_AZ[direction]) <= 67.5
            if kind in BLOCKING_KINDS and near and alt < 62.0:
                return 0.0
            if near and alt < 75.0:
                factor = min(factor, 0.35)
        for direction in state.extra_avoid:
            if direction in DIRECTION_AZ and _az_distance(az, DIRECTION_AZ[direction]) <= 67.5 and alt < 70.0:
                factor = min(factor, LLM_AVOID_FACTOR)
        for blocked_az, blocked_alt in state.blocked[-40:]:
            if _az_distance(az, blocked_az) <= 12.0 and alt <= blocked_alt + 3.0:
                factor = min(factor, 0.2)
        return factor

    def _value(self, i: int) -> float:
        """Planning value of fully completing target i from here (ignores how much
        exposure is achievable tonight)."""
        state = self.state
        f = state.factor[i]
        damp = 0.6 ** state.misses[i]
        threshold = state.scoring.required_threshold
        bonus = self.request_bonus.get(i, 0.0)
        if state.required[i]:
            if f >= threshold:
                return state.weight[i] * max(0.0, 1.0 - f * f) * damp + bonus
            return (state.weight[i] * (1.0 - f * f) + REQUIRED_BONUS * (1.0 if f < 0.5 else 0.35)) * damp + bonus
        return bonus if f >= DONE_FACTOR else state.weight[i] * (1.0 - f * f) * damp + bonus

    # -- main planning pass -------------------------------------------------------

    def plan(self, now, night_end, night_index: int, hours: float):
        state = self.state
        state.update_scale(hours)
        lst = local_sidereal_deg(now, state.lon)
        horizon = min(night_end, state.survey_end)
        seconds_left = (horizon - now).total_seconds()
        if seconds_left < state.min_exposure:
            return None
        min_visible = min(MIN_VISIBLE_SECONDS, seconds_left) * SIDEREAL_DEG_PER_SECOND

        # Cached planning values: _value(i) only moves when a target's factor,
        # miss count or request bonus moves, which is a handful of targets per
        # decision -- not all ~45k live ones. The scan below reorders nothing:
        # it returns exactly the values _value(i) would.
        if self._dirty_values:
            cache = self._value_cache
            for i in self._dirty_values:
                cache[i] = self._value(i)
            self._dirty_values.clear()
        value = self._value_cache.get

        still_active = []
        candidates: list[tuple[float, int]] = []
        ra, hmax = state.ra, state.hmax
        weight, factor = state.weight, state.factor
        last_night = state.last_night
        append = candidates.append
        for i in state.active:
            v = value(i, -1.0)
            if v < 0.0:
                v = self._value(i)
                self._value_cache[i] = v
            if v <= 0.0:
                continue
            still_active.append(i)
            ha = lst - ra[i]
            if ha > 180.0:
                ha -= 360.0
            elif ha < -180.0:
                ha += 360.0
            h = hmax[i]
            if -h <= ha <= h - min_visible:
                nights_left = max(1, last_night[i] - night_index + 1)
                setting = (1.0 + 0.5 * max(0.0, ha / h)) if h < 180 else 1.0
                append((v * (1.0 + 2.0 / nights_left) * setting, i))
        state.active = still_active
        if not candidates:
            return None
        candidates.sort(key=lambda t: -t[0])

        moon = Moon(now + timedelta(seconds=450), lst, state.lat)
        altaz_cache: dict[int, tuple[float, float]] = {}

        def altaz(i: int) -> tuple[float, float]:
            cached = altaz_cache.get(i)
            if cached is None:
                cached = radec_to_altaz(state.ra[i], state.dec[i], lst, state.lat)
                altaz_cache[i] = cached
            return cached

        visible = {i for _, i in candidates}
        achievable_cache: dict[int, float] = {}
        scoring = state.scoring

        def achievable(i: int) -> float:
            cached = achievable_cache.get(i)
            if cached is not None:
                return cached
            alt, az = altaz(i)
            lunar = lunar_factor(moon, state.ra[i], state.dec[i], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            k = (state.flux[i] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            ha = wrap180(lst - state.ra[i])
            up = (state.hmax[i] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[i] < 180 else 1e9
            reach = min(1.0, k * min(state.max_exposure, up, seconds_left))
            f = state.factor[i]
            gain = state.weight[i] * max(0.0, reach * reach - f * f)
            if state.required[i] and f < scoring.required_threshold and reach >= scoring.required_threshold:
                gain += REQUIRED_BONUS
            gain += self.request_bonus.get(i, 0.0) * (1.0 if reach >= scoring.required_threshold else 0.0)
            # Uncompleted required targets must not be buried after failed tries --
            # but only when a retry can physically still cross the threshold.
            if state.required[i] and state.factor[i] < scoring.required_threshold and RESCUE_DAMP and self._req_rescuable[i]:
                damp = (0.6 ** state.misses[i]) * (0.9 ** state.attempts[i])
            else:
                damp = (0.6 ** state.misses[i]) * (0.7 ** state.attempts[i])
            result = gain * damp * self._direction_factor(alt, az)
            achievable_cache[i] = result
            return result

        anchors: list[tuple[float, int]] = []
        for checked, (priority, i) in enumerate(candidates):
            if checked >= ANCHOR_POOL and len(anchors) >= 3 * ANCHORS:
                break
            weighted = achievable(i) * priority / max(1e-9, value(i, self._value(i)))
            if weighted > 0:
                anchors.append((weighted, i))
        if not anchors:
            return None
        anchors.sort(key=lambda t: -t[0])

        level = state.fast_level
        n_anchors = ANCHORS if level == 0 else (3 if level == 1 else 2)
        # The fibre-fill loop is O(anchors x fibres x neighbours) in trig-heavy
        # geometry; under pace pressure only the top-valued neighbours can win a
        # fibre anyway, so cap the fill set before any offsets are computed.
        fill_cap = None if level == 0 else (64 if level == 1 else 32)
        # Level 2 probes a spread of fibres instead of the full grid; the old
        # hard-coded (5,6,9,10) would index past a 9-fibre card and crash.
        fibers = (range(self.grid.n) if level < 2
                  else tuple(range(0, self.grid.n, max(1, self.grid.n // 4))) or (0,))
        best = None  # (total, c_alt, c_az, chosen)
        tried = 0
        for _, anchor in anchors:
            if tried >= n_anchors and best is not None:
                break
            if tried >= n_anchors + 8:
                break
            tried += 1
            a_alt, a_az = altaz(anchor)
            near = [j for j in state.neighbours(state.ra[anchor], state.dec[anchor], NEIGHBOUR_RADIUS_DEG) if j in visible]
            if fill_cap is not None and len(near) > fill_cap:
                near.sort(key=lambda j: -value(j, 0.0))
                del near[fill_cap:]
            near_values = {j: achievable(j) for j in near}
            for fiber in fibers:
                d_north, d_east = self.grid.fiber_center(fiber)
                c_alt, c_az = shift_altaz(a_alt, a_az, -d_north, -d_east)
                if not (state.min_alt + 1.5 <= c_alt <= 89.0):
                    continue
                c_alt = round(c_alt, 4)
                c_az = round(c_az, 4) % 360.0
                chosen: dict[int, tuple[float, int, float]] = {}  # fiber -> (score, j, margin)
                for j, v in near_values.items():
                    if v <= 0.0:
                        continue
                    alt, az = altaz(j)
                    offsets = tangent_offsets(alt, az, c_alt, c_az)
                    if offsets is None:
                        continue
                    fib, margin = self.grid.classify(*offsets)
                    if fib is None:
                        continue
                    score = v * (1.0 if margin >= EDGE_MARGIN_DEG * (1 + 1.5 * state.misses[j]) else 0.4)
                    existing = chosen.get(fib)
                    if existing is None or score > existing[0]:
                        chosen[fib] = (score, j, margin)
                if not chosen:
                    continue
                total = sum(score for score, _, _ in chosen.values())
                if best is None or total > best[0]:
                    best = (total, c_alt, c_az, chosen)
        if best is None:
            return None
        _, c_alt, c_az, chosen = best
        return self._finish_plan(now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index)

    def _finish_plan(self, now, lst, c_alt, c_az, chosen, seconds_left, moon, altaz, hours, night_index):
        state = self.state
        scoring = state.scoring
        c_ra, c_dec = altaz_to_radec(c_alt, c_az, lst, state.lat)
        c_hmax = max_hour_angle_deg(c_dec, state.lat, state.min_alt + 0.3)
        c_ha = wrap180(lst - c_ra)

        info: dict[int, dict] = {}
        for fiber, (_, j, _margin) in chosen.items():
            alt, az = altaz(j)
            lunar = lunar_factor(moon, state.ra[j], state.dec[j], scoring.lunar_model)
            model = scoring.quality_model(alt, lunar) or 0.0
            ha = wrap180(lst - state.ra[j])
            up = (state.hmax[j] - ha) / SIDEREAL_DEG_PER_SECOND if state.hmax[j] < 180 else 1e9
            k = (state.flux[j] * model * state.scale * PLAN_FACTOR_SAFETY) / scoring.f0t0
            info[fiber] = {"i": j, "alt": alt, "az": az, "model": model, "up": up, "k": k}
        center_up = (c_hmax - c_ha) / SIDEREAL_DEG_PER_SECOND if c_hmax < 180 else 1e9

        best = None  # (objective, duration)
        th = scoring.required_threshold
        # Pace decides the exposure objective AND the gain shape.
        # - Starved cards (level >= 1) pay for every DECISION (~0.15 s of the 900 s
        #   budget each) while night sim time is plentiful, so they want the most
        #   REAL score per decision. The scorer is linear in the completion factor
        #   (score = weight * factor * program multiplier), so the objective is the
        #   linear field gain, stopping at the shortest duration within 2% of the
        #   best -- that lands at field saturation instead of stretching for cents.
        # - Relaxed cards keep the classic rate objective on the quadratic gain:
        #   reach^2 penalizes shallow exposures, which is what keeps per-second
        #   efficiency high when wall clock is not the binding constraint
        #   (linear-under-rate was measured flooding L cards with 300 s hits).
        starved = state.fast_level >= 1
        curve: list[tuple[float, float]] = []  # (duration, gain)
        for base in DURATIONS:
            duration = round((base * state.duration_scale) / 30.0) * 30
            duration = int(max(state.min_exposure, min(state.max_exposure, duration)))
            if duration > seconds_left or duration > center_up:
                continue
            gain = 0.0
            for item in info.values():
                if item["up"] < duration:
                    continue
                reached = min(1.0, item["k"] * duration)
                f = state.factor[item["i"]]
                if starved:
                    gain += state.weight[item["i"]] * max(0.0, reached - f)
                else:
                    gain += state.weight[item["i"]] * max(0.0, reached * reached - f * f)
                if state.required[item["i"]] and f < th and reached >= th:
                    # True marginal value of crossing the threshold: the bonus above
                    # the line plus the avoided end-of-survey penalty. Fainter
                    # targets that can never cross keep the plain bonus.
                    gain += REQUIRED_BONUS + (scoring.required_penalty if RESCUE_DUR and self._req_rescuable[item["i"]] else 0.0)
                if reached >= self.request_threshold.get(item["i"], th):
                    gain += self.request_bonus_full.get(item["i"], 0.0)
            curve.append((duration, gain))
        if not curve:
            return None
        if starved:
            max_gain = max(g for _, g in curve)
            duration = next(d for d, g in curve if g >= 0.98 * max_gain)
            best = (max_gain, duration)
        else:
            best = max(((g / d, d) for d, g in curve), key=lambda t: t[0])
        duration = best[1]
        if best[0] <= 0.0:
            if state.has_recent_sample(hours):
                return None
            fallback = next((d for d in (900, 600, 300) if d <= seconds_left and d <= center_up), None)
            if fallback is None:
                return None
            duration = fallback

        assignments: dict[str, str] = {}
        for fiber, item in info.items():
            if item["up"] >= duration:
                assignments[str(fiber)] = state.ids[item["i"]]
        if not assignments:
            return None

        band_scale = (state.scale / 0.95) * state.band_bias
        votes = {"DARK": 0.0, "BRIGHT": 0.0, "BACKUP": 0.0}
        for fiber, item in info.items():
            if str(fiber) not in assignments:
                continue
            band = scoring.program_band(item["model"] * band_scale)
            votes[band] += state.weight[item["i"]] * min(1.0, item["k"] * duration) + \
                (REQUIRED_BONUS * 0.02 if state.required[item["i"]] else 0.0)
        program, best_score = "BACKUP", float("-inf")
        for name in ("DARK", "BRIGHT", "BACKUP"):
            matched = votes[name] * scoring.program_multipliers.get(name, 1.0)
            mismatched = (votes["DARK"] + votes["BRIGHT"] + votes["BACKUP"] - votes[name]) * scoring.mismatch_multiplier
            score = matched + mismatched
            if score > best_score:
                best_score, program = score, name
        if state.force_program:
            program = state.force_program

        clean = not state.all_sky_notice()
        state.pending.clear()
        for fiber, item in info.items():
            if str(fiber) in assignments:
                state.pending[state.ids[item["i"]]] = PendingPrediction(
                    model=item["model"], band_model=item["model"] / 0.95, alt=item["alt"], az=item["az"],
                    clean=clean and self._direction_factor(item["alt"], item["az"]) >= 1.0,
                    scale=state.scale,
                )
        state.pending_program = program
        state.pending_duration = duration
        state.pending_night = night_index

        return {
            "action": "observe",
            "pointing": {"alt_deg": c_alt, "az_deg": c_az},
            "assignments": assignments,
            "duration_seconds": duration,
            "program": program,
        }
