"""Survey state: the target catalogue, learned sky-quality scale, and per-target
progress. Built once from `initialize`, then updated from every `decision_request`'s
messages and `last_result`. Holds no hidden data -- only what the public protocol
hands us, plus what we infer from our own hits (never from a file).

Mirrors the structure-of-parallel-arrays design (ids/ra/dec/flux/weight/required/...,
all indexed by the same integer i) used by this project's companion TypeScript example,
so both examples solve the same problem the same way and can be compared directly.
"""
from __future__ import annotations

import bisect
import math
import os
from collections import deque
from typing import NamedTuple, Optional

from .geometry import FiberGrid, max_hour_angle_deg, parse_utc, wrap180
from .scoring import ScoringModel

ALT_MARGIN_DEG = 0.6
SKY_MEMORY_HOURS = 2.0
RECENT_SAMPLES = 60
EARLIER_SAMPLES = 60
MIN_NIGHTS_FOR_EVIDENCE = 10
RISE_STEP_PER_NIGHT = float(os.environ.get("SAC_RISE_STEP", "0.40"))
SIDEREAL_DEG_PER_SECOND = 360.98564736629 / 86400.0


class PendingPrediction(NamedTuple):
    model: float          # lunar/airmass quality model used at planning time
    band_model: float      # model / 0.95, used for program-band back-estimation
    alt: float
    az: float
    clean: bool            # true when no all-sky notice / directional block applied at plan time
    scale: float = 1.0     # sky scale in effect when planned (for band calibration)


class FaultEvidence(NamedTuple):
    recent_median: float
    earlier_median: float   # v8: healthy reference (median of the top half of past nights)
    drop: float             # v8: median of the last three nightly medians / healthy reference
    recent_samples: int
    recent_nights: int
    earlier_samples: int
    dark_checks: int
    dark_matched: int
    recent_nightly: tuple = ()   # the last three nightly medians, oldest first
    quad_spread: float = 1.0     # min/max azimuth-quadrant median in the recent window


def _mod(a: float, n: float) -> float:
    m = a % n
    return m + n if m < 0 else m


class SurveyState:
    def __init__(self, init_payload: dict):
        site = init_payload["site"]
        survey = init_payload["survey"]
        instrument = init_payload["instrument"]
        limits = init_payload.get("limits", {})

        self.lat = float(site["latitude_deg"])
        self.lon = float(site["longitude_deg"])
        self.min_alt = float(site.get("minimum_altitude_deg", 30.0))
        self.sun_altitude_limit_deg = float(site.get("sun_altitude_limit_deg", -18.0))

        self.survey_start = parse_utc(survey["start_utc"])
        self.survey_end = parse_utc(survey["end_utc"])
        self.slot_seconds = int(survey.get("slot_seconds", 900))
        self.nights = [(parse_utc(n["observing_start_utc"]), parse_utc(n["observing_end_utc"]))
                       for n in survey.get("nights", [])]

        self.fiber_grid = FiberGrid(instrument)
        exposure = instrument.get("exposure", {})
        self.min_exposure = int(exposure.get("min_duration_seconds", 60))
        self.max_exposure = int(exposure.get("max_duration_seconds", 3600))

        self.scoring = ScoringModel(init_payload.get("scoring", {}), site)

        reporting = init_payload.get("scoring", {}).get("reporting", {})
        self.max_consecutive_reports = int(reporting.get("max_consecutive_reports", limits.get("max_consecutive_reports", 32)))
        self.false_report_free_allowance = int(reporting.get("false_report_free_allowance", 0))
        self.response_max_bytes = int(limits.get("response_max_bytes", 524288))

        # Parallel arrays, one slot per target, in catalogue order.
        self.ids: list[str] = []
        self.ra: list[float] = []
        self.dec: list[float] = []
        self.flux: list[float] = []
        self.weight: list[float] = []
        self.required: list[bool] = []
        self.index_of: dict[str, int] = {}

        columns = init_payload.get("targets", {}).get("columns", [])
        col = {name: idx for idx, name in enumerate(columns)}
        for row in init_payload.get("targets", {}).get("rows", []):
            target_id = str(row[col["target_id"]])
            self.index_of[target_id] = len(self.ids)
            self.ids.append(target_id)
            self.ra.append(float(row[col["ra_deg"]]))
            self.dec.append(float(row[col["dec_deg"]]))
            self.flux.append(float(row[col["feature_flux"]]))
            self.weight.append(float(row[col["science_weight"]]))
            self.required.append(bool(row[col["required"]]))

        n = len(self.ids)
        self.hmax = [max_hour_angle_deg(self.dec[i], self.lat, self.min_alt + ALT_MARGIN_DEG) for i in range(n)]
        self.factor = [0.0] * n
        self.best_dur = [0] * n  # exposure seconds behind the current best factor
        self.misses = [0] * n
        self.attempts = [0] * n
        self.active = [i for i in range(n) if self.hmax[i] > 0.0]

        self._cells: dict[int, list[tuple[float, int]]] = {}
        self._build_index()
        self.first_night, self.last_night = self._build_windows()

        self.scale = 1.0
        self.prior_scale = 1.0
        self._samples: deque = deque(maxlen=24)           # (hours, ratio)
        self._all_ratios: deque = deque(maxlen=400)        # ratio
        self.clean_history: list[tuple[float, int, float]] = []  # (hours, night, ratio)
        self.quality_log: deque = deque(maxlen=20000)      # (hours, night, ratio, clean, az_quad)
        self.last_quake_at = None                          # datetime of the latest earthquake bulletin seen
        self.pending_night = -1
        self._band_checks: deque = deque(maxlen=60)        # (program, matched, model)
        self.band_bias = 1.0  # closed-loop program declaration bias (see on_result)
        self.force_program: Optional[str] = None
        self.pending: dict[str, PendingPrediction] = {}
        self.pending_program = "BACKUP"
        self.pending_duration = 0
        self.blocked: list[tuple[float, float]] = []       # (az, alt) where a hit scored zero
        self.notices: set[str] = set()                      # "kind|direction"
        self.terrain: set[str] = set()
        self.extra_avoid: set[str] = set()
        self.duration_scale = 1.0
        self.fast_level = 0

    # -- spatial index -------------------------------------------------------

    def _build_index(self) -> None:
        for i in self.active:
            key = math.floor(self.dec[i])
            self._cells.setdefault(key, []).append((self.ra[i], i))
        for band in self._cells.values():
            band.sort(key=lambda pair: pair[0])

    def neighbours(self, ra: float, dec: float, radius: float):
        """Indices within `radius` degrees of (ra, dec), using the 1-degree declination-band index."""
        found: list[int] = []
        cos_dec = max(0.05, math.cos(math.radians(min(89.0, abs(dec) + radius))))
        width = radius / cos_dec
        lo_key, hi_key = math.floor(dec - radius), math.floor(dec + radius)
        for key in range(lo_key, hi_key + 1):
            band = self._cells.get(key)
            if not band:
                continue
            spans: list[tuple[float, float]]
            lo, hi = ra - width, ra + width
            if lo < 0:
                spans = [(0.0, hi), (lo + 360.0, 360.0)]
            elif hi >= 360:
                spans = [(lo, 360.0), (0.0, hi - 360.0)]
            else:
                spans = [(lo, hi)]
            keys = [r for r, _ in band]
            for low, high in spans:
                start = bisect.bisect_left(keys, low)
                end = bisect.bisect_right(keys, high)
                for k in range(start, end):
                    found.append(band[k][1])
        return found

    def _build_windows(self):
        """First/last night index on which each target has >=20 minutes above the limit."""
        need = 20 * 60 * SIDEREAL_DEG_PER_SECOND
        spans = []
        for start, end in self.nights:
            from .geometry import local_sidereal_deg
            l0 = local_sidereal_deg(start, self.lon)
            span = (end - start).total_seconds() * SIDEREAL_DEG_PER_SECOND
            spans.append((l0, span))
        n = len(self.ra)
        first_night = [len(self.nights)] * n
        last_night = [-1] * n
        for i in self.active:
            h = self.hmax[i]
            for k, (l0, span) in enumerate(spans):
                if h >= 180.0:
                    overlap = span
                else:
                    a = _mod(self.ra[i] - h - l0, 360.0)
                    overlap = max(0.0, min(span, a + 2 * h) - a) + max(0.0, min(span, a - 360.0 + 2 * h))
                if overlap >= need:
                    if first_night[i] > k:
                        first_night[i] = k
                    last_night[i] = k
        return first_night, last_night

    # -- messages and results -------------------------------------------------

    def on_messages(self, messages: list[dict], latest_bulletin: Optional[dict]) -> None:
        for message in messages:
            if message.get("record_type") == "bulletin" and message.get("initial"):
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "terrain_obstruction":
                        self.terrain.add(notice.get("direction"))
            elif message.get("record_type") == "state_resync":
                self._resync(message.get("observed_target_ids", []), message.get("best_scores", []))
            # Earthquake bulletins mark efficiency drops a report cannot fix; the
            # planner uses this to avoid wasting the false-report budget on them.
            if message.get("record_type") == "bulletin":
                for notice in message.get("notices", []):
                    if notice.get("event_kind") == "earthquake":
                        when = parse_utc(message.get("issued_at_utc") or "")
                        if when is not None and (self.last_quake_at is None or when > self.last_quake_at):
                            self.last_quake_at = when
        notices = (latest_bulletin or {}).get("notices", [])
        self.notices = {f"{n.get('event_kind')}|{n.get('direction')}" for n in notices
                        if n.get("event_kind") != "terrain_obstruction"}

    def _resync(self, observed_ids: list, best_scores) -> None:
        best: dict[str, float] = {}
        if best_scores and isinstance(best_scores[0], dict):
            for row in best_scores:
                best[row.get("target_id")] = float(row.get("best_score", 0.0))
        else:
            for target_id, score in zip(observed_ids, best_scores):
                best[target_id] = float(score)
        top_multiplier = max(self.scoring.program_multipliers.values()) if self.scoring.program_multipliers else 1.2
        for i in range(len(self.ids)):
            score = best.get(self.ids[i], 0.0)
            new_factor = min(1.0, score / (self.weight[i] * top_multiplier)) if score > 0 and self.weight[i] > 0 else 0.0
            if new_factor < self.factor[i] - 0.05:
                # Data loss undid the exposures: they never happened, so the
                # failed-attempt counters that gate the rescue pass must reset too.
                self.attempts[i] = 0
            self.factor[i] = new_factor
        self.active = [i for i in range(len(self.ids)) if self.hmax[i] > 0.0]
        self.pending.clear()

    def site_closed(self) -> bool:
        for key in self.notices:
            kind, _, direction = key.partition("|")
            if kind in ("rain", "storm") and direction == "ALL":
                return True
        return False

    def all_sky_notice(self) -> bool:
        return any(key.partition("|")[2] == "ALL" for key in self.notices)

    def on_result(self, last_result: Optional[dict], hours: float) -> None:
        if not last_result or last_result.get("action") != "observe" or not self.pending:
            self.pending.clear()
            return
        hits = {h.get("target_id"): float(h.get("score", 0.0)) for h in last_result.get("hits", [])}
        any_positive = any(score > 0 for score in hits.values())
        scoring = self.scoring
        multipliers = scoring.program_multipliers
        mismatch = scoring.mismatch_multiplier
        declared_multiplier = multipliers.get(self.pending_program, 1.0)
        f0t0 = scoring.f0t0

        clean_matched = 0
        clean_mismatched = 0
        for target_id, prediction in self.pending.items():
            i = self.index_of.get(target_id)
            if i is None:
                continue
            if target_id not in hits:
                self.misses[i] += 1
                continue
            score = hits[target_id]
            if score <= 0.0:
                if any_positive:
                    self.blocked.append((prediction.az, prediction.alt))
                continue
            weight = self.weight[i] if self.weight[i] > 0 else 1e-9
            multiplier_seen = score / weight
            if prediction.clean:
                if abs(multiplier_seen - declared_multiplier) < 2e-4:
                    self._band_checks.append((self.pending_program, True, prediction.model))
                    clean_matched += 1
                elif abs(multiplier_seen - mismatch) < 2e-4:
                    self._band_checks.append((self.pending_program, False, prediction.model))
                    clean_mismatched += 1
            factor_if_match = score / (weight * declared_multiplier) if declared_multiplier > 0 else 0.0
            factor_if_miss = score / (weight * mismatch) if mismatch > 0 else 0.0
            ratio_match = (factor_if_match * f0t0) / (self.flux[i] * self.pending_duration * prediction.model) \
                if self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0 else 0.0
            band = scoring.program_band(ratio_match * prediction.band_model)
            matched = band == self.pending_program
            factor = factor_if_match if matched else factor_if_miss
            if min(1.0, factor) > self.factor[i]:
                self.best_dur[i] = self.pending_duration
            self.factor[i] = max(self.factor[i], min(1.0, factor))
            if self.required[i] and self.factor[i] < scoring.required_threshold:
                self.attempts[i] += 1
            if factor < 0.97 and self.flux[i] > 0 and self.pending_duration > 0 and prediction.model > 0:
                ratio = (factor * f0t0) / (self.flux[i] * self.pending_duration * prediction.model)
                self._samples.append((hours, ratio))
                self._all_ratios.append(ratio)
                self.quality_log.append((hours, self.pending_night, ratio, prediction.clean,
                                         int((prediction.az % 360.0) // 90)))
                if prediction.clean:
                    self.clean_history.append((hours, self.pending_night, ratio))
        self.pending.clear()
        # Closed-loop declaration bias, once per exposure: a mostly-mismatched
        # BACKUP/BRIGHT declaration means the actual bands were better than declared
        # (the band formula excludes instrument efficiency, but the learned scale
        # absorbs it, skewing predictions low); a mostly-mismatched DARK means the
        # opposite. Only mismatch outcomes carry this signal, not the ratio samples.
        if clean_mismatched > clean_matched and clean_matched + clean_mismatched >= 2:
            if self.pending_program in ("BACKUP", "BRIGHT"):
                self.band_bias = min(1.35, self.band_bias * 1.04)
            elif self.pending_program == "DARK":
                self.band_bias = max(0.8, self.band_bias * 0.96)
        self.update_scale(hours)

    def has_recent_sample(self, hours: float) -> bool:
        return any(when >= hours - SKY_MEMORY_HOURS for when, _ in self._samples)

    def update_scale(self, hours: float) -> None:
        if len(self._all_ratios) >= 8:
            ordered = sorted(self._all_ratios)
            self.prior_scale = ordered[len(ordered) // 2]
        recent = sorted(ratio for when, ratio in self._samples if when >= hours - SKY_MEMORY_HOURS)
        self.scale = max(0.05, recent[len(recent) // 2]) if len(recent) >= 4 else self.prior_scale

    # -- fault diagnostics ------------------------------------------------------

    def night_medians(self, min_samples: int = 5) -> dict[int, float]:
        """Median quality ratio per completed night (nights with too few samples
        are weather-truncated and would mislead both the reference and the drop)."""
        by_night: dict[int, list[float]] = {}
        for _h, night, ratio, _clean, _az in self.quality_log:
            by_night.setdefault(night, []).append(ratio)
        return {n: sorted(v)[len(v) // 2] for n, v in by_night.items() if len(v) >= min_samples}

    def quality_recovering(self) -> bool:
        """True when the nightly medians are in a strong sustained climb --
        the signature of earthquake damage decaying, as opposed to a stuck fault."""
        return self.night_median_trend() == "rising"

    def night_median_trend(self) -> str:
        """Compare the last three night medians of the quality ratio log.

        A fault holds the ratio flat and low; earthquake damage decays in a
        steep monotonic nightly climb (the L3 quake recovered ~2x per night);
        weather on top of a stuck fault wobbles by tens of percent. "rising"
        therefore needs >RISE_STEP_PER_NIGHT on BOTH steps."""
        by_night: dict[int, list[float]] = {}
        for _hours, night, ratio, _clean, _az in self.quality_log:
            by_night.setdefault(night, []).append(ratio)
        nights = sorted(by_night)
        if len(nights) < 3:
            return "unknown"
        med = lambda k: sorted(by_night[k])[len(by_night[k]) // 2]  # noqa: E731
        m1, m2, m3 = (med(k) for k in nights[-3:])
        step = 1.0 + RISE_STEP_PER_NIGHT
        if m2 > step * m1 and m3 > step * m2:
            return "rising"
        return "flat"

    def fault_evidence(self) -> Optional[FaultEvidence]:
        """v8 detector signal, built from nightly medians rather than a pooled
        sample window.

        The pooled window failed twice on real cards: a weeks-long fault drags
        the all-history `earlier` median down with it (the L3 quake-masked fault
        never dropped below 0.79 of a poisoned baseline), and intra-night sample
        selection during storms made the pooled median report 0.17 on nights
        whose engine quality was normal (two paid false reports on formal D).
        The reference here is instead the median of the TOP HALF of past nights
        -- what the instrument demonstrably achieves in decent weather -- which
        faults and bad seasons do not drag down."""
        nightly = self.night_medians()
        if len(nightly) < MIN_NIGHTS_FOR_EVIDENCE:
            return None
        ordered = sorted(nightly.values())
        half = len(ordered) // 2
        top = ordered[-half:] if half else ordered  # the worse half is discarded
        ref = top[len(top) // 2]
        if ref <= 1e-9:
            return None
        recent_nightly = tuple(nightly[k] for k in sorted(nightly)[-3:])
        samples = [(n, r, az) for _h, n, r, _c, az in self.quality_log]
        recent_samples = samples[-RECENT_SAMPLES:]
        span_nights = len({n for n, _r, _az in recent_samples})
        quads: dict[int, list[float]] = {}
        for _n, r, az in recent_samples:
            quads.setdefault(az, []).append(r)
        qm = [sorted(v)[len(v) // 2] for v in quads.values() if len(v) >= 4]
        quad_spread = (min(qm) / max(qm)) if len(qm) >= 2 else 1.0
        recent_median = recent_nightly[len(recent_nightly) // 2]
        dark_line = self.scoring.program_bands["DARK"] * 1.3
        dark = [c for c in list(self._band_checks)[-16:]
                if c[0] == "DARK" and (c[2] * ref) / 0.95 >= dark_line]
        return FaultEvidence(
            recent_median=round(recent_median, 3),
            earlier_median=round(ref, 3),
            drop=round(recent_median / ref, 3),
            recent_samples=len(recent_samples),
            recent_nights=span_nights,
            earlier_samples=len(top),
            dark_checks=len(dark),
            dark_matched=sum(1 for c in dark if c[1]),
            recent_nightly=tuple(round(m, 3) for m in recent_nightly),
            quad_spread=round(quad_spread, 3),
        )

    def forget_quality_history(self) -> None:
        """After a correct report repairs the instrument: re-learn the sky scale
        from scratch (efficiency just jumped), but KEEP the quality log and band
        checks -- the v8 healthy reference is a top-half median, so the
        fault-period samples sitting in the discarded half cannot poison it, and
        keeping the history means the detector re-arms immediately instead of
        going blind for ten nights on a 30-night card."""
        self.clean_history = []
        self._samples.clear()
        self._all_ratios.clear()
        self.prior_scale = 1.0
        self.band_bias = 1.0

    # -- night lookup -------------------------------------------------------------

    def current_night(self, now):
        for index, (start, end) in enumerate(self.nights):
            if start <= now < end:
                return index, start, end
        return None

    def next_night_start(self, now):
        for start, _end in self.nights:
            if start > now:
                return start
        return None
