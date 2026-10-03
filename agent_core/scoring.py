"""Scoring estimates built only from the PUBLIC parts of initialize.payload.scoring.

The real config shape (confirmed against the platform's own TypeScript types, not
guessed): `q0`, `flux_zero_point`, `exposure_zero_point_seconds`, `airmass_exponent`,
`lunar_model`, `program.{bands,multipliers,mismatch_multiplier}`,
`required.{penalty_per_missing,observed_factor_threshold}`,
`uniformity.{weight,ra_band_width_deg,observed_factor_threshold}`,
`reporting.{correct_reward,false_penalty,false_report_free_allowance,max_consecutive_reports}`.
All of it is public -- it ships in `initialize`. The hidden sky terms (instrument
efficiency, transparency, cloud, seeing) are never read from any file; `SurveyState`
tracks a single learned "scale" from the agent's own hits instead (see state.py).
"""
from __future__ import annotations

from .geometry import normalized_airmass

DEFAULT_LUNAR_MODEL = {"maximum_penalty": 0.5, "altitude_exponent": 1.0, "angular_decay_scale_deg": 40.0}
DEFAULT_BANDS = {"DARK": 0.65, "BRIGHT": 0.40}
DEFAULT_MULTIPLIERS = {"DARK": 1.20, "BRIGHT": 1.12, "BACKUP": 1.06}
DEFAULT_MISMATCH_MULTIPLIER = 1.0
DEFAULT_REQUIRED_PENALTY = 50.0
DEFAULT_REQUIRED_THRESHOLD = 0.5


class ScoringModel:
    """Reads the public scoring config once and offers factor/score/band estimates."""

    def __init__(self, scoring_config: dict, site: dict):
        self.q0 = float(scoring_config.get("q0", 1.0))
        self.flux_zero_point = float(scoring_config.get("flux_zero_point", 0.5))
        self.exposure_zero_point_seconds = float(scoring_config.get("exposure_zero_point_seconds", 900.0))
        self.f0t0 = max(1e-9, self.flux_zero_point * self.exposure_zero_point_seconds)
        self.airmass_exponent = float(scoring_config.get("airmass_exponent", 0.6))
        self.lunar_model = scoring_config.get("lunar_model") or DEFAULT_LUNAR_MODEL

        program = scoring_config.get("program") or {}
        self.program_bands = {**DEFAULT_BANDS, **(program.get("bands") or {})}
        self.program_multipliers = {**DEFAULT_MULTIPLIERS, **(program.get("multipliers") or {})}
        self.mismatch_multiplier = float(program.get("mismatch_multiplier", DEFAULT_MISMATCH_MULTIPLIER))

        required = scoring_config.get("required") or {}
        self.required_penalty = float(required.get("penalty_per_missing", DEFAULT_REQUIRED_PENALTY))
        self.required_threshold = float(required.get("observed_factor_threshold", DEFAULT_REQUIRED_THRESHOLD))

        uniformity = scoring_config.get("uniformity") or {}
        self.uniformity_weight = float(uniformity.get("weight", 200.0))
        self.uniformity_band_width_deg = float(uniformity.get("ra_band_width_deg", 10.0))
        self.uniformity_threshold = float(uniformity.get("observed_factor_threshold", DEFAULT_REQUIRED_THRESHOLD))

        self.latitude_deg = float(site.get("latitude_deg", 0.0))
        self.longitude_deg = float(site.get("longitude_deg", 0.0))

    def quality_model(self, alt_deg: float, lunar_factor: float) -> float:
        """A per-second "quality model" for a target: lunar factor / (q0 * airmass^beta).
        The agent's own learned sky scale (SurveyState.scale) is applied separately by the
        caller -- this is the part of the estimate that depends only on public config and
        geometry. Used to size exposures, not to claim the real hidden q."""
        airmass = normalized_airmass(max(alt_deg, 1.0))
        return lunar_factor / (self.q0 * (airmass ** self.airmass_exponent))

    def completion_factor(self, flux: float, duration_seconds: float, quality: float) -> float:
        return max(0.0, min(1.0, flux * duration_seconds * quality / self.f0t0))

    def program_band(self, q_band: float) -> str:
        if q_band >= self.program_bands["DARK"]:
            return "DARK"
        if q_band >= self.program_bands["BRIGHT"]:
            return "BRIGHT"
        return "BACKUP"

    def program_multiplier(self, declared: str, actual_band: str) -> float:
        return self.program_multipliers.get(declared, 1.0) if declared == actual_band else self.mismatch_multiplier
