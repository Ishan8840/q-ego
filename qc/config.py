from pathlib import Path
from typing import Self

import yaml
from pydantic import Field, model_validator

from qc.schemas import Model


class TechnicalConfig(Model):
    decode_timeout_seconds: float = Field(default=300, gt=0)
    timestamp_gap_factor: float = Field(default=1.8, gt=1)
    max_dropped_frame_ratio: float = Field(default=0.05, ge=0, le=1)
    phase_response_min: float = Field(default=0.1, ge=0, le=1)
    sample_fps: float = Field(default=2, gt=0, le=30)
    min_width: int = Field(default=640, gt=0)
    min_height: int = Field(default=480, gt=0)
    min_fps: float = Field(default=15, gt=0)
    min_duration: float = Field(default=2, ge=0)
    max_duration: float = Field(default=600, gt=0)
    blur_laplacian: float = Field(default=60, gt=0)
    dark_pixel: int = Field(default=15, ge=0, le=255)
    bright_pixel: int = Field(default=240, ge=0, le=255)
    max_clipped_ratio: float = Field(default=0.65, ge=0, le=1)
    duplicate_difference: float = Field(default=0.1, ge=0)
    low_motion_difference: float = Field(default=1, ge=0)
    max_frozen_ratio: float = Field(default=0.2, ge=0, le=1)
    max_low_motion_seconds: float = Field(default=10, gt=0)
    min_quality_ratio: float = Field(default=0.5, ge=0, le=1)
    max_shake_fraction: float = Field(default=0.08, gt=0)


class HandConfig(Model):
    model_path: str = "models/hand_landmarker.task"
    num_hands: int = Field(default=2, ge=1, le=8)
    detection_threshold: float = Field(default=0.5, ge=0, le=1)
    presence_threshold: float = Field(default=0.5, ge=0, le=1)
    tracking_threshold: float = Field(default=0.5, ge=0, le=1)
    box_padding: float = Field(default=0.05, ge=0, le=0.5)
    swap_handedness: bool = False
    max_track_gap_seconds: float = Field(default=1, gt=0)
    max_track_distance: float = Field(default=0.25, gt=0, le=1)


class VLMConfig(Model):
    model: str = "gpt-4.1-mini-2025-04-14"
    temperature: float | None = Field(default=0, ge=0, le=2)
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    max_output_tokens: int = Field(default=1500, ge=256, le=10000)
    image_detail: str = "high"

    @model_validator(mode="after")
    def detail_valid(self) -> Self:
        if self.image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be low, high or auto")
        return self


class Config(Model):
    hands: HandConfig = Field(default_factory=HandConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    visibility_sample_fps: float = Field(default=5, gt=0, le=30)
    report_max_frames: int = Field(default=60, ge=2, le=300)
    technical: TechnicalConfig = Field(default_factory=TechnicalConfig)
    weights: dict[str, float] = Field(
        default_factory=lambda: dict(
            task=30, hand_interaction=25, object=20, technical=15, framing=10
        )
    )
    pass_threshold: float = Field(default=85, ge=0, le=100)
    review_threshold: float = Field(default=65, ge=0, le=100)
    disagreement_threshold: float = Field(default=0.35, ge=0, le=1)
    no_manipulation_threshold: float = Field(default=0.05, ge=0, le=1)
    incomplete_threshold: float = Field(default=0.25, ge=0, le=1)
    critical_visibility_threshold: float = Field(default=0.15, ge=0, le=1)
    irrelevant_threshold: float = Field(default=0.95, ge=0, le=1)
    audit_percent: float = Field(default=5, ge=0, le=100)
    audit_seed: str = "ego-q-v1"
    borderline_margin: float = Field(default=2, ge=0, le=20)
    boundary_margin: float = Field(default=0.05, ge=0, le=0.5)
    offscreen_min_seconds: float = Field(default=2, gt=0)
    semantic_frames: int = Field(default=12, ge=2, le=100)
    bonus_threshold: float = Field(default=90, ge=0, le=100)
    full_pay_threshold: float = Field(default=75, ge=0, le=100)
    reduced_pay_threshold: float = Field(default=60, ge=0, le=100)
    bonus_multiplier: float = Field(default=1, ge=1)
    reduced_multiplier: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def valid(self) -> Self:
        if (
            set(self.weights) != {"task", "hand_interaction", "object", "technical", "framing"}
            or any(v < 0 for v in self.weights.values())
            or abs(sum(self.weights.values()) - 100) > 1e-6
        ):
            raise ValueError("weights must contain the five components and sum to 100")
        if (
            self.review_threshold > self.pass_threshold
            or not self.reduced_pay_threshold <= self.full_pay_threshold <= self.bonus_threshold
        ):
            raise ValueError("thresholds must be ordered")
        if self.technical.dark_pixel >= self.technical.bright_pixel:
            raise ValueError("dark_pixel must be below bright_pixel")
        if self.technical.min_duration > self.technical.max_duration:
            raise ValueError("duration limits must be ordered")
        return self


def load_config(path: Path | None = None) -> Config:
    return Config.model_validate(yaml.safe_load(path.read_text()) or {}) if path else Config()
