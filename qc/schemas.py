from typing import Annotated, Any, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Ratio = Annotated[float, Field(ge=0, le=1)]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Metadata(Model):
    collector_id: str = Field(min_length=1)
    task_name: str = Field(min_length=1)
    expected_task_description: str = Field(min_length=1)
    expected_object: str | None = None
    min_duration: float | None = Field(default=None, ge=0)
    max_duration: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if (
            self.min_duration is not None
            and self.max_duration is not None
            and self.min_duration > self.max_duration
        ):
            raise ValueError("min_duration exceeds max_duration")
        return self


class Semantic(Model):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    task_correct: bool
    task_complete_score: float = Field(ge=0, le=1)
    hand_visibility_score: float = Field(ge=0, le=1)
    object_visibility_score: float = Field(ge=0, le=1)
    interaction_visibility_score: float = Field(ge=0, le=1)
    final_state_success: bool
    irrelevant_footage_score: float = Field(ge=0, le=1)
    protocol_violation: bool
    major_occlusion: bool | None = None  # Legacy records did not contain this judgment.
    reasoning_summary: str = Field(min_length=1, max_length=2000)


class SemanticJudgment(Semantic):
    """Wire schema for live providers: occlusion evidence is required."""

    major_occlusion: bool


class Technical(Model):
    available: bool = True
    corrupted: bool = False
    error: str | None = None
    duration: float = 0
    width: int = 0
    height: int = 0
    fps: float = 0
    decoded_frames: int = 0
    blur_score: Ratio = 0
    exposure_score: Ratio = 0
    camera_stability_score: Ratio = 0
    frozen_frame_ratio: Ratio = 0
    duplicate_frame_ratio: Ratio = 0
    low_motion_ratio: Ratio = 0
    longest_low_motion_seconds: float = 0
    dropped_frame_ratio: Ratio | None = None
    timestamp_gap_count: int = 0
    timing_available: bool = False
    low_motion_periods: list[dict[str, float]] = Field(default_factory=list)
    flags: dict[str, bool | None] = Field(default_factory=dict)
    frame_metrics: list[dict[str, Any]] = Field(default_factory=list)


class Visibility(Model):
    detector_id: str
    hand_visible_ratio: Ratio | None = None
    hand_boundary_ratio: Ratio | None = None
    object_visible_ratio: Ratio | None = None
    hand_object_visible_ratio: Ratio | None = None
    longest_hand_missing_interval: float | None = Field(default=None, ge=0)
    manipulation_offscreen_ratio: Ratio | None = None
    offscreen_evidence_source: str = "unavailable"
    hand_tracking_continuity_score: Ratio | None = None
    hand_missing_periods: list[dict[str, float]] = Field(default_factory=list)
    offscreen_periods: list[dict[str, float]] = Field(default_factory=list)
    frame_metrics: list[dict[str, Any]] = Field(default_factory=list)


class Result(Model):
    video_id: str
    collector_id: str
    task_name: str
    technical: Technical
    visibility: Visibility
    semantic: Semantic | None = None
    quality_score: float = Field(ge=0, le=100)
    component_scores: dict[str, Ratio | None]
    weighted_contributions: dict[str, float]
    metadata: Metadata
    evaluated_at: str
    semantic_provider_id: str | None = None
    config_snapshot: dict[str, Any]
    decision: Literal["PASS", "REVIEW", "FAIL"]
    payment_multiplier: float
    payment_status: Literal["recommended", "held_for_review", "rejected"]
    failure_reasons: list[str]
    human_review_required: bool
    audit_selected: bool = False
    evidence_notes: list[str] = Field(default_factory=list)
    evidence_confidence: dict[str, Any] = Field(default_factory=dict)
    source_video: str | None = None
    config_digest: str


class HumanReview(Model):
    video_id: str
    reviewer_id: str = Field(min_length=1)
    quality_score: float = Field(ge=0, le=100)
    decision: Literal["PASS", "REVIEW", "FAIL"]
    task_correct: bool | None = None
    notes: str = ""
    reviewed_at: AwareDatetime


Rating = Annotated[int, Field(ge=1, le=5)]


class CalibrationReview(Model):
    video_id: str = Field(min_length=1)
    evaluation_id: str = Field(min_length=1)
    reviewer_id: str = Field(min_length=1)
    timestamp: AwareDatetime
    task_correct: bool
    completeness: Rating
    hand_visibility: Rating
    object_visibility: Rating
    interaction_visibility: Rating
    technical_quality: Rating
    overall_usability: Rating
    wrong_task: bool
    task_incomplete: bool
    critical_interaction_missing: bool
    no_meaningful_manipulation: bool
    severe_protocol_violation: bool
    corrupted_video: bool
    human_decision: Literal["PASS", "REVIEW", "FAIL"]
    notes: str = ""

    @model_validator(mode="after")
    def coherent(self) -> Self:
        hard = any(
            getattr(self, field)
            for field in (
                "wrong_task",
                "task_incomplete",
                "critical_interaction_missing",
                "no_meaningful_manipulation",
                "severe_protocol_violation",
                "corrupted_video",
            )
        )
        if self.wrong_task == self.task_correct:
            raise ValueError("wrong_task must be the inverse of task_correct")
        if (hard or not self.task_correct) and self.human_decision == "PASS":
            raise ValueError("Human PASS conflicts with a hard failure")
        return self
