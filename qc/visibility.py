"""Replaceable detectors; unknown capability is distinct from no detection."""

from pathlib import Path
from typing import Literal, Protocol, Self

import cv2
import numpy as np
from pydantic import Field, model_validator

from qc.config import Config
from qc.schemas import Model, Ratio, Visibility


class Box(Model):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("Invalid bounding box")
        return self


class HandBox(Box):
    confidence: Ratio | None = None
    confidence_kind: str = "unavailable"
    detection_confidence: Ratio | None = None
    handedness: Literal["Left", "Right"] | None = None
    track_id: int | None = Field(default=None, ge=1)


class Detections(Model):
    hands: list[HandBox | Box] | None = None
    objects: list[Box] | None = None


class Detector(Protocol):
    @property
    def cache_id(self) -> str: ...
    def detect(self, frame: np.ndarray, expected_object: str | None) -> Detections: ...


class UnavailableDetector:
    cache_id = "unavailable-v1"

    def detect(self, frame: np.ndarray, expected_object: str | None) -> Detections:
        return Detections()


def intervals(rows: list[dict], field: str, end: float) -> list[dict[str, float]]:
    """False runs using sample-and-hold intervals, never bridge unknown evidence."""
    periods = []
    start = None
    for row in rows + [dict(timestamp=end, **{field: None})]:
        absent = row[field] is False
        if absent and start is None:
            start = row["timestamp"]
        if not absent and start is not None:
            periods.append(dict(start=start, end=row["timestamp"]))
            start = None
    return periods


def analyze(path: Path, detector: Detector, expected_object: str | None, cfg: Config) -> Visibility:
    result = Visibility(detector_id=detector.cache_id)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        cap.release()
        raise ValueError("Cannot decode visibility samples")
    index, next_sample, timestamp = 0, 0.0, 0.0
    try:
        start_video = getattr(detector, "start_video", None)
        if start_video:
            start_video()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamp = raw_time if np.isfinite(raw_time) and raw_time > 0 else index / fps
            frame_index = index
            index += 1
            if timestamp + 1e-8 < next_sample:
                continue
            next_sample = timestamp + 1 / cfg.visibility_sample_fps
            detect_at = getattr(detector, "detect_at", None)
            detection = (
                detect_at(frame, expected_object, timestamp)
                if detect_at
                else detector.detect(frame, expected_object)
            )
            hands, objects = detection.hands, detection.objects
            details = None
            if hands is not None:
                details = []
                height, width = frame.shape[:2]
                for box in hands:
                    detail = HandBox.model_validate(box.model_dump()).model_dump()
                    detail["boundary_distance"] = min(box.x1, box.y1, 1 - box.x2, 1 - box.y2)
                    detail["boundary_distance_pixels"] = min(
                        box.x1 * width, box.y1 * height, (1 - box.x2) * width, (1 - box.y2) * height
                    )
                    details.append(detail)
            hand = None if hands is None else bool(hands)
            obj = None if objects is None else bool(objects)
            boundary = (
                None
                if details is None
                else any(d["boundary_distance"] <= cfg.boundary_margin for d in details)
            )
            simultaneous = None if hands is None or objects is None else bool(hands and objects)
            # With no object model this is explicitly only a hand-absence proxy.
            in_view = (
                False if hand is False or obj is False else (hand if obj is None else simultaneous)
            )
            result.frame_metrics.append(
                dict(
                    frame=frame_index,
                    timestamp=float(timestamp),
                    hand=hand,
                    object=obj,
                    boundary=boundary,
                    simultaneous=simultaneous,
                    in_view=in_view,
                    hands=details,
                    objects=None if objects is None else [box.model_dump() for box in objects],
                )
            )
    finally:
        cap.release()
        close = getattr(detector, "close", None)
        if close:
            close()
    rows = result.frame_metrics
    if not rows:
        raise ValueError("No visibility frames decoded")
    end = timestamp + 1 / fps
    for field, key in [
        ("hand_visible_ratio", "hand"),
        ("object_visible_ratio", "object"),
        ("hand_boundary_ratio", "boundary"),
        ("hand_object_visible_ratio", "simultaneous"),
    ]:
        values = [r[key] for r in rows]
        if all(v is not None for v in values):
            setattr(result, field, sum(values) / len(values))
    if result.hand_visible_ratio is not None:
        result.hand_missing_periods = intervals(rows, "hand", end)
        result.longest_hand_missing_interval = max(
            (p["end"] - p["start"] for p in result.hand_missing_periods), default=0
        )
        if all(all(h["track_id"] is not None for h in r["hands"]) for r in rows) and len(rows) > 1:
            ids = [{h["track_id"] for h in r["hands"]} for r in rows]
            result.hand_tracking_continuity_score = sum(
                bool(a & b) for a, b in zip(ids, ids[1:])
            ) / (len(rows) - 1)
    periods = intervals(rows, "in_view", end)
    result.offscreen_periods = [
        p for p in periods if p["end"] - p["start"] >= cfg.offscreen_min_seconds
    ]
    if all(r["in_view"] is not None for r in rows):
        result.manipulation_offscreen_ratio = sum(p["end"] - p["start"] for p in periods) / end
        result.offscreen_evidence_source = (
            "hand_and_object_absence"
            if result.object_visible_ratio is not None
            else "hand_absence_proxy"
        )
    return result
