"""CPU hand tracking; only 2D boxes are used. No 3D reconstruction.

The task API exposes a handedness classification score, not the palm/presence
probabilities. Never represent that category score as a calibrated hand-presence
probability. Track IDs below are local geometric associations, not identities.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qc.config import HandConfig
from qc.visibility import Detections, HandBox


class MediaPipeHands:
    def __init__(self, config: HandConfig | None = None) -> None:
        self.config = config or HandConfig()
        model = Path(self.config.model_path)
        if not model.is_file():
            raise ValueError(f"Hand model missing: {model}. Run download-hand-model first.")
        self._model_bytes = model.read_bytes()
        version = importlib.metadata.version("mediapipe")
        identity = dict(
            version=version,
            model=hashlib.sha256(self._model_bytes).hexdigest(),
            settings=self.config.model_dump(exclude={"model_path"}),
        )
        self.cache_id = (
            "mediapipe-hands-v1:"
            + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        )
        self._landmarker: Any = None
        self._mp: Any = None
        self._last_ms = -1
        self._previous: list[HandBox] = []
        self._previous_time = -1.0
        self._next_id = 1

    def start_video(self) -> None:
        self.close()
        import mediapipe as mp

        self._mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_buffer=self._model_bytes, delegate=mp.tasks.BaseOptions.Delegate.CPU
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=self.config.num_hands,
            min_hand_detection_confidence=self.config.detection_threshold,
            min_hand_presence_confidence=self.config.presence_threshold,
            min_tracking_confidence=self.config.tracking_threshold,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_ms, self._previous_time, self._next_id = -1, -1.0, 1
        self._previous = []

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def detect(self, frame: np.ndarray, expected_object: str | None) -> Detections:
        return self.detect_at(frame, expected_object, (self._last_ms + 1) / 1000)

    def detect_at(
        self, frame: np.ndarray, expected_object: str | None, timestamp: float
    ) -> Detections:
        if self._landmarker is None:
            self.start_video()
        millis = max(self._last_ms + 1, round(timestamp * 1000))
        self._last_ms = millis
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)),
        )
        detected = self._landmarker.detect_for_video(image, millis)
        hands = self.convert(detected)
        self._associate(hands, timestamp)
        return Detections(hands=hands, objects=None)

    def convert(self, detected: Any) -> list[HandBox]:
        hands = []
        for index, landmarks in enumerate(detected.hand_landmarks):
            points = np.array([(p.x, p.y) for p in landmarks])
            if not len(points) or not np.all(np.isfinite(points)):
                continue
            low, high = points.min(axis=0), points.max(axis=0)
            padding = (high - low) * self.config.box_padding
            low, high = np.clip(low - padding, 0, 1), np.clip(high + padding, 0, 1)
            if np.any(high <= low):
                continue
            categories = detected.handedness[index] if index < len(detected.handedness) else []
            category = max(categories, key=lambda c: c.score) if categories else None
            label = category.category_name if category else None
            if label not in {"Left", "Right"}:
                label = None
            if label and self.config.swap_handedness:
                label = "Right" if label == "Left" else "Left"
            hands.append(
                HandBox(
                    x1=float(low[0]),
                    y1=float(low[1]),
                    x2=float(high[0]),
                    y2=float(high[1]),
                    confidence=float(category.score) if category else None,
                    confidence_kind="handedness_classification",
                    handedness=label,
                )
            )
        return hands

    def _associate(self, hands: list[HandBox], timestamp: float) -> None:
        candidates = []
        if 0 <= timestamp - self._previous_time <= self.config.max_track_gap_seconds:
            for i, current in enumerate(hands):
                for j, previous in enumerate(self._previous):
                    distance = np.hypot(
                        (current.x1 + current.x2 - previous.x1 - previous.x2) / 2,
                        (current.y1 + current.y2 - previous.y1 - previous.y2) / 2,
                    )
                    if distance <= self.config.max_track_distance:
                        candidates.append((float(distance), i, j))
        used_current, used_previous = set(), set()
        for _, i, j in sorted(candidates):
            if i not in used_current and j not in used_previous:
                hands[i].track_id = self._previous[j].track_id
                used_current.add(i)
                used_previous.add(j)
        for hand in hands:
            if hand.track_id is None:
                hand.track_id = self._next_id
                self._next_id += 1
        self._previous, self._previous_time = hands, timestamp


def make_detector() -> MediaPipeHands:
    return MediaPipeHands()
