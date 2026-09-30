"""Full decode plus sampled spatial metrics, all deterministic.

Motion and timestamp gaps are proxies: VFR, static scenes, texture, intentional
camera motion and encoder behavior affect them. Never infer task semantics here.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np

from qc.config import TechnicalConfig
from qc.ingest import InfrastructureError, probe, verify_decode
from qc.schemas import Metadata, Technical


def estimate_dropped_frames(
    timestamps: list[float], fps: float, gap_factor: float
) -> tuple[float | None, int]:
    """Estimate missing nominal intervals; VFR gaps are not proof of capture loss."""
    deltas = np.diff(timestamps)
    if not len(deltas) or not np.all(np.isfinite(deltas)) or not np.all(deltas > 0):
        return None, 0
    nominal = 1 / fps
    gaps = deltas[deltas > gap_factor * nominal]
    missing = sum(max(1, round(float(delta / nominal)) - 1) for delta in gaps)
    return missing / (len(timestamps) + missing), len(gaps)


def analyze(path: Path, metadata: Metadata, cfg: TechnicalConfig) -> Technical:
    result = Technical()
    cap = None
    try:
        info = probe(path, cfg.decode_timeout_seconds)
        verify_decode(path, cfg.decode_timeout_seconds)
        stream = info["streams"][0]
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError("OpenCV cannot open video")
        result.fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(result.fps) or result.fps <= 0:
            raise ValueError("Invalid frame rate")
        result.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        result.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_duration = stream.get("duration") or info.get("format", {}).get("duration")
        result.duration = float(raw_duration) if raw_duration not in (None, "N/A") else 0
        if not np.isfinite(result.duration) or result.duration < 0:
            raise ValueError("Invalid duration")
        count = stream.get("nb_frames", "0")
        expected = int(count) if str(count).isdigit() else 0
        previous = None
        duplicates = low = transitions = 0
        next_sample = 0.0
        low_start = None
        timestamps: list[float] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            index = result.decoded_frames
            result.decoded_frames += 1
            raw_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
            timestamps.append(float(raw_time))
            timestamp = raw_time if np.isfinite(raw_time) and raw_time > 0 else index / result.fps
            gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
            difference = 0.0
            shake, response = None, None
            if previous is not None:
                transitions += 1
                difference = float(np.abs(gray.astype(float) - previous.astype(float)).mean())
                duplicates += difference <= cfg.duplicate_difference
                is_low = difference <= cfg.low_motion_difference
                low += is_low
                if is_low and low_start is None:
                    low_start = max(0.0, timestamp - 1 / result.fps)
                if not is_low and low_start is not None:
                    result.low_motion_periods.append(dict(start=low_start, end=timestamp))
                    low_start = None
                if timestamp >= next_sample:
                    shift, response = cv2.phaseCorrelate(
                        previous.astype(np.float32), gray.astype(np.float32)
                    )
                    if response > cfg.phase_response_min and np.all(np.isfinite(shift)):
                        shake = float(np.hypot(*shift) / 400)
            if timestamp >= next_sample:
                lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                dark = float(np.mean(gray <= cfg.dark_pixel))
                bright = float(np.mean(gray >= cfg.bright_pixel))
                result.frame_metrics.append(
                    dict(
                        frame=index,
                        timestamp=float(timestamp),
                        laplacian_variance=lap,
                        dark_ratio=dark,
                        bright_ratio=bright,
                        difference=difference,
                        camera_translation_fraction=shake,
                        phase_response=float(response)
                        if response is not None and np.isfinite(response)
                        else None,
                    )
                )
                next_sample = timestamp + 1 / cfg.sample_fps
            previous = gray
        if result.decoded_frames == 0:
            raise ValueError("No decodable frames")
        if expected and result.decoded_frames < expected:
            raise ValueError(f"Truncated decode: {result.decoded_frames}/{expected} frames")
        if not result.duration:
            result.duration = result.decoded_frames / result.fps
        if low_start is not None:
            result.low_motion_periods.append(dict(start=low_start, end=result.duration))
        result.longest_low_motion_seconds = max(
            (p["end"] - p["start"] for p in result.low_motion_periods), default=0
        )
        result.dropped_frame_ratio, result.timestamp_gap_count = estimate_dropped_frames(
            timestamps, result.fps, cfg.timestamp_gap_factor
        )
        result.timing_available = result.dropped_frame_ratio is not None
        rows = result.frame_metrics
        result.blur_score = float(
            np.mean([r["laplacian_variance"] >= cfg.blur_laplacian for r in rows])
        )
        result.exposure_score = float(
            np.mean(
                [max(r["dark_ratio"], r["bright_ratio"]) <= cfg.max_clipped_ratio for r in rows]
            )
        )
        shake_values = [
            r["camera_translation_fraction"]
            for r in rows
            if r["camera_translation_fraction"] is not None
        ]
        result.camera_stability_score = (
            float(np.mean([v <= cfg.max_shake_fraction for v in shake_values]))
            if shake_values
            else 0
        )
        result.duplicate_frame_ratio = duplicates / max(1, transitions)
        result.frozen_frame_ratio = result.duplicate_frame_ratio
        result.low_motion_ratio = low / max(1, transitions)
        minimum = metadata.min_duration if metadata.min_duration is not None else cfg.min_duration
        maximum = metadata.max_duration if metadata.max_duration is not None else cfg.max_duration
        if minimum > maximum:
            raise InfrastructureError("Metadata and config produce inverted duration limits")
        result.flags = dict(
            readable=True,
            duration=minimum <= result.duration <= maximum,
            resolution=result.width >= cfg.min_width and result.height >= cfg.min_height,
            fps=result.fps >= cfg.min_fps,
            frozen_frames=result.frozen_frame_ratio <= cfg.max_frozen_ratio,
            dropped_frames=result.dropped_frame_ratio <= cfg.max_dropped_frame_ratio
            if result.timing_available
            else None,
            blur=result.blur_score >= cfg.min_quality_ratio,
            exposure=result.exposure_score >= cfg.min_quality_ratio,
            camera_stability=result.camera_stability_score >= cfg.min_quality_ratio
            if shake_values
            else None,
            low_motion=result.longest_low_motion_seconds <= cfg.max_low_motion_seconds,
        )
    except InfrastructureError as exc:
        result.available = False
        result.error = str(exc)
    except (ValueError, OSError, cv2.error, subprocess.CalledProcessError) as exc:
        result.corrupted = True
        result.error = str(exc)
        if isinstance(exc, subprocess.CalledProcessError):
            result.error += ": " + (exc.stderr or "")[-2000:]
        result.flags["readable"] = False
    finally:
        if cap is not None:
            cap.release()
    return result
