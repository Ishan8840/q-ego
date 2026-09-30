"""Self-contained visual reports. Annotations are tied to exact sampled frames."""

import base64
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from qc.config import Config
from qc.pipeline import atomic_writer
from qc.schemas import Result

STYLE = """body{font:15px system-ui;background:#101723;color:#e9eef6;margin:0;padding:28px;max-width:1600px;margin:auto}
h1{margin-bottom:8px}a{color:#90c5ff}p{line-height:1.6}.summary{display:flex;gap:18px;flex-wrap:wrap}.metric,figure{background:#1b2737;border:1px solid #35445a;border-radius:10px;padding:14px;margin:0}
.metric strong{font-size:25px;display:block}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-top:20px}img{width:100%;border-radius:5px}figcaption{margin-top:8px;line-height:1.5}.warning{color:#ffce80}.PASS{color:#82e4b1}.FAIL{color:#ff969c}.REVIEW{color:#ffce80}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:16px 0}video{width:100%;max-height:650px}button{padding:9px;margin:10px 0}small{color:#b5c5da}"""


def frame_warnings(frame: np.ndarray, row: dict[str, Any], cfg: Config) -> list[str]:
    gray = cv2.cvtColor(cv2.resize(frame, (320, 240)), cv2.COLOR_BGR2GRAY)
    warnings = []
    if row.get("hand") is False:
        warnings.append("HAND MISSING")
    if row.get("in_view") is False:
        warnings.append("OFFSCREEN / OCCLUSION CANDIDATE")
    if row.get("boundary"):
        warnings.append("HAND AT EDGE")
    if cv2.Laplacian(gray, cv2.CV_64F).var() < cfg.technical.blur_laplacian:
        warnings.append("BLUR")
    if np.mean(gray <= cfg.technical.dark_pixel) > cfg.technical.max_clipped_ratio:
        warnings.append("DARK")
    if np.mean(gray >= cfg.technical.bright_pixel) > cfg.technical.max_clipped_ratio:
        warnings.append("OVEREXPOSED")
    return warnings


def select_rows(result: Result, maximum: int) -> list[dict[str, Any]]:
    rows = result.visibility.frame_metrics or result.technical.frame_metrics
    if len(rows) <= maximum:
        return rows
    indices = np.unique(np.linspace(0, len(rows) - 1, maximum).astype(int))
    return [rows[int(i)] for i in indices]


def generate_report(
    video: Path,
    result: Result,
    destination: Path,
    *,
    blind: bool = False,
    video_link: str | None = None,
) -> None:
    cfg = Config.model_validate(result.config_snapshot)
    escaped = html.escape
    title = "Independent video review" if blind else "Egocentric manipulation QC"
    parts = [
        f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title}</title><style>{STYLE}</style><body>',
        f"<h1>{title}</h1><h2>{escaped(result.task_name)}</h2>",
        f"<p>{escaped(result.metadata.expected_task_description)}</p>",
        f"<p>Expected object: {escaped(result.metadata.expected_object or 'not specified')}</p>",
    ]
    if video_link:
        link = escaped(video_link, quote=True)
        parts.append(
            f'<video controls preload="metadata" src="{link}"></video><p><a href="{link}" download>Download original video</a> if the browser cannot play this codec.</p>'
        )
    if blind:
        parts.append(
            "<p>Watch the full video before rating. These still frames are a navigation aid and cannot establish continuity. Automated scores and warnings are intentionally withheld.</p>"
        )
    else:
        parts.append(
            f'<div class="summary"><div class="metric"><strong class="{result.decision}">{result.decision}</strong>QC decision</div><div class="metric"><strong>{result.quality_score:.2f}/100</strong>Quality score</div><div class="metric"><strong>{escaped(result.evidence_confidence.get("level", "uncalibrated"))}</strong>Judgment confidence — not a probability</div></div>'
        )
        parts.append(
            f"<p>Payment: {escaped(result.payment_status)} · Human review required: {result.human_review_required}</p>"
        )
        parts.append(
            "<p class=warning>Reasons: "
            + escaped(", ".join(result.failure_reasons) or "No rejection flags")
            + "</p>"
        )
        if result.semantic:
            parts.append(
                "<p>Semantic judgment: " + escaped(result.semantic.reasoning_summary) + "</p>"
            )
        cards = []
        for name, value in result.component_scores.items():
            display = f"{value * 100:.1f}" if value is not None else "Unavailable"
            cards.append(
                f'<div class="metric"><strong>{display}</strong>{escaped(name.replace("_", " "))}</div>'
            )
        parts.append('<div class="summary">' + "".join(cards) + "</div>")
        parts.append(
            "<p><small>Boxes: side confidence is handedness classification confidence, not hand-presence probability. Tracking IDs are local geometric associations.</small></p>"
        )
        metrics = dict(
            components=result.component_scores,
            weighted_points=result.weighted_contributions,
            visibility=result.visibility.model_dump(exclude={"frame_metrics"}),
            technical=result.technical.model_dump(exclude={"frame_metrics"}),
            semantic=result.semantic.model_dump() if result.semantic else None,
            confidence=result.evidence_confidence,
            notes=result.evidence_notes,
        )
        parts.append(
            "<details><summary>Individual metrics, evidence and limitations</summary><pre>"
            + escaped(json.dumps(metrics, indent=2))
            + "</pre></details>"
        )
    rows = select_rows(result, cfg.report_max_frames)
    parts.append(
        f"<p>Chronological samples shown: {len(rows)}. Sampling can miss brief interactions. Hand boxes indicate model estimates, not verified contact.</p>"
        if not blind
        else f"<p>Chronological samples shown: {len(rows)}.</p>"
    )
    parts.append('<div class="grid">')
    cap = cv2.VideoCapture(str(video))
    try:
        for row in rows:
            index = row.get("frame", round(row["timestamp"] * result.technical.fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                parts.append(
                    f"<figure><figcaption>{row['timestamp']:.3f}s — frame unavailable</figcaption></figure>"
                )
                continue
            warnings = [] if blind else frame_warnings(frame, row, cfg)
            if not blind:
                height, width = frame.shape[:2]
                for hand in row.get("hands") or []:
                    x1, y1, x2, y2 = [
                        int(hand[k] * size)
                        for k, size in [
                            ("x1", width),
                            ("y1", height),
                            ("x2", width),
                            ("y2", height),
                        ]
                    ]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 230, 120), 2)
                    label = f"{hand.get('handedness') or 'Hand'} #{hand.get('track_id') or '?'}"
                    if hand.get("confidence") is not None:
                        label += f" side={hand['confidence']:.2f}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (80, 230, 120),
                        1,
                        cv2.LINE_AA,
                    )
            scale = min(1, 640 / frame.shape[1])
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
            cv2.putText(
                frame,
                f"{row['timestamp']:.3f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"{row['timestamp']:.3f}s",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise ValueError("Report JPEG encoding failed")
            image = base64.b64encode(encoded).decode()
            caption = " · ".join(warnings)
            if not blind and row.get("hand") is None:
                caption += " · HAND DETECTOR UNAVAILABLE"
            parts.append(
                f'<figure><img alt="Frame at {row["timestamp"]:.3f} seconds" src="data:image/jpeg;base64,{image}"><figcaption><b>{row["timestamp"]:.3f}s</b> · frame {index}<br><span class="warning">{escaped(caption)}</span></figcaption></figure>'
            )
    finally:
        cap.release()
    if not rows:
        parts.append("<p>No decodable frame evidence available.</p>")
    parts.append("</div></body></html>")
    with atomic_writer(destination) as output:
        output.write("\n".join(parts))
