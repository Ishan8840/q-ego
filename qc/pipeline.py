import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TextIO, TypeVar

from pydantic import BaseModel

from qc import technical, visibility, vlm
from qc.audit import selected
from qc.config import Config
from qc.ingest import video_digest
from qc.payment import recommend
from qc.schemas import Metadata, Result, Semantic, Technical, Visibility
from qc.scoring import score

LOG = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
CACHE_VERSION = "ego-q-4"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def atomic_writer(path: Path) -> Iterator[TextIO]:
    """Replace completed files atomically, including on Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            yield stream
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    with atomic_writer(path) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def cached(directory: Path, stage: str, key: dict, model: type[T], compute: Callable[[], T]) -> T:
    path = directory / stage / (digest(dict(version=CACHE_VERSION, **key)) + ".json")
    if path.exists():
        try:
            result = model.model_validate_json(path.read_text())
            LOG.info("Cache hit: %s", stage)
            return result
        except (ValueError, OSError):
            LOG.warning("Ignoring invalid cache entry %s", path)
    result = compute()
    # Environment failures and corrupt-input diagnostics should be retried.
    if not (isinstance(result, Technical) and (result.corrupted or not result.available)):
        write_json(path, result.model_dump())
    return result


def evaluate(
    path: Path,
    metadata: Metadata,
    cfg: Config,
    cache_dir: Path,
    detector: visibility.Detector | None = None,
    provider: vlm.Provider | None = None,
    request_output: Path | None = None,
) -> Result:
    detector = detector or visibility.UnavailableDetector()
    identifier = video_digest(path)
    notes = []
    t = cached(
        cache_dir,
        "technical",
        dict(
            video=identifier,
            min_duration=metadata.min_duration,
            max_duration=metadata.max_duration,
            config=cfg.technical.model_dump(),
        ),
        Technical,
        lambda: technical.analyze(path, metadata, cfg.technical),
    )
    v = Visibility(detector_id=detector.cache_id)
    s = None
    if not t.corrupted and t.available:
        try:
            v = cached(
                cache_dir,
                "visibility",
                dict(
                    video=identifier,
                    detector=detector.cache_id,
                    object=metadata.expected_object,
                    sample_fps=cfg.visibility_sample_fps,
                    margin=cfg.boundary_margin,
                    offscreen=cfg.offscreen_min_seconds,
                ),
                Visibility,
                lambda: visibility.analyze(path, detector, metadata.expected_object, cfg),
            )
        except Exception as exc:
            LOG.exception("Visibility evaluation failed")
            notes.append(f"Detector error: {type(exc).__name__}: {exc}")
        if provider or request_output:

            def semantic_compute() -> Semantic:
                assert provider is not None
                return vlm.evaluate(provider, vlm.make_request(path, metadata, cfg.semantic_frames))

            try:
                if request_output:
                    write_json(
                        request_output, vlm.make_request(path, metadata, cfg.semantic_frames)
                    )
                if provider:
                    s = cached(
                        cache_dir,
                        "semantic",
                        dict(
                            video=identifier,
                            metadata=vlm.task_context(metadata),
                            provider=provider.cache_id,
                            prompt=vlm.PROMPT_VERSION,
                            frames=cfg.semantic_frames,
                        ),
                        Semantic,
                        semantic_compute,
                    )
            except Exception as exc:
                LOG.exception("Semantic evaluation failed")
                notes.append(f"Semantic error: {type(exc).__name__}: {exc}")
    scored = score(t, v, s, cfg)
    audited = scored["decision"] == "PASS" and selected(
        identifier, cfg.audit_percent, cfg.audit_seed
    )
    if audited:
        scored["human_review_required"] = True
        notes.append("PASS clip selected for random human audit.")
    multiplier, status = recommend(
        scored["quality_score"], scored["decision"], scored["human_review_required"], cfg
    )
    scored["evidence_notes"].extend(notes)
    scored["evidence_notes"].append(
        "Dropped-frame ratio estimates missing nominal timestamp intervals, not verified capture loss (VFR can produce gaps); null means timing unavailable. Motion/blur metrics are heuristic proxies."
    )
    return Result(
        source_video=str(path.resolve()),
        evidence_confidence=evidence_confidence(t, v, s, scored["evidence_notes"]),
        video_id=identifier,
        collector_id=metadata.collector_id,
        task_name=metadata.task_name,
        technical=t,
        visibility=v,
        semantic=s,
        **scored,
        audit_selected=audited,
        payment_multiplier=multiplier,
        payment_status=status,
        config_digest=digest(cfg.model_dump()),
        metadata=metadata,
        config_snapshot=cfg.model_dump(),
        evaluated_at=datetime.now(timezone.utc).isoformat(),
        semantic_provider_id=provider.cache_id if provider else None,
    )


def save_report(result: Result, path: Path) -> None:
    lines = [
        f"Video: {result.video_id}",
        f"Collector: {result.collector_id} | Task: {result.task_name}",
        f"Decision: {result.decision} | Quality: {result.quality_score:.2f}/100",
        f"Payment: {result.payment_multiplier:.2f} × base ({result.payment_status})",
        f"Human review: {result.human_review_required}",
        "Components (0–1):",
    ]
    lines += [
        f"  {key}: {value if value is not None else 'unavailable'}"
        for key, value in result.component_scores.items()
    ]
    lines += [
        f"  {key}: {value:.2f} weighted points"
        for key, value in result.weighted_contributions.items()
    ]
    if result.technical.error:
        lines.append("Technical error: " + result.technical.error)
    lines += ["Reasons: " + (", ".join(result.failure_reasons) or "none")]
    if result.semantic:
        lines.append("Semantic explanation: " + result.semantic.reasoning_summary)
    lines += result.evidence_notes
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_writer(path) as stream:
        stream.write("\n".join(lines) + "\n")
    if result.source_video:
        from qc.report import generate_report

        generate_report(Path(result.source_video), result, path.with_suffix(".html"))


def evidence_confidence(t: Technical, v: Visibility, s: Semantic | None, notes: list[str]) -> dict:
    sources = dict(
        technical=t.available,
        hand_detector=v.hand_visible_ratio is not None,
        object_detector=v.object_visible_ratio is not None,
        semantic=s is not None,
    )
    limitations = [
        "Judgment confidence is not calibrated against human labels.",
        "Sparse semantic frames cannot prove uninterrupted contact or completion.",
    ]
    if not all(sources.values()):
        limitations.append("Some independent evidence sources are unavailable.")
    if any("differ" in note for note in notes):
        limitations.append("Detector and semantic evidence disagree.")
    return dict(
        level="insufficient" if s is None or not sources["hand_detector"] else "limited",
        calibrated_probability=None,
        available_sources=sources,
        limitations=limitations,
    )
