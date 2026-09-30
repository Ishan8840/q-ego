import hashlib
from pathlib import Path

from qc.schemas import HumanReview, Result


def selected(video_id: str, percent: float, seed: str) -> bool:
    value = int(hashlib.sha256(f"{seed}:{video_id}".encode()).hexdigest()[:16], 16) / 2**64
    return value < percent / 100


def append_review(path: Path, review: HumanReview) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(review.model_dump_json() + "\n")


def compare(results: list[Result], reviews: list[HumanReview]) -> dict:
    # Latest timestamp per clip avoids counting repeat reviews as independent clips.
    latest = {r.video_id: r for r in sorted(reviews, key=lambda r: r.reviewed_at)}
    pairs = [(r, latest[r.video_id]) for r in results if r.video_id in latest]
    semantic_pairs = [
        (r, h) for r, h in pairs if r.semantic is not None and h.task_correct is not None
    ]
    return dict(
        reviewed_clips=len(pairs),
        decision_disagreement_rate=sum(r.decision != h.decision for r, h in pairs) / len(pairs)
        if pairs
        else None,
        score_mean_absolute_error=sum(abs(r.quality_score - h.quality_score) for r, h in pairs)
        / len(pairs)
        if pairs
        else None,
        vlm_human_disagreement_rate=sum(
            r.semantic.task_correct != h.task_correct for r, h in semantic_pairs
        )
        / len(semantic_pairs)
        if semantic_pairs
        else None,
        vlm_human_compared_clips=len(semantic_pairs),
    )
