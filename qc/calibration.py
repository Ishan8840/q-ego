"""Blinded export, strict human ratings, and descriptive calibration only."""

import csv
import hashlib
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from qc.audit import selected
from qc.ingest import video_digest
from qc.pipeline import atomic_writer, write_json
from qc.report import generate_report
from qc.schemas import CalibrationReview, Result

DIMENSIONS = (
    "completeness",
    "hand_visibility",
    "object_visibility",
    "interaction_visibility",
    "technical_quality",
    "overall_usability",
)
DECISIONS = ("PASS", "REVIEW", "FAIL")
HARD_FIELDS = (
    "wrong_task",
    "task_incomplete",
    "critical_interaction_missing",
    "no_meaningful_manipulation",
    "severe_protocol_violation",
    "corrupted_video",
)


def evaluation_id(result: Result) -> str:
    # Stable across relocation/rerun, specific to the actual evidence and business rules.
    payload = result.model_dump(mode="json", exclude={"evaluated_at", "source_video"})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def export_review(
    results: list[Result], destination: Path, percent: float = 100, seed: str = "calibration-v1"
) -> dict:
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            "Review export requires an empty directory to protect existing annotations"
        )
    chosen = [
        r
        for r in {evaluation_id(r): r for r in results}.values()
        if selected(r.video_id, percent, seed)
    ]
    # Validate every source before producing a partial handoff.
    for result in chosen:
        if not result.source_video or not Path(result.source_video).is_file():
            raise ValueError(
                f"Original video unavailable for {result.video_id}; re-evaluate with this version"
            )
        if video_digest(Path(result.source_video)) != result.video_id:
            raise ValueError(f"Original video contents changed for {result.video_id}")
    reviewer = destination / "reviewer"
    analyst = destination / "analyst"
    reviewer.mkdir(parents=True, exist_ok=True)
    analyst.mkdir(parents=True, exist_ok=True)
    rows, links = [], []
    with atomic_writer(analyst / "automated_results.jsonl") as snapshot:
        for result in chosen:
            identifier = evaluation_id(result)
            source = Path(result.source_video)
            clip = reviewer / "videos" / (result.video_id + source.suffix)
            clip.parent.mkdir(parents=True, exist_ok=True)
            if not clip.exists():
                shutil.copyfile(source, clip)
            report = reviewer / (identifier + ".html")
            generate_report(source, result, report, blind=True, video_link="videos/" + clip.name)
            row = dict.fromkeys(CalibrationReview.model_fields, "")
            row.update(video_id=result.video_id, evaluation_id=identifier)
            rows.append(row)
            links.append(
                f'<li><a href="{identifier}.html">{html.escape(result.task_name)} · {identifier[:12]}</a></li>'
            )
            snapshot.write(result.model_dump_json() + "\n")
    with atomic_writer(reviewer / "human_reviews.csv") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(CalibrationReview.model_fields))
        writer.writeheader()
        writer.writerows(rows)
    with atomic_writer(reviewer / "index.html") as stream:
        stream.write(
            '<!doctype html><html lang="en"><meta charset="utf-8"><title>Independent review</title><h1>Independent video review</h1><p>Watch the full video, then fill human_reviews.csv using rating_guide.md. Do not consult automated judgments.</p><ul>'
            + "".join(links)
            + "</ul></html>"
        )
    (reviewer / "rating_guide.md").write_text(RATING_GUIDE)
    summary = dict(
        exported=len(chosen),
        seed=seed,
        percent=percent,
        reviewer_package="reviewer/",
        analyst_snapshot="analyst/automated_results.jsonl",
    )
    write_json(analyst / "manifest.json", summary)
    return summary


RATING_GUIDE = """# Independent manipulation-video rating
Watch the complete original video; stills are navigation aids only. Rate independently before viewing automated QC. Share only this reviewer directory with annotators.

Keep video_id and evaluation_id unchanged. Enter reviewer_id and timestamp in ISO 8601 with timezone (e.g. 2026-09-29T14:00:00Z). Multiple reviewers can fill separate copies; concatenate data rows for calibration.

For completeness, hand_visibility, object_visibility, interaction_visibility, technical_quality, and overall_usability use integers 1–5:
1 = unusable/absent; 2 = poor/major gaps; 3 = usable only with reservations; 4 = good/minor issues; 5 = clear and complete.

Completeness: all required task stages and final state. Hand visibility: important hand actions remain visible. Object visibility: active object and its changes remain visible. Interaction visibility: hand-object contact is observable, not merely co-occurrence. Technical quality: usable focus, exposure, motion and framing. Overall usability: value as a complete manipulation-training demonstration, considering all dimensions.

Use true/false for task_correct and every hard-failure field: wrong_task, task_incomplete, critical_interaction_missing, no_meaningful_manipulation, severe_protocol_violation, corrupted_video. wrong_task must be the inverse of task_correct. These flags describe clear critical failures, not minor imperfections. If unsure, choose REVIEW and explain in notes.

human_decision: PASS = usable demonstration; REVIEW = uncertain/borderline; FAIL = unusable or clear hard failure. PASS cannot coexist with a hard-failure flag. Leave all rating fields blank for clips not yet reviewed. Do not leave individual fields blank in a completed review.
"""


def load_annotations(path: Path) -> tuple[list[CalibrationReview], int]:
    reviews, incomplete = [], 0
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = set(CalibrationReview.model_fields) - {"notes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Annotation CSV requires columns: {sorted(required)}")
        for index, row in enumerate(reader, 2):
            if not any(
                (row.get(k) or "").strip() for k in required - {"video_id", "evaluation_id"}
            ):
                incomplete += 1
                continue
            try:
                data = {k: (row.get(k) or "").strip() for k in CalibrationReview.model_fields}
                for key in (*HARD_FIELDS, "task_correct"):
                    if data[key].lower() not in {"true", "false", "0", "1"}:
                        raise ValueError(f"{key} requires true/false")
                    data[key] = data[key].lower() in {"true", "1"}
                for key in DIMENSIONS:
                    data[key] = int(data[key])
                reviews.append(CalibrationReview.model_validate(data))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid annotation at row {index}: {exc}") from exc
    return reviews, incomplete


def correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: list[float]) -> list[float]:
    array = np.asarray(values)
    return [float(np.sum(array < v) + (np.sum(array == v) + 1) / 2) for v in array]


def rate(numerator: int, denominator: int) -> dict:
    return dict(
        numerator=numerator,
        denominator=denominator,
        rate=numerator / denominator if denominator else None,
    )


def automated_dimensions(result: Result) -> dict[str, float | None]:
    s, v = result.semantic, result.visibility
    return dict(
        task_correct=float(s.task_correct) if s else None,
        completeness=s.task_complete_score if s else None,
        hand_visibility=min(s.hand_visibility_score, v.hand_visible_ratio)
        if s and v.hand_visible_ratio is not None
        else v.hand_visible_ratio
        if v.hand_visible_ratio is not None
        else s.hand_visibility_score
        if s
        else None,
        object_visibility=min(s.object_visibility_score, v.object_visible_ratio)
        if s and v.object_visible_ratio is not None
        else s.object_visibility_score
        if s
        else v.object_visible_ratio,
        interaction_visibility=s.interaction_visibility_score if s else None,
        technical_quality=result.component_scores["technical"],
        overall_usability=result.quality_score / 100,
    )


def analyze(results: list[Result], annotations: list[CalibrationReview]) -> tuple[dict, list[dict]]:
    by_id = {evaluation_id(r): r for r in results}
    # Repeated reviews by the same reviewer replace older ones, not duplicate votes.
    latest = {
        (a.evaluation_id, a.reviewer_id): a for a in sorted(annotations, key=lambda a: a.timestamp)
    }
    grouped = defaultdict(list)
    paired = []
    for annotation in latest.values():
        result = by_id.get(annotation.evaluation_id)
        if result is None or result.video_id != annotation.video_id:
            raise ValueError(
                f"Annotation references unknown/stale evaluation: {annotation.evaluation_id}"
            )
        grouped[annotation.evaluation_id].append(annotation)
        paired.append(
            dict(
                video_id=result.video_id,
                evaluation_id=annotation.evaluation_id,
                automated_decision=result.decision,
                automated_metrics=result.model_dump(mode="json"),
                human_decision=annotation.human_decision,
                human_metrics=annotation.model_dump(mode="json"),
                reviewer_id=annotation.reviewer_id,
                timestamp=annotation.timestamp.isoformat(),
            )
        )
    samples = []
    for identifier, reviews in grouped.items():
        votes = Counter(r.human_decision for r in reviews)
        winner, count = votes.most_common(1)[0]
        decision = winner if count > len(reviews) / 2 else "REVIEW"
        human = {
            key: float(np.mean([(getattr(r, key) - 1) / 4 for r in reviews])) for key in DIMENSIONS
        }
        correct = sum(r.task_correct for r in reviews)
        human["task_correct"] = (
            None if correct == len(reviews) / 2 else float(correct > len(reviews) / 2)
        )
        for field in HARD_FIELDS:
            count = sum(getattr(review, field) for review in reviews)
            human[field] = None if count == len(reviews) / 2 else float(count > len(reviews) / 2)
        samples.append((by_id[identifier], decision, human))
    auto_scores = [r.quality_score for r, _, _ in samples]
    human_scores = [h["overall_usability"] * 100 for _, _, h in samples]
    matrix = {human: {auto: 0 for auto in DECISIONS} for human in DECISIONS}
    for r, decision, _ in samples:
        matrix[decision][r.decision] += 1
    dimensions = {}
    for key in ("task_correct", *DIMENSIONS):
        pairs = [(automated_dimensions(r)[key], h[key]) for r, _, h in samples]
        pairs = [(a, h) for a, h in pairs if a is not None and h is not None]
        dimensions[key] = dict(
            count=len(pairs),
            mae=float(np.mean([abs(a - h) for a, h in pairs])) if pairs else None,
            mean_bias=float(np.mean([a - h for a, h in pairs])) if pairs else None,
            correlation=correlation([a for a, _ in pairs], [h for _, h in pairs]),
            disagreement_rate=sum(abs(a - h) >= 0.25 for a, h in pairs) / len(pairs)
            if pairs
            else None,
        )
    hard_disagreement = {}
    reason_map = dict(
        wrong_task="wrong_task",
        task_incomplete="task_clearly_incomplete",
        critical_interaction_missing="critical_interaction_not_visible",
        no_meaningful_manipulation="no_meaningful_manipulation",
        severe_protocol_violation="severe_protocol_violation",
        corrupted_video="corrupted_video",
    )
    for field, reason in reason_map.items():
        pairs = [
            (reason in r.failure_reasons, h[field])
            for r, _, h in samples
            if h[field] is not None
            and (r.technical.available if field == "corrupted_video" else r.semantic is not None)
        ]
        hard_disagreement[field] = rate(sum(a != h for a, h in pairs), len(pairs))
    complete = [
        (r, h)
        for r, _, h in samples
        if r.technical.available
        and r.semantic is not None
        and r.visibility.hand_visible_ratio is not None
        and r.visibility.object_visible_ratio is not None
    ]
    coverage = dict(
        complete_evidence_evaluations=len(complete),
        partial_evidence_evaluations=len(samples) - len(complete),
        complete_evidence_mae=float(
            np.mean([abs(r.quality_score - h["overall_usability"] * 100) for r, h in complete])
        )
        if complete
        else None,
    )
    features = defaultdict(list)
    for r, decision, _ in samples:
        for section in ("technical", "visibility", "semantic"):
            model = getattr(r, section)
            if model is not None:
                for name, value in model.model_dump().items():
                    if isinstance(value, (int, float)):
                        features[f"{section}.{name}"].append(
                            (float(value), float(decision == "PASS"))
                        )
        for name, value in r.component_scores.items():
            if value is not None:
                features[f"component.{name}"].append((value, float(decision == "PASS")))
    associations = [
        dict(
            metric=name,
            count=len(values),
            correlation_with_acceptance=correlation([x for x, _ in values], [y for _, y in values]),
        )
        for name, values in features.items()
    ]
    associations.sort(
        key=lambda v: (
            abs(v["correlation_with_acceptance"])
            if v["correlation_with_acceptance"] is not None
            else -1
        ),
        reverse=True,
    )
    summary = dict(
        evaluations=len(samples),
        annotations=len(latest),
        unreviewed_evaluations=len(by_id) - len(samples),
        unit="one consensus per evaluation; latest vote per reviewer; decision ties become REVIEW",
        overall=dict(
            pearson=correlation(auto_scores, human_scores),
            spearman=correlation(ranks(auto_scores), ranks(human_scores)),
            mae=float(np.mean(np.abs(np.array(auto_scores) - human_scores))) if samples else None,
            scale="0-100; human (rating-1)*25",
        ),
        confusion_matrix=dict(rows="human", columns="automated", counts=matrix),
        false_acceptance=rate(matrix["FAIL"]["PASS"], sum(matrix["FAIL"].values())),
        false_rejection=rate(matrix["PASS"]["FAIL"], sum(matrix["PASS"].values())),
        unsafe_pass_fraction=rate(
            matrix["FAIL"]["PASS"], sum(matrix[d]["PASS"] for d in DECISIONS)
        ),
        review_rate=rate(sum(r.decision == "REVIEW" for r, _, _ in samples), len(samples)),
        evidence_coverage=coverage,
        hard_failure_disagreement=hard_disagreement,
        individual_dimensions=dimensions,
        acceptance_associations=associations,
        limitations=[
            "Descriptive associations are not causal and can be confounded by task or collector.",
            "REVIEW is abstention: false acceptance counts auto PASS/human FAIL; false rejection counts auto FAIL/human PASS.",
            "Dimension disagreement means at least one rating point (0.25 on the normalized scale).",
            "Confidence is not calibrated; small/selected review sets may not generalize. No thresholds or payments were changed.",
        ],
    )
    return summary, paired


def save_analysis(results: list[Result], annotation_path: Path, output: Path) -> dict:
    annotations, unfilled = load_annotations(annotation_path)
    summary, paired = analyze(results, annotations)
    summary["unfilled_annotation_rows"] = unfilled
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "calibration.json", summary)
    with atomic_writer(output / "paired_reviews.jsonl") as stream:
        for row in paired:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    with atomic_writer(output / "calibration.txt") as stream:
        stream.write(
            "Calibration analysis (no threshold optimization)\n"
            + json.dumps(summary, indent=2)
            + "\n"
        )
    return summary
