import json

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from cli import main
from qc.analytics import summarize
from qc.audit import selected
from qc.config import Config
from qc.payment import recommend
from qc.pipeline import evaluate
from qc.schemas import HumanReview, Metadata, Semantic, Technical, Visibility
from qc.scoring import score
from qc.visibility import Box, Detections


@pytest.fixture
def metadata():
    return Metadata(
        collector_id="c1", task_name="move cup", expected_task_description="Move the cup"
    )


@pytest.fixture
def semantic():
    return Semantic(
        task_correct=True,
        task_complete_score=1.0,
        hand_visibility_score=1.0,
        object_visibility_score=1.0,
        interaction_visibility_score=1.0,
        final_state_success=True,
        irrelevant_footage_score=0.0,
        protocol_violation=False,
        reasoning_summary="Complete visible movement",
    )


@pytest.fixture
def technical():
    return Technical(
        blur_score=1,
        exposure_score=1,
        camera_stability_score=1,
        flags=dict(resolution=True, fps=True, duration=True),
    )


@pytest.fixture
def visible():
    return Visibility(
        detector_id="test",
        hand_visible_ratio=1,
        object_visible_ratio=1,
        hand_object_visible_ratio=1,
        hand_boundary_ratio=0,
    )


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "video.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
    assert writer.isOpened()
    rng = np.random.default_rng(1)
    background = rng.integers(30, 220, (480, 640, 3), dtype=np.uint8)
    for i in range(50):
        frame = np.roll(background, i, axis=1)
        writer.write(frame)
    writer.release()
    return path


class Detector:
    cache_id = "test-v1"

    def detect(self, frame, expected_object):
        return Detections(
            hands=[Box(x1=0.2, y1=0.2, x2=0.5, y2=0.5)],
            objects=[Box(x1=0.3, y1=0.3, x2=0.6, y2=0.6)],
        )


class Provider:
    cache_id = "test-v1"

    def __init__(self, semantic):
        self.semantic, self.calls = semantic, 0

    def evaluate(self, request):
        self.calls += 1
        assert len(request["frames"]) >= 2
        return self.semantic.model_dump_json()


def test_perfect_score(technical, visible, semantic):
    result = score(technical, visible, semantic, Config())
    assert result["quality_score"] == 100
    assert result["decision"] == "PASS"


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("task_correct", False, "wrong_task"),
        ("task_complete_score", 0.1, "task_clearly_incomplete"),
        ("interaction_visibility_score", 0.0, "critical_interaction_not_visible"),
        ("protocol_violation", True, "severe_protocol_violation"),
        ("irrelevant_footage_score", 1.0, "no_meaningful_manipulation"),
    ],
)
def test_hard_reject(technical, visible, semantic, field, value, reason):
    changed = semantic.model_copy(update={field: value})
    result = score(technical, visible, changed, Config())
    assert result["decision"] == "FAIL"
    assert reason in result["failure_reasons"]
    assert result["human_review_required"]


def test_unknown_review(technical):
    result = score(technical, Visibility(detector_id="none"), None, Config())
    assert result["decision"] == "REVIEW"
    assert result["component_scores"]["task"] is None


def test_conflict_review(technical, visible, semantic):
    visible.hand_visible_ratio = 0.1
    assert score(technical, visible, semantic, Config())["decision"] == "REVIEW"


@pytest.mark.parametrize("value", ["true", 1, None])
def test_strict_semantic(semantic, value):
    data = semantic.model_dump()
    data["task_correct"] = value
    with pytest.raises(ValidationError):
        Semantic.model_validate_json(json.dumps(data))


def test_config_validation():
    with pytest.raises(ValidationError):
        Config(weights={"task": 100})
    with pytest.raises(ValidationError):
        Config(pass_threshold=50, review_threshold=70)


@pytest.mark.parametrize("score_,expected", [(95, 1.1), (80, 1), (70, 0.5), (59, 0)])
def test_payment(score_, expected):
    assert recommend(score_, "PASS", False, Config(bonus_multiplier=1.1))[0] == expected
    assert recommend(score_, "PASS", True, Config())[1] == "held_for_review"
    assert recommend(score_, "FAIL", False, Config())[0] == 0


def test_pipeline_cache(video, metadata, semantic, tmp_path):
    provider = Provider(semantic)
    cfg = Config(audit_percent=0)
    first = evaluate(video, metadata, cfg, tmp_path / "cache", Detector(), provider)
    second = evaluate(video, metadata, cfg, tmp_path / "cache", Detector(), provider)
    assert first.model_dump(exclude={"evaluated_at"}) == second.model_dump(exclude={"evaluated_at"})
    assert provider.calls == 1
    assert first.decision == "PASS"
    assert first.technical.decoded_frames == 50
    assert first.visibility.hand_visible_ratio == 1
    assert first.technical.frame_metrics
    # Business rules change without paying for another semantic call.
    evaluate(video, metadata, Config(audit_percent=100), tmp_path / "cache", Detector(), provider)
    assert provider.calls == 1
    metadata.expected_task_description = "A different task"
    evaluate(video, metadata, cfg, tmp_path / "cache", Detector(), provider)
    assert provider.calls == 2


def test_corrupt(tmp_path, metadata):
    path = tmp_path / "bad.mp4"
    path.write_bytes(b"not video")
    result = evaluate(path, metadata, Config(), tmp_path / "cache")
    assert result.decision == "FAIL"
    assert result.technical.corrupted
    assert result.payment_multiplier == 0


@pytest.mark.parametrize("brightness", [0, 255])
def test_frozen_exposure(tmp_path, metadata, brightness):
    path = tmp_path / "frozen.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
    for _ in range(50):
        writer.write(np.full((480, 640, 3), brightness, dtype=np.uint8))
    writer.release()
    result = evaluate(path, metadata, Config(), tmp_path / "cache")
    assert result.technical.frozen_frame_ratio == 1
    assert not result.technical.flags["blur"]
    assert not result.technical.flags["exposure"]


def test_provider_failure(video, metadata, tmp_path):
    class Broken:
        cache_id = "broken"

        def evaluate(self, request):
            return '{"task_correct": "yes"}'

    result = evaluate(video, metadata, Config(), tmp_path / "cache", provider=Broken())
    assert result.decision == "REVIEW"
    assert result.semantic is None
    assert any("Semantic error" in note for note in result.evidence_notes)


def test_audit_and_analytics(video, metadata, semantic, tmp_path):
    assert selected("x", 100, "seed")
    assert not selected("x", 0, "seed")
    result = evaluate(
        video,
        metadata,
        Config(audit_percent=100),
        tmp_path / "cache",
        Detector(),
        Provider(semantic),
    )
    assert result.audit_selected
    assert result.decision == "PASS"
    assert result.payment_status == "held_for_review"
    human = HumanReview(
        video_id=result.video_id,
        reviewer_id="r",
        quality_score=60,
        decision="FAIL",
        task_correct=False,
        reviewed_at="2026-01-01T00:00:00Z",
    )
    report = summarize([result], [human])
    assert report["human_comparison"]["vlm_human_disagreement_rate"] == 1
    assert report["by_collector"]["c1"]["count"] == 1
    assert summarize([], [])["human_comparison"]["decision_disagreement_rate"] is None


def test_cli_folder(video, tmp_path):
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "video,collector_id,task_name,expected_task_description\nvideo.avi,c1,move,move cup\nmissing.mp4,c2,move,move cup\n"
    )
    output = tmp_path / "results.jsonl"
    assert (
        main(
            [
                "evaluate-folder",
                "--input",
                str(tmp_path),
                "--metadata",
                str(metadata),
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 1
    )
    assert len(output.read_text().splitlines()) == 1
    assert json.loads(output.with_suffix(".errors.json").read_text())["errors"][0]["row"] == 3


@pytest.mark.parametrize(
    "quality,expected", [(64, "FAIL"), (65, "REVIEW"), (84, "REVIEW"), (85, "PASS"), (100, "PASS")]
)
def test_score_boundaries(technical, visible, semantic, quality, expected):
    cfg = Config(
        weights=dict(task=0, hand_interaction=100, object=0, technical=0, framing=0),
        borderline_margin=0,
    )
    semantic.hand_visibility_score = quality / 100
    semantic.interaction_visibility_score = quality / 100
    visible.hand_visible_ratio = quality / 100
    result = score(technical, visible, semantic, cfg)
    assert result["quality_score"] == quality
    assert result["decision"] == expected
    assert sum(result["weighted_contributions"].values()) == quality


def test_borderline_override(technical, visible, semantic):
    cfg = Config(weights=dict(task=0, hand_interaction=100, object=0, technical=0, framing=0))
    semantic.hand_visibility_score = semantic.interaction_visibility_score = 0.86
    visible.hand_visible_ratio = 0.86
    result = score(technical, visible, semantic, cfg)
    assert result["decision"] == "REVIEW"
    assert any("boundary" in note for note in result["evidence_notes"])


def test_unconfirmed_final_state(technical, visible, semantic):
    semantic.final_state_success = False
    result = score(technical, visible, semantic, Config())
    assert result["quality_score"] == 90
    assert result["decision"] == "REVIEW"
    assert "final_state_not_confirmed" in result["failure_reasons"]


def test_infrastructure_not_corruption(video, metadata, tmp_path, monkeypatch):
    from qc.ingest import InfrastructureError

    def unavailable(*args):
        raise InfrastructureError("Decoder unavailable")

    monkeypatch.setattr("qc.technical.verify_decode", unavailable)
    result = evaluate(video, metadata, Config(), tmp_path / "cache")
    assert not result.technical.corrupted
    assert not result.technical.available
    assert result.decision == "REVIEW"
    assert result.payment_status == "held_for_review"
    assert not list((tmp_path / "cache").rglob("*.json"))


def test_duration_resolution_and_fps(video, metadata, tmp_path):
    from qc.config import TechnicalConfig

    metadata.max_duration = 1
    result = evaluate(
        video,
        metadata,
        Config(technical=TechnicalConfig(min_width=1920, min_fps=30, min_duration=0)),
        tmp_path / "cache",
    )
    assert not result.technical.flags["duration"]
    assert not result.technical.flags["resolution"]
    assert not result.technical.flags["fps"]
    assert result.technical.timing_available
    assert result.technical.dropped_frame_ratio == 0


def test_boundary_and_offscreen(video, metadata, semantic, tmp_path):
    class MissingHands:
        cache_id = "missing-hands"

        def detect(self, frame, expected_object):
            return Detections(hands=[], objects=[Box(x1=0.2, y1=0.2, x2=0.5, y2=0.5)])

    result = evaluate(
        video, metadata, Config(), tmp_path / "cache", MissingHands(), Provider(semantic)
    )
    assert result.visibility.hand_visible_ratio == 0
    assert result.visibility.object_visible_ratio == 1
    assert result.visibility.hand_object_visible_ratio == 0
    assert result.visibility.offscreen_periods[0]["start"] == 0
    assert result.visibility.offscreen_periods[0]["end"] == pytest.approx(2.5)
    assert result.decision == "REVIEW"
    assert "possible_manipulation_offscreen" in result.failure_reasons

    class EdgeHands:
        cache_id = "edge-hands"

        def detect(self, frame, expected_object):
            return Detections(hands=[Box(x1=0, y1=0, x2=0.5, y2=0.5)], objects=[])

    edge_result = evaluate(video, metadata, Config(), tmp_path / "cache", EdgeHands())
    assert edge_result.visibility.hand_boundary_ratio == 1
    assert edge_result.visibility.hand_visible_ratio == 1
    assert edge_result.visibility.object_visible_ratio == 0


def test_corrupt_semantic_cache_recomputed(video, metadata, semantic, tmp_path):
    provider = Provider(semantic)
    cache = tmp_path / "cache"
    evaluate(video, metadata, Config(), cache, Detector(), provider)
    next((cache / "semantic").glob("*.json")).write_text("invalid JSON")
    evaluate(video, metadata, Config(), cache, Detector(), provider)
    assert provider.calls == 2


def test_single_cli_and_audit_export(video, tmp_path):
    output = tmp_path / "single.json"
    assert (
        main(
            [
                "evaluate",
                "--video",
                str(video),
                "--task",
                "move cup",
                "--output",
                str(output),
                "--cache",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    assert output.with_suffix(".report.txt").exists()
    result = json.loads(output.read_text())
    reviews = tmp_path / "reviews.jsonl"
    assert (
        main(
            [
                "audit-add",
                "--video-id",
                result["video_id"],
                "--reviewer-id",
                "human",
                "--score",
                "80",
                "--decision",
                "PASS",
                "--output",
                str(reviews),
            ]
        )
        == 0
    )
    analytics = tmp_path / "analytics.json"
    assert (
        main(
            [
                "analytics",
                "--results",
                str(output),
                "--reviews",
                str(reviews),
                "--output",
                str(analytics),
            ]
        )
        == 0
    )
    assert json.loads(analytics.read_text())["human_comparison"]["reviewed_clips"] == 1


def test_cli_rejects_overwriting_video(video):
    before = video.read_bytes()
    assert main(["evaluate", "--video", str(video), "--task", "move", "--output", str(video)]) == 1
    assert video.read_bytes() == before


def test_shake_detection(tmp_path, metadata):
    path = tmp_path / "shake.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 480))
    background = np.random.default_rng(10).integers(30, 220, (480, 640, 3), dtype=np.uint8)
    for i in range(50):
        writer.write(np.roll(background, (i % 2) * 120, axis=1))
    writer.release()
    result = evaluate(path, metadata, Config(), tmp_path / "cache")
    assert result.technical.camera_stability_score < 0.5
    assert result.technical.flags["camera_stability"] is False


@pytest.mark.parametrize("extension", ["csv", "jsonl"])
def test_aggregate_exports(tmp_path, extension):
    results = tmp_path / "empty.jsonl"
    results.write_text("")
    output = tmp_path / f"analytics.{extension}"
    assert main(["analytics", "--results", str(results), "--output", str(output)]) == 0
    if extension == "csv":
        import csv

        with output.open() as stream:
            rows = list(csv.DictReader(stream))
    else:
        rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert {row["section"] for row in rows} >= {
        "dataset",
        "human_comparison",
        "quality_distribution",
    }


@pytest.mark.parametrize(
    "timestamps,ratio,gaps",
    [([0, 0.05, 0.1], 0, 0), ([0, 0.05, 0.2], 0.4, 1), ([0, 0], None, 0), ([], None, 0)],
)
def test_timestamp_gap_estimates(timestamps, ratio, gaps):
    from qc.technical import estimate_dropped_frames

    measured, count = estimate_dropped_frames(timestamps, 20, 1.8)
    assert count == gaps
    if ratio is None:
        assert measured is None
    else:
        assert measured == pytest.approx(ratio)


def test_collector_identity_not_sent_to_vlm(video, metadata, semantic, tmp_path):
    class AnonymousProvider(Provider):
        def evaluate(self, request):
            assert "collector_id" not in request["metadata"]
            return super().evaluate(request)

    provider = AnonymousProvider(semantic)
    evaluate(video, metadata, Config(), tmp_path / "cache", Detector(), provider)
    metadata.collector_id = "another-collector"
    second = evaluate(video, metadata, Config(), tmp_path / "cache", Detector(), provider)
    assert provider.calls == 1
    assert second.collector_id == "another-collector"
