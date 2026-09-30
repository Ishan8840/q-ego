"""Model contracts, visual evidence, independent annotations and calibration math."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from cli import main
from qc.adapters.openai_vlm import OpenAIProvider
from qc.calibration import (
    DIMENSIONS,
    HARD_FIELDS,
    analyze,
    evaluation_id,
    export_review,
    load_annotations,
)
from qc.config import Config, HandConfig, VLMConfig
from qc.ingest import video_digest
from qc.pipeline import evaluate
from qc.report import generate_report
from qc.schemas import (
    CalibrationReview,
    Metadata,
    Result,
    Semantic,
    SemanticJudgment,
    Technical,
    Visibility,
)
from qc.scoring import score
from qc.visibility import Detections, HandBox
from qc.visibility import analyze as visibility_analyze


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
    assert writer.isOpened()
    background = np.random.default_rng(2).integers(30, 220, (480, 640, 3), dtype=np.uint8)
    for i in range(20):
        writer.write(np.roll(background, i, axis=1))
    writer.release()
    return path


@pytest.fixture
def judgment():
    return SemanticJudgment(
        task_correct=True,
        task_complete_score=1.0,
        hand_visibility_score=1.0,
        object_visibility_score=1.0,
        interaction_visibility_score=1.0,
        final_state_success=True,
        major_occlusion=False,
        irrelevant_footage_score=0.0,
        protocol_violation=False,
        reasoning_summary="Visible task and final state.",
    )


@pytest.fixture
def result(video, judgment):
    metadata = Metadata(
        collector_id="c1", task_name="move cup", expected_task_description="Move cup onto table"
    )
    config = Config(audit_percent=0)
    t = Technical(
        duration=2,
        width=640,
        height=480,
        fps=10,
        blur_score=1,
        exposure_score=1,
        camera_stability_score=1,
        flags=dict(resolution=True, fps=True, duration=True),
        frame_metrics=[dict(frame=0, timestamp=0.0)],
    )
    v = Visibility(
        detector_id="fixture",
        hand_visible_ratio=1,
        hand_boundary_ratio=0,
        object_visible_ratio=1,
        hand_object_visible_ratio=1,
        frame_metrics=[
            dict(
                frame=0,
                timestamp=0.0,
                hand=True,
                object=True,
                in_view=True,
                boundary=False,
                hands=[
                    dict(
                        x1=0.2,
                        y1=0.2,
                        x2=0.5,
                        y2=0.5,
                        handedness="Left",
                        track_id=1,
                        confidence=0.95,
                        confidence_kind="handedness_classification",
                    )
                ],
            )
        ],
    )
    return Result(
        video_id=video_digest(video),
        collector_id="c1",
        task_name="move cup",
        metadata=metadata,
        technical=t,
        visibility=v,
        semantic=judgment,
        **score(t, v, judgment, config),
        payment_multiplier=1,
        payment_status="recommended",
        config_snapshot=config.model_dump(),
        config_digest="config",
        evaluated_at="2026-09-29T12:00:00Z",
        source_video=str(video),
    )


def human(result, decision="PASS", rating=5, reviewer="human-1"):
    return CalibrationReview(
        video_id=result.video_id,
        evaluation_id=evaluation_id(result),
        reviewer_id=reviewer,
        timestamp=datetime(2026, 9, 29, tzinfo=timezone.utc),
        task_correct=True,
        **{k: rating for k in DIMENSIONS},
        **{k: False for k in HARD_FIELDS},
        human_decision=decision,
    )


def test_openai_payload_and_validation(judgment):
    recorded = {}

    def create(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(status="completed", output_text=judgment.model_dump_json())

    provider = OpenAIProvider(client=SimpleNamespace(responses=SimpleNamespace(create=create)))
    output = provider.evaluate(
        dict(
            prompt="Evaluate evidence",
            metadata={"expected_task_description": "move cup"},
            frames=[dict(timestamp=0.0, jpeg_base64="AAA"), dict(timestamp=1.0, jpeg_base64="BBB")],
        )
    )
    assert SemanticJudgment.model_validate_json(output) == judgment
    assert recorded["temperature"] == 0
    assert recorded["store"] is False
    assert recorded["model"] == "gpt-4.1-mini-2025-04-14"
    schema = recorded["text"]["format"]["schema"]
    assert "major_occlusion" in schema["required"]
    assert schema["additionalProperties"] is False
    content = recorded["input"][1]["content"]
    assert content[1]["text"] == "Frame at 0.000 seconds"
    assert content[4]["image_url"].endswith("BBB")


@pytest.mark.parametrize(
    "status,text", [("incomplete", "{}"), ("completed", ""), ("completed", '{"task_correct":true}')]
)
def test_openai_failures_not_scores(status, text):
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(status=status, output_text=text)
        )
    )
    with pytest.raises(ValueError):
        OpenAIProvider(client=client).evaluate(
            dict(prompt="p", metadata={}, frames=[dict(timestamp=0, jpeg_base64="AA")])
        )


def test_provider_cache_identity_changes():
    default = OpenAIProvider().cache_id
    assert OpenAIProvider(VLMConfig(model="other-version")).cache_id != default
    assert OpenAIProvider(VLMConfig(temperature=0.2)).cache_id != default
    assert OpenAIProvider().cache_id == default


def test_occlusion_requires_review(result):
    result.semantic.major_occlusion = True
    scored = score(result.technical, result.visibility, result.semantic, Config())
    assert scored["decision"] == "REVIEW"
    assert "major_occlusion" in scored["failure_reasons"]


def test_legacy_semantic_remains_readable(judgment):
    data = judgment.model_dump()
    del data["major_occlusion"]
    assert Semantic.model_validate(data).major_occlusion is None
    with pytest.raises(ValidationError):
        SemanticJudgment.model_validate(data)


def test_visibility_temporal_metrics_and_lifecycle(video):
    class Sequence:
        cache_id = "sequence"

        def start_video(self):
            self.index = 0
            self.started = True

        def close(self):
            self.closed = True

        def detect_at(self, frame, expected_object, timestamp):
            present = self.index not in {2, 3}
            self.index += 1
            return Detections(
                hands=[HandBox(x1=0.1, y1=0.2, x2=0.4, y2=0.5, confidence=0.9, track_id=1)]
                if present
                else []
            )

    detector = Sequence()
    cfg = Config(visibility_sample_fps=2)
    first = visibility_analyze(video, detector, None, cfg)
    assert detector.started and detector.closed
    assert first.hand_visible_ratio == 0.5
    assert first.longest_hand_missing_interval == pytest.approx(1)
    assert first.manipulation_offscreen_ratio == pytest.approx(0.5)
    assert first.offscreen_evidence_source == "hand_absence_proxy"
    assert first.hand_tracking_continuity_score == pytest.approx(1 / 3)
    detail = first.frame_metrics[0]["hands"][0]
    assert detail["confidence"] == 0.9
    assert detail["boundary_distance"] == 0.1
    assert detail["boundary_distance_pixels"] == 64
    second = visibility_analyze(video, detector, None, cfg)
    assert second == first


def test_visibility_unknown_is_not_absence(video):
    class Unknown:
        cache_id = "unknown"

        def detect(self, frame, expected_object):
            return Detections()

    result = visibility_analyze(video, Unknown(), None, Config())
    assert result.hand_visible_ratio is None
    assert result.longest_hand_missing_interval is None
    assert result.manipulation_offscreen_ratio is None
    assert result.hand_tracking_continuity_score is None


def test_close_after_detector_exception(video):
    class Broken:
        cache_id = "broken"
        closed = False

        def close(self):
            self.closed = True

        def detect(self, *args):
            raise RuntimeError("inference failed")

    detector = Broken()
    with pytest.raises(RuntimeError):
        visibility_analyze(video, detector, None, Config())
    assert detector.closed


def test_report_annotations_and_blinding(video, result, tmp_path):
    result.metadata.expected_task_description = "<script>alert(1)</script>"
    result.semantic.reasoning_summary = "PRIVATE AUTOMATED REASON"
    path = tmp_path / "qc_report.html"
    generate_report(video, result, path)
    text = path.read_text()
    assert "data:image/jpeg;base64," in text
    assert "0.000s" in text
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;" in text
    assert "PRIVATE AUTOMATED REASON" in text
    blind = tmp_path / "blind.html"
    generate_report(video, result, blind, blind=True, video_link="videos/original.avi")
    text = blind.read_text()
    assert "PRIVATE AUTOMATED REASON" not in text
    assert "Quality score" not in text
    assert 'class="PASS"' not in text
    assert "<video controls" in text


def test_missing_hand_report_warning(video, result, tmp_path):
    result.visibility.frame_metrics[0].update(hand=False, in_view=False, hands=[])
    path = tmp_path / "warnings.html"
    generate_report(video, result, path)
    assert "HAND MISSING" in path.read_text()
    assert "OFFSCREEN" in path.read_text()


def test_blind_export_and_template(result, tmp_path):
    destination = tmp_path / "review"
    summary = export_review([result], destination)
    assert summary["exported"] == 1
    assert len(list((destination / "reviewer/videos").iterdir())) == 1
    template = destination / "reviewer/human_reviews.csv"
    with template.open() as stream:
        row = next(csv.DictReader(stream))
    assert row["reviewer_id"] == ""
    assert row["overall_usability"] == ""
    assert "automated_decision" not in row
    annotations, blank = load_annotations(template)
    assert annotations == [] and blank == 1
    with pytest.raises(ValueError, match="empty directory"):
        export_review([result], destination)


def test_export_rejects_changed_source(result, video, tmp_path):
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="contents changed"):
        export_review([result], tmp_path / "review")


def test_calibration_known_errors(result):
    results = []
    annotations = []
    for index, (auto, quality, decision, rating) in enumerate(
        [
            ("PASS", 90, "PASS", 5),
            ("PASS", 80, "FAIL", 1),
            ("FAIL", 20, "PASS", 4),
            ("REVIEW", 50, "REVIEW", 3),
        ]
    ):
        r = result.model_copy(deep=True)
        r.video_id = f"video-{index}"
        r.quality_score = quality
        r.decision = auto
        results.append(r)
        annotations.append(human(r, decision, rating))
    summary, paired = analyze(results, annotations)
    assert summary["overall"]["mae"] == 36.25
    assert summary["overall"]["pearson"] < 0
    assert summary["false_acceptance"] == dict(numerator=1, denominator=1, rate=1)
    assert summary["false_rejection"] == dict(numerator=1, denominator=2, rate=0.5)
    assert summary["unsafe_pass_fraction"]["rate"] == 0.5
    assert summary["confusion_matrix"]["counts"]["PASS"]["FAIL"] == 1
    assert len(paired) == 4
    assert "automated_metrics" in paired[0] and "human_metrics" in paired[0]
    assert summary["individual_dimensions"]["overall_usability"]["mae"] == pytest.approx(0.3625)


def test_empty_and_constant_calibration(result):
    summary, _ = analyze([], [])
    assert summary["overall"]["pearson"] is None
    assert summary["overall"]["mae"] is None
    assert summary["false_rejection"]["rate"] is None
    single, _ = analyze([result], [human(result)])
    assert single["overall"]["pearson"] is None


def test_duplicate_votes_consensus_and_stale_join(result):
    first = human(result)
    second = human(result, "FAIL", 1, "human-2")
    summary, paired = analyze([result], [first, first, second])
    assert len(paired) == 2
    assert summary["confusion_matrix"]["counts"]["REVIEW"]["PASS"] == 1
    result.config_digest = "changed"
    with pytest.raises(ValueError, match="stale evaluation"):
        analyze([result], [first])


def test_human_schema_and_csv_reject_bad_ratings(result, tmp_path):
    row = human(result).model_dump(mode="json")
    row["hand_visibility"] = 6
    path = tmp_path / "ratings.csv"
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="row 2"):
        load_annotations(path)
    row["hand_visibility"] = 5
    row["corrupted_video"] = True
    with pytest.raises(ValidationError, match="hard failure"):
        CalibrationReview.model_validate(row)


def test_calibration_cli_and_saved_pairs(result, tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(result.model_dump_json() + "\n")
    annotations = tmp_path / "human_reviews.csv"
    row = human(result).model_dump(mode="json")
    with annotations.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "calibration"
    assert (
        main(
            [
                "calibrate",
                "--results",
                str(results),
                "--annotations",
                str(annotations),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert (output / "calibration.txt").is_file()
    saved = json.loads((output / "paired_reviews.jsonl").read_text())
    assert saved["reviewer_id"] == "human-1"
    assert saved["automated_decision"] == "PASS"


def test_native_hand_tracker_blank_frame_and_reset():
    pytest.importorskip("mediapipe")
    model = Path("models/hand_landmarker.task")
    if not model.is_file():
        pytest.skip("Optional local model not downloaded")
    from qc.adapters.mediapipe_hands import MediaPipeHands

    detector = MediaPipeHands(HandConfig(model_path=str(model)))
    try:
        assert detector.detect_at(np.zeros((480, 640, 3), dtype=np.uint8), None, 0).hands == []
        detector.start_video()
        assert detector.detect_at(np.zeros((480, 640, 3), dtype=np.uint8), None, 0).hands == []
    finally:
        detector.close()


def test_mediapipe_conversion_confidence_and_geometry():
    from qc.adapters.mediapipe_hands import MediaPipeHands

    detector = object.__new__(MediaPipeHands)
    detector.config = HandConfig(box_padding=0, swap_handedness=True)
    native = SimpleNamespace(
        hand_landmarks=[[SimpleNamespace(x=-0.1, y=0.1), SimpleNamespace(x=0.8, y=0.9)]],
        handedness=[[SimpleNamespace(score=0.96, category_name="Left")]],
    )
    box = detector.convert(native)[0]
    assert box.x1 == 0 and box.x2 == 0.8
    assert box.handedness == "Right"
    assert box.confidence == 0.96
    assert box.confidence_kind == "handedness_classification"
    assert box.detection_confidence is None


def test_occlusion_and_provider_cache_pipeline(video, judgment, tmp_path):
    calls = []
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **kwargs: (
                calls.append(kwargs)
                or SimpleNamespace(status="completed", output_text=judgment.model_dump_json())
            )
        )
    )
    provider = OpenAIProvider(client=client)
    metadata = Metadata(collector_id="c", task_name="move", expected_task_description="move cup")
    for _ in range(2):
        result = evaluate(video, metadata, Config(), tmp_path / "cache", provider=provider)
        assert result.semantic.major_occlusion is False
    assert len(calls) == 1
    assert result.evidence_confidence["calibrated_probability"] is None


def test_actual_sdk_contract_with_mock_transport(judgment):
    openai = pytest.importorskip("openai")
    import httpx

    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=dict(
                id="resp_test",
                object="response",
                created_at=0,
                status="completed",
                model="gpt-4.1-mini-2025-04-14",
                error=None,
                output=[
                    dict(
                        id="msg_test",
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[
                            dict(
                                type="output_text", text=judgment.model_dump_json(), annotations=[]
                            )
                        ],
                    )
                ],
            ),
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as transport:
        with openai.OpenAI(
            api_key="test-not-a-real-key", http_client=transport, max_retries=0
        ) as client:
            answer = OpenAIProvider(client=client).evaluate(
                dict(
                    prompt="Evaluate task",
                    metadata={},
                    frames=[dict(timestamp=0, jpeg_base64="AA")],
                )
            )
    assert SemanticJudgment.model_validate_json(answer) == judgment
    assert requests[0]["text"]["format"]["strict"] is True


def test_timeout_changes_reuse_provider_identity():
    assert (
        OpenAIProvider(VLMConfig(timeout_seconds=10, max_retries=0)).cache_id
        == OpenAIProvider().cache_id
    )
