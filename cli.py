"""Command-line entry point. Batch metadata paths are relative to --input."""

import argparse
import csv
import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from qc.analytics import summarize, summary_rows
from qc.audit import append_review
from qc.config import Config, load_config
from qc.pipeline import atomic_writer, evaluate, save_report, write_json
from qc.schemas import HumanReview, Metadata, Result
from qc.vlm import FileProvider

LOG = logging.getLogger("ego-q")
T = TypeVar("T", bound=BaseModel)


def plugin(spec: str | None) -> Any:
    """Load a trusted local adapter factory (never taken from video metadata)."""
    if not spec:
        return None
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)()


def model_adapters(args: argparse.Namespace, cfg: Config) -> tuple[Any, Any]:
    if args.hand_model:
        cfg.hands.model_path = str(args.hand_model)
    if args.vlm_model:
        cfg.vlm.model = args.vlm_model
    if args.detector == "mediapipe":
        from qc.adapters.mediapipe_hands import MediaPipeHands

        detector = MediaPipeHands(cfg.hands)
    else:
        detector = plugin(args.detector)
    if args.provider == "openai":
        from qc.adapters.openai_vlm import OpenAIProvider

        provider = OpenAIProvider(cfg.vlm)
    else:
        provider = plugin(args.provider)
    return detector, provider


def read_results(path: Path) -> list[Result]:
    return (
        [Result.model_validate_json(path.read_text())]
        if path.suffix == ".json"
        else read_jsonl(path, Result)
    )


def metadata_row(row: dict[str, str]) -> Metadata:
    return Metadata.model_validate(
        {
            key: value
            for key, value in row.items()
            if key in Metadata.model_fields and value not in (None, "")
        }
    )


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    return [
        model.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def protect_inputs(outputs: list[Path], inputs: list[Path]) -> None:
    resolved = [p.resolve() for p in outputs]
    if len(set(resolved)) != len(resolved) or set(resolved).intersection(
        p.resolve() for p in inputs
    ):
        raise ValueError("Output paths must be distinct and must not overwrite inputs")


def batch(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    detector, provider = model_adapters(args, cfg)
    with args.metadata.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"video", "collector_id", "task_name", "expected_task_description"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Metadata CSV requires {sorted(required)}")
        rows = list(reader)
    error_output = args.output.with_suffix(".errors.json")
    protect_inputs(
        [args.output, error_output],
        [args.metadata] + [args.input / r["video"] for r in rows if r.get("video")],
    )
    errors = []
    with atomic_writer(args.output) as output:
        for index, row in enumerate(rows, 2):
            try:
                if not row.get("video"):
                    raise ValueError("Video path is required")
                path = (args.input / row["video"]).resolve()
                if not path.is_relative_to(args.input.resolve()):
                    raise ValueError("Video path escapes input directory")
                result = evaluate(path, metadata_row(row), cfg, args.cache, detector, provider)
                save_report(
                    result,
                    args.output.parent
                    / (args.output.stem + "-reports")
                    / f"{index}-{result.video_id}.txt",
                )
                output.write(result.model_dump_json() + "\n")
            except Exception as exc:
                LOG.exception("Failed metadata row %d", index)
                errors.append(dict(row=index, video=row.get("video"), error=str(exc)))
    write_json(error_output, dict(errors=errors))
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explainable egocentric manipulation QC")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("evaluate", "evaluate-folder"):
        p = sub.add_parser(command)
        p.add_argument("--config", type=Path)
        p.add_argument("--cache", type=Path, default=Path(".qc-cache"))
        p.add_argument(
            "--detector", help="mediapipe or Python module:factory implementing Detector"
        )
        p.add_argument("--provider", help="openai or Python module:factory implementing Provider")
        p.add_argument("--hand-model", type=Path, help="MediaPipe .task asset path")
        p.add_argument("--vlm-model", help="Override pinned VLM model in configuration")
        p.add_argument(
            "--output",
            type=Path,
            default=Path("results.json" if command == "evaluate" else "results.jsonl"),
        )
        if command == "evaluate":
            p.add_argument("--video", type=Path, required=True)
            p.add_argument("--task", required=True)
            p.add_argument("--collector-id", default="unknown")
            p.add_argument("--expected-task-description")
            p.add_argument("--expected-object")
            p.add_argument("--min-duration", type=float)
            p.add_argument("--max-duration", type=float)
            p.add_argument(
                "--semantic-json", type=Path, help="Replay one external VLM JSON response"
            )
            p.add_argument("--export-vlm-request", type=Path)
        else:
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--metadata", type=Path, required=True)
    p = sub.add_parser("audit-add")
    p.add_argument("--video-id", required=True)
    p.add_argument("--reviewer-id", required=True)
    p.add_argument("--score", type=float, required=True)
    p.add_argument("--decision", choices=["PASS", "REVIEW", "FAIL"], required=True)
    p.add_argument("--task-correct", choices=["true", "false"])
    p.add_argument("--notes", default="")
    p.add_argument("--output", type=Path, default=Path("reviews.jsonl"))
    p = sub.add_parser("analytics")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--reviews", type=Path)
    p.add_argument("--output", type=Path, default=Path("analytics.json"))
    p = sub.add_parser("download-hand-model")
    p.add_argument("--output", type=Path, default=Path("models/hand_landmarker.task"))
    p.add_argument("--sha256", help="Optional expected SHA256")
    p = sub.add_parser("export-review")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--percent", type=float, default=100)
    p.add_argument("--seed", default="calibration-v1")
    p = sub.add_parser("calibrate")
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("calibration-output"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "download-hand-model":
            from qc.models import download_hand_model

            print("Hand model SHA256:", download_hand_model(args.output, args.sha256))
            return 0
        if args.command == "export-review":
            from qc.calibration import export_review

            print(
                json.dumps(
                    export_review(read_results(args.results), args.output, args.percent, args.seed)
                )
            )
            return 0
        if args.command == "calibrate":
            from qc.calibration import save_analysis

            protect_inputs(
                [
                    args.output / "calibration.json",
                    args.output / "paired_reviews.jsonl",
                    args.output / "calibration.txt",
                ],
                [args.results, args.annotations],
            )
            report = save_analysis(read_results(args.results), args.annotations, args.output)
            print(
                json.dumps(
                    dict(
                        evaluations=report["evaluations"],
                        overall=report["overall"],
                        output=str(args.output),
                    )
                )
            )
            return 0
        if args.command == "evaluate-folder":
            return batch(args)
        if args.command == "audit-add":
            review = HumanReview(
                video_id=args.video_id,
                reviewer_id=args.reviewer_id,
                quality_score=args.score,
                decision=args.decision,
                notes=args.notes,
                task_correct=None if args.task_correct is None else args.task_correct == "true",
                reviewed_at=datetime.now(timezone.utc),
            )
            append_review(args.output, review)
        elif args.command == "analytics":
            inputs = [args.results] + ([args.reviews] if args.reviews else [])
            protect_inputs([args.output], inputs)
            if args.results.suffix == ".json":
                results = [Result.model_validate_json(args.results.read_text())]
            else:
                results = read_jsonl(args.results, Result)
            reviews = read_jsonl(args.reviews, HumanReview) if args.reviews else []
            summary = summarize(results, reviews)
            if args.output.suffix in {".jsonl", ".csv"}:
                rows = summary_rows(summary)
                with atomic_writer(args.output) as stream:
                    if args.output.suffix == ".jsonl":
                        for row in rows:
                            stream.write(json.dumps(row, allow_nan=False) + "\n")
                    else:
                        columns = ["section"] + sorted(
                            set().union(*(row.keys() for row in rows)) - {"section"}
                        )
                        writer = csv.DictWriter(stream, fieldnames=columns)
                        writer.writeheader()
                        writer.writerows(rows)
            else:
                write_json(args.output, summary)
        else:
            cfg = load_config(args.config)
            detector, provider = model_adapters(args, cfg)
            if args.semantic_json:
                if provider:
                    raise ValueError("Choose --provider or --semantic-json")
                provider = FileProvider(args.semantic_json)
            report_output = args.output.with_suffix(".report.txt")
            protect_inputs(
                [args.output, report_output, report_output.with_suffix(".html")]
                + ([args.export_vlm_request] if args.export_vlm_request else []),
                [args.video]
                + ([args.semantic_json] if args.semantic_json else [])
                + ([args.config] if args.config else []),
            )
            metadata = Metadata(
                collector_id=args.collector_id,
                task_name=args.task,
                expected_task_description=args.expected_task_description or args.task,
                expected_object=args.expected_object,
                min_duration=args.min_duration,
                max_duration=args.max_duration,
            )
            result = evaluate(
                args.video, metadata, cfg, args.cache, detector, provider, args.export_vlm_request
            )
            write_json(args.output, result.model_dump(mode="json"))
            save_report(result, report_output)
            print(f"{result.decision} {result.quality_score:.2f}/100 → {args.output}")
    except Exception as exc:
        LOG.error("%s: %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
