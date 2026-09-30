"""Reproducible public-data integration test; dataset labels are not human QC truth.

Prepare selected original MP4s from the official test ZIP, then run existing QC.
Only HDF5 language attributes are read; no pose/reconstruction code is used.
"""

import argparse
import csv
import hashlib
import html
import io
import json
import logging
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from qc.config import Config
from qc.ingest import video_digest
from qc.pipeline import atomic_writer, evaluate, save_report, write_json
from qc.schemas import Metadata

SOURCE = "https://ml-site.cdn-apple.com/datasets/egodex/test.zip"
LOG = logging.getLogger(__name__)


def description(attrs: dict[str, Any]) -> str:
    """Resolve reversible-task direction, refusing an ambiguous annotation."""

    def scalar(value: Any) -> Any:
        if hasattr(value, "item"):
            value = value.item()
        return value.decode("utf-8") if isinstance(value, bytes) else value

    attrs = {
        key: scalar(attrs[key])
        for key in ("llm_description", "llm_description2", "which_llm_description")
        if key in attrs
    }
    # The official archive uses the literal string "None" for absent alternatives.
    for key, value in attrs.items():
        if isinstance(value, str) and value.strip().lower() in {"none", "null", ""}:
            attrs[key] = None
    direction = attrs.get("which_llm_description")
    if direction is None:
        if attrs.get("llm_description2"):
            raise ValueError("Reversible task has no description selector")
        key = "llm_description"
    elif str(direction) in {"1", "1.0"}:
        key = "llm_description"
    elif str(direction) in {"2", "2.0"}:
        key = "llm_description2"
    else:
        raise ValueError(f"Invalid description selector: {direction!r}")
    text = attrs.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Missing {key}")
    return text.strip()


def select_members(names: list[str], tasks: int, per_task: int, seed: str) -> list[str]:
    if tasks < 1 or per_task < 1:
        raise ValueError("Sample counts must be positive")
    available = set(names)
    groups: dict[str, list[str]] = defaultdict(list)
    for name in sorted(available):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 3:
            continue
        if path.suffix == ".mp4" and str(path.with_suffix(".hdf5")) in available:
            groups[path.parent.name].append(name)

    def order(name: str) -> str:
        return hashlib.sha256((seed + ":" + name).encode()).hexdigest()

    eligible = sorted((task for task in groups if len(groups[task]) >= per_task), key=order)
    if len(eligible) < tasks:
        raise ValueError("Not enough tasks with paired clips")
    return [
        name for task in eligible[:tasks] for name in sorted(groups[task], key=order)[:per_task]
    ]


def prepare(archive: Path, destination: Path, tasks: int, per_task: int, seed: str) -> None:
    import h5py

    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Preparation requires an empty destination")
    destination.mkdir(parents=True, exist_ok=True)
    cases = []
    with zipfile.ZipFile(archive) as zipped:
        for member in select_members(zipped.namelist(), tasks, per_task, seed):
            name = PurePosixPath(member)
            annotation = zipped.read(str(name.with_suffix(".hdf5")))
            with h5py.File(io.BytesIO(annotation), "r") as hdf:
                task_description = description(dict(hdf.attrs))
            relative = Path("videos") / name.parent.name / name.name
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, path.open("wb") as output:
                shutil.copyfileobj(source, output)
            metadata = Metadata(
                collector_id="egodex:unknown",
                task_name=name.parent.name.replace("_", " "),
                expected_task_description=task_description,
            )
            cases.append(
                dict(
                    case_id=f"original-{len(cases):03}",
                    kind="original",
                    video=str(relative),
                    archive_member=member,
                    video_sha256=video_digest(path),
                    annotation_sha256=hashlib.sha256(annotation).hexdigest(),
                    metadata=metadata.model_dump(),
                )
            )
    write_json(
        destination / "manifest.json",
        dict(
            dataset="EgoDex",
            source=SOURCE,
            license="CC-BY-NC-ND (dataset)",
            seed=seed,
            cases=cases,
            annotation_origin="Dataset LLM/VLM-generated task descriptions",
            human_qc_ground_truth=False,
        ),
    )
    with (destination / "metadata.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video", *Metadata.model_fields])
        writer.writeheader()
        writer.writerows(dict(video=c["video"], **c["metadata"]) for c in cases)
    print(json.dumps(dict(prepared=len(cases), destination=str(destination))), flush=True)


def run(dataset: Path, output: Path, controls: bool) -> None:
    from qc.adapters.mediapipe_hands import MediaPipeHands
    from qc.adapters.ollama_vlm import make_provider

    manifest = json.loads((dataset / "manifest.json").read_text())
    cases = list(manifest["cases"])
    if controls:
        # A mismatched instruction probes rejection; this is a designed expectation,
        # not a human annotation. Original video bytes remain unchanged.
        seen = set()
        for case in manifest["cases"]:
            task = case["metadata"]["task_name"]
            if task in seen:
                continue
            seen.add(task)
            cases.append(
                dict(
                    case,
                    case_id=case["case_id"].replace("original", "mismatch"),
                    kind="mismatched_instruction",
                    metadata=Metadata(
                        collector_id="egodex:unknown",
                        task_name="cook an egg on a stove",
                        expected_task_description="Crack a raw chicken egg into a frying pan on a stove, cook the egg, and show the cooked egg in the pan.",
                        expected_object="egg",
                    ).model_dump(),
                )
            )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "cases.json", dict(source_manifest=manifest, cases=cases))
    cfg = Config()  # Preserve all production scoring/payment thresholds.
    detector, provider = MediaPipeHands(cfg.hands), make_provider()
    results, rows, errors = [], [], []
    try:
        for case in cases:
            start = time.perf_counter()
            try:
                path = (dataset / case["video"]).resolve()
                if not path.is_relative_to(dataset.resolve()):
                    raise ValueError("Video path escapes dataset")
                if video_digest(path) != case["video_sha256"]:
                    raise ValueError("Video differs from prepared manifest")
                result = evaluate(
                    path,
                    Metadata.model_validate(case["metadata"]),
                    cfg,
                    output / "cache",
                    detector,
                    provider,
                )
                report = output / "reports" / (case["case_id"] + ".txt")
                save_report(result, report)
                write_json(report.with_suffix(".json"), result.model_dump(mode="json"))
                results.append(result)
                semantic = result.semantic
                hand = result.visibility.hand_visible_ratio
                semantic_hand = semantic.hand_visibility_score if semantic else None
                rows.append(
                    dict(
                        case_id=case["case_id"],
                        kind=case["kind"],
                        task=result.task_name,
                        video_id=result.video_id,
                        quality_score=result.quality_score,
                        decision=result.decision,
                        human_review_required=result.human_review_required,
                        technical_available=result.technical.available,
                        semantic_available=semantic is not None,
                        task_correct=semantic.task_correct if semantic else None,
                        hand_visible_ratio=hand,
                        vlm_hand_visibility=semantic_hand,
                        hand_disagreement=abs(hand - semantic_hand)
                        if hand is not None and semantic_hand is not None
                        else None,
                        completeness=semantic.task_complete_score if semantic else None,
                        seconds=round(time.perf_counter() - start, 3),
                        failure_reasons=";".join(result.failure_reasons),
                        report="reports/" + report.with_suffix(".html").name,
                    )
                )
                print(json.dumps(rows[-1]), flush=True)
            except Exception as exc:
                LOG.exception("Case failed: %s", case["case_id"])
                errors.append(dict(case_id=case["case_id"], error=str(exc)))
            # Preserve completed cases even if the process is interrupted later.
            with atomic_writer(output / "results.jsonl") as stream:
                for result in results:
                    stream.write(result.model_dump_json() + "\n")
            write_json(output / "progress.json", dict(completed=len(rows), errors=errors))
    finally:
        detector.close()
        provider.close()
    with atomic_writer(output / "metrics.csv") as stream:
        if rows:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    groups = {}
    for kind in sorted({row["kind"] for row in rows}):
        group = [row for row in rows if row["kind"] == kind]
        judged = [row for row in group if row["semantic_available"]]
        groups[kind] = dict(
            count=len(group),
            decisions=dict(Counter(row["decision"] for row in group)),
            semantic_available=len(judged),
            task_correct=sum(row["task_correct"] for row in judged),
            mean_quality=sum(row["quality_score"] for row in group) / len(group),
            hand_disagreement_over_threshold=sum(
                row["hand_disagreement"] is not None
                and row["hand_disagreement"] > cfg.disagreement_threshold
                for row in group
            ),
        )
    summary = dict(
        attempted=len(cases),
        completed=len(rows),
        errors=errors,
        by_case_kind=groups,
        provider=dict(
            model=provider.config.model,
            digest=provider.model_digest,
            runtime=provider.runtime_version,
        ),
        human_annotations=0,
        human_agreement=None,
        limitations=[
            "Dataset descriptions are machine-generated, not human QC ground truth.",
            "Mismatch rejection is a designed control, not an accuracy estimate.",
            "Independent object detection is unavailable; acceptance safeguards remain.",
            "Hand sampling and VLM interaction-conditioned visibility differ in meaning.",
            "Small task-stratified sample; no payment threshold optimization.",
        ],
    )
    write_json(output / "summary.json", summary)
    cells = []
    for row in rows:
        cells.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row[k]))}</td>"
                for k in (
                    "case_id",
                    "task",
                    "quality_score",
                    "decision",
                    "hand_visible_ratio",
                    "vlm_hand_visibility",
                    "task_correct",
                )
            )
            + f'<td><a href="{html.escape(row["report"])}">Inspect</a></td></tr>'
        )
    (output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>EgoDex QC test</title>'
        "<style>body{font:16px system-ui;margin:30px}td,th{padding:8px;border-bottom:1px solid #ddd}"
        "table{border-collapse:collapse}pre{white-space:pre-wrap}</style>"
        "<h1>EgoDex public-data QC test</h1><p>Original clips and mismatched-instruction controls. "
        "No human QC labels; decisions are automated hypotheses.</p><table><tr>"
        "<th>Case</th><th>Task</th><th>Score</th><th>Decision</th><th>MediaPipe hands</th>"
        "<th>Qwen hands</th><th>Task correct</th><th>Report</th></tr>"
        + "".join(cells)
        + "</table><h2>Summary</h2><pre>"
        + html.escape(json.dumps(summary, indent=2))
        + "</pre></html>"
    )
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--per-task", type=int, default=3)
    p.add_argument("--seed", default="ego-q-public-v1")
    p = commands.add_parser("run")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.command == "prepare":
        prepare(args.archive, args.output, args.tasks, args.per_task, args.seed)
    else:
        run(args.dataset, args.output, args.controls)


if __name__ == "__main__":
    main()
