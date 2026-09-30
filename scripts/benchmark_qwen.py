"""Measure real local inference on an explicit negative-control clip.

Run as: python -m scripts.benchmark_qwen --video clip.avi --output output/qwen-benchmark
This is an inference/contract smoke test, not a manipulation-accuracy benchmark.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from qc.adapters.mediapipe_hands import MediaPipeHands
from qc.adapters.ollama_vlm import make_provider
from qc.config import Config
from qc.pipeline import evaluate, save_report, write_json
from qc.schemas import Metadata, SemanticJudgment
from qc.vlm import make_request


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provider = make_provider()
    metadata = Metadata(
        collector_id="synthetic-smoke-test",
        task_name="put cup in drawer",
        expected_task_description="Open a drawer, place a cup inside, then close the drawer. The complete hand-object interaction must be visible.",
        expected_object="cup",
    )
    samples = []
    stopped = threading.Event()

    def monitor():
        while not stopped.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                )
                used, utilization = map(int, output.strip().splitlines()[0].split(","))
                samples.append(
                    dict(time=time.time(), memory_mib=used, utilization_percent=utilization)
                )
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            stopped.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    runs = []
    try:
        for count in (2, 6, 12):
            request = make_request(args.video, metadata, count)
            start = time.perf_counter()
            judgment = SemanticJudgment.model_validate_json(provider.evaluate(request))
            elapsed = time.perf_counter() - start
            metrics = provider.last_metrics
            duration = metrics.get("eval_duration") or 0
            run = dict(
                frames=count,
                wall_seconds=elapsed,
                api_metrics=metrics,
                output_tokens_per_second=metrics["eval_count"] / (duration / 1e9)
                if duration
                else None,
                semantic=judgment.model_dump(),
                negative_control_rejected=not judgment.task_correct,
            )
            runs.append(run)
            write_json(args.output / f"{count}-frames.json", run)
            print(
                json.dumps(
                    dict(
                        frames=count, seconds=round(elapsed, 2), task_correct=judgment.task_correct
                    )
                ),
                flush=True,
            )
        cfg = Config(audit_percent=0)
        detector = MediaPipeHands(cfg.hands)
        start = time.perf_counter()
        result = evaluate(args.video, metadata, cfg, args.output / "cache", detector, provider)
        pipeline_seconds = time.perf_counter() - start
        write_json(args.output / "result.json", result.model_dump(mode="json"))
        save_report(result, args.output / "result.report.txt")
        start = time.perf_counter()
        cached = evaluate(args.video, metadata, cfg, args.output / "cache", detector, provider)
        cache_seconds = time.perf_counter() - start
        assert cached.semantic == result.semantic
        summary = dict(
            model=provider.config.model,
            model_digest=provider.model_digest,
            runtime_version=provider.runtime_version,
            provider_config=provider.config.model_dump(),
            runs=runs,
            pipeline_seconds=pipeline_seconds,
            cached_pipeline_seconds=cache_seconds,
            decision=result.decision,
            quality_score=result.quality_score,
            peak_gpu_memory_mib=max((s["memory_mib"] for s in samples), default=None),
            max_gpu_utilization_percent=max(
                (s["utilization_percent"] for s in samples), default=None
            ),
            limitation="Negative control made from repeated public hand photograph. Not a representative manipulation benchmark or human calibration dataset.",
        )
        write_json(args.output / "benchmark.json", summary)
    finally:
        stopped.set()
        thread.join(timeout=6)
        provider.close()
    write_json(args.output / "gpu_samples.json", dict(samples=samples))


if __name__ == "__main__":
    main()
