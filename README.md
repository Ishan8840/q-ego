# Ego-Q

A local, explainable quality-control MVP for egocentric manipulation videos. It decodes videos, measures technical quality, evaluates detector visibility and optional VLM evidence, applies configurable scoring/payment rules, and exports human-audit analytics.

**Real adapters are available:** MediaPipe CPU hand tracking, OpenAI structured semantic QC, and local Qwen vision via Ollama. Enable them explicitly with `--detector mediapipe --provider openai`. Without configured models, missing evidence still goes to `REVIEW`. No payment is executed. See [models and visual reports](docs/models-and-reports.md) and [human calibration](docs/calibration.md).

## Architecture and dependencies

```text
video + metadata
  ├─ ingest.py       SHA-256 identity, FFprobe metadata, FFmpeg decode validation
  ├─ technical.py    full decode; sampled blur/exposure/shake; motion/timing metrics
  ├─ visibility.py   replaceable hand/active-object detector; sampled visibility
  └─ vlm.py          timestamped JPEGs + task + strict schema → provider judgment
          ↓ separately cached evidence
     scoring.py      weighted components, hard overrides, review gates
     payment.py      percentage-of-base recommendation / review hold
     audit.py        reproducible PASS sampling, stored human judgments, comparisons
     analytics.py    collector/task summaries and score distributions
          ↓
     result JSON / JSONL + compact text report
```

`schemas.py` contains Pydantic output models, `config.py` validates YAML, `pipeline.py` orchestrates stages, and `cli.py` exposes commands. `tests/` contains rule tests and synthetic-video integration tests.

Dependencies: Python 3.11+, **FFmpeg and FFprobe on PATH**, OpenCV contrib, NumPy, Pydantic 2, PyYAML. Development dependencies: pytest and Ruff. No credentials or network access are needed to evaluate with deterministic checks or replay supplied judgments.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[models,test]'
ffmpeg -version
ffprobe -version
pytest -q
ruff check .
```


## Real-model evaluation and calibration

```sh
python cli.py download-hand-model
# Configure OPENAI_API_KEY in your environment for remote semantic evaluation.
python cli.py evaluate --video example.mp4 --task "put cup in drawer" \
  --collector-id c17 --expected-object cup \
  --detector mediapipe --provider openai --output output/clip.json

python cli.py export-review --results output/results.jsonl --output review-bundle
# Share only review-bundle/reviewer/; reviewers complete human_reviews.csv.
python cli.py calibrate --annotations review-bundle/reviewer/human_reviews.csv \
  --results review-bundle/analyst/automated_results.jsonl --output calibration-output
```

The HTML report shows chronological annotated frames, warning labels, scores, reasons, and evidence limitations. Review exports include full original clips, blinded HTML, a rating rubric and CSV template. Calibration emits correlations, MAE, a decision confusion matrix, false acceptance/rejection rates, dimension disagreement, metric associations and joined human/automated records. No weights, payment amounts or thresholds are optimized.

## Evaluate a video

```sh
python cli.py evaluate --video example.mp4 --task "put cup in drawer"

python cli.py evaluate --video example.mp4 \
  --collector-id collector-17 \
  --task "put cup in drawer" \
  --expected-task-description "Open the drawer, put the cup inside, then close it. Show the complete action." \
  --expected-object cup --min-duration 3 --max-duration 90 \
  --config config.yaml --output output/clip.json
```

Outputs: `output/clip.json`, `output/clip.report.txt`, and the self-contained annotated `output/clip.report.html`. The short command uses `collector_id="unknown"` and the task name as the expected description; supply actual collector IDs for payment analytics. CLI exit status indicates processing success (`0`) or an operational/input error (`1`), **not** whether the clip passed QC. A valid corrupt-file evaluation produces a `FAIL` result with exit status 0.

JSON includes raw sampled frame metrics, technical flags (`true`, `false`, or `null` for unavailable), detector visibility ratios and possible offscreen periods, semantic evidence, five component scores, weighted point contributions, decision reasons, review/payment status, metadata, configuration snapshot/digest, provider identity, and evaluation timestamp. `video_id` is the full SHA-256 of the file contents.

## Batch evaluation

```sh
python cli.py evaluate-folder --input ./videos --metadata metadata.csv \
  --config config.yaml --output output/results.jsonl
```

The CSV drives the batch; it must list every clip you want evaluated. `video` is relative to `--input`, may contain subdirectories, and cannot escape that directory. There are four required columns; optional columns can be blank:

```csv
video,collector_id,task_name,expected_task_description,expected_object,min_duration,max_duration
clip-001.mp4,c17,put cup in drawer,"Open drawer, insert cup, close drawer",cup,3,90
clip-002.mp4,c18,fold towel,Fold the towel completely,towel,,
```

Each processed video generates a JSONL row plus text and annotated HTML files in `results-reports/`. Missing files, bad metadata, and row failures appear in `results.errors.json`; the rest of the batch continues and the command exits 1 if any row failed. Completed output files are replaced atomically. Re-running a batch replaces its JSONL; evidence caches avoid repeated expensive work. A batch with no data rows produces an empty JSONL.

## Connect a detector or VLM

Use trusted local factories:

```sh
python cli.py evaluate --video example.mp4 --task "put cup in drawer" \
  --detector my_adapters:make_detector --provider my_adapters:make_provider
```

`qc.visibility.Detector` requires:

- `cache_id: str`, identifying the model revision, confidence thresholds, and preprocessing.
- `detect(frame: numpy.ndarray, expected_object: str | None) -> Detections`.
- Frames are OpenCV BGR. Boxes use normalized `x1,y1,x2,y2` coordinates in `[0,1]`.
- `hands=[]` means no hands detected; `hands=None` means the capability is unavailable. Objects use the same distinction. Object detections must refer to the **active/expected manipulated object**, not every object in the scene. Apply confidence filtering inside the adapter.
- A detector exception invalidates visibility evidence for that evaluation and routes it to review. Partial detector coverage also leaves aggregate ratios unavailable.

`qc.vlm.Provider` requires:

- `cache_id: str`, identifying the provider, model revision, and generation settings.
- `evaluate(request: dict) -> str`, returning only strict semantic JSON.
- The request contains the video content ID, prompt/version, task metadata (excluding collector identity and deterministic duration limits), JSON Schema, and ordered `{timestamp, jpeg_base64}` frames. Images are JPEG, with the longer side at most 768 pixels. Sampling includes endpoints and defaults to 12 uniformly spaced frame indices.
- The prompt covers task correctness/completion, hand and object visibility, observable contact, final state, occlusion, irrelevant footage, and severe obvious instruction violations. High irrelevant-footage scores are bad; other numeric scores are higher-is-better.
- Implement authentication, provider-specific image payloads, finite timeouts and bounded retries inside the adapter. Keep credentials outside `cache_id` and output metadata. Invalid JSON, missing/extra fields, strings masquerading as booleans, and out-of-range scores are rejected and routed to review.

To use an external VLM without writing an adapter, export a request and later replay its response:

```sh
python cli.py evaluate --video example.mp4 --task "put cup in drawer" \
  --export-vlm-request output/request.json --output output/pending.json
# Submit request.json using your provider tooling; save its JSON response.
python cli.py evaluate --video example.mp4 --task "put cup in drawer" \
  --semantic-json response.json --output output/evaluated.json
```

`--semantic-json` is an offline single-clip replay, not a model inference call. You are responsible for matching that response to the video and task. `examples/adapters.py` includes a directory-based replay provider for batch integration; its cache identity hashes response contents. No example pretends to detect hands.

## Technical metric definitions

All spatial metrics operate on a fixed 320×240 grayscale proxy. `technical.sample_fps` defaults to 2; every frame is decoded for duplicate and low-motion checks.

| Metric | Meaning |
| --- | --- |
| `blur_score` | Fraction of sampled frames above a Laplacian-variance threshold; higher is better |
| `exposure_score` | Fraction with neither dark nor bright pixel clipping above the threshold |
| `camera_stability_score` | Fraction of reliable sampled adjacent-frame phase-correlation translations within the allowed fraction of the image diagonal |
| `duplicate_frame_ratio`, `frozen_frame_ratio` | Fraction of adjacent decoded frame pairs with mean absolute grayscale difference below the configured threshold; these are the same MVP proxy |
| `low_motion_ratio` | Fraction of adjacent pairs below the separate low-motion difference threshold |
| `low_motion_periods` | Consecutive low-motion intervals and longest interval duration |
| `dropped_frame_ratio` | Estimated missing nominal intervals at timestamp gaps, divided by decoded plus estimated missing frames; `null` if timestamps are unavailable/nonmonotonic |
| `flags` | Duration/resolution/FPS, blur, exposure, freeze, low-motion, shake, timestamp-gap and readability checks |

FFmpeg performs a full decode with errors treated as failures. FFprobe/OpenCV check metadata, frame availability and reported frame count. A missing decoder executable or decoder timeout is an infrastructure problem, not evidence of corruption. OpenCV sampling uses presentation timestamps when available and nominal frame rate otherwise.

Visibility ratios are proportions of frames sampled at `visibility_sample_fps` (default 5 Hz, independent of technical sampling). Boundary ratio counts samples with any hand box inside the configured edge margin, with all samples as the denominator. `offscreen_periods` identifies runs where a hand **or** the active object was not detected, so these are candidate offscreen/occlusion periods rather than proof that an important interaction was missed.

## Scoring, review and payment

The five normalized components are:

| Component | Weight | Formula |
| --- | ---: | --- |
| Task | 30 | Mean of task-correct boolean, completion score, final-success boolean |
| Hand/interaction | 25 | Mean of VLM hand and interaction scores, capped by available detector hand and simultaneous visibility ratios |
| Object | 20 | VLM object score, capped by available detector object visibility |
| Technical | 15 | Mean of blur, exposure, `1−frozen_ratio`, resolution pass, FPS pass, duration pass |
| Framing | 10 | Mean of stability and `1−hand_boundary_ratio`; stability alone if boundaries unavailable |

Score = sum of weight × component, rounded to two decimals. Missing semantic components receive zero points, are explicitly `null`, and force review. Thus an evidence-incomplete score is a **conservative evidence score**, not an estimate of the collector's true quality. Unavailable technical evidence also receives no points. Do not rank incomplete evaluations as if they were fully evaluated clips.

Default numeric bands: `PASS >=85`, `REVIEW >=65`, otherwise `FAIL`. Additional review gates cover missing evidence, detector/VLM disagreement, unconfirmed final state, unavailable technical checks, candidate offscreen periods, and scores within 2 points of either decision threshold. Set `borderline_margin: 0` to disable the borderline override. A failed technical check prevents an otherwise high score from passing. A fully evidenced low score can still fail. Notes and reason codes explain overrides.

Hard rejects override the bands: corruption, wrong task, very low completeness, critical interaction not visible, no meaningful manipulation (near-zero completeness or almost entirely irrelevant footage), or severe protocol violation. Corruption forces score 0. Other hard rejects retain the weighted score for diagnosis. **Semantic hard rejects keep `decision=FAIL` but require human confirmation and hold payment.** VLM evidence never silently finalizes these payment rejections. The detector and deterministic metrics contribute independently and can prevent automatic acceptance.

Payment recommendations:

| Score | Fraction of base |
| --- | --- |
| 90+ | `bonus_multiplier`, default 1.0 (set e.g. 1.1 for a 10% bonus) |
| 75–<90 | 1.0 |
| 60–<75 | `reduced_multiplier`, default 0.5 |
| <60 | 0.0 |

Decision/review gates take precedence: any review/audit holds payment (`payment_status=held_for_review`, multiplier 0 as a hold sentinel, **not a finalized rejection**); confirmed automated `FAIL` recommends 0 with `rejected`. Consequently, some score-based payment tiers are unreachable without review under the default QC bands. Human records are stored separately; the MVP does not release payments or overwrite automated judgments after review.

Every numerical threshold, top-level component weight, audit percentage, and payment multiplier is configurable through YAML. Fields omitted from YAML use validated defaults in `qc/config.py`. Unknown config fields, nonfinite values, inverted ranges, and weights not summing to 100 are rejected. Durations in per-video metadata override corresponding global limits.

## Human audit and analytics

A deterministic hash lottery selects approximately `audit_percent` percent of PASS clips using the video ID and configured seed. Selection is reproducible across reruns; the configured percentage is an independent probability, not an exact batch quota. Selected clips retain `PASS` with `audit_selected=true`, `human_review_required=true`, and payment held. Change the seed to draw a different sample.

```sh
python cli.py audit-add --video-id VIDEO_SHA256 --reviewer-id reviewer-1 \
  --score 82 --decision REVIEW --task-correct true \
  --notes "Hand leaves frame during final placement" --output reviews.jsonl

python cli.py analytics --results output/results.jsonl \
  --reviews reviews.jsonl --output output/analytics.json
```

Choose an output suffix of `.csv` or `.jsonl` for flattened aggregate rows with a `section` field, or `.json` for the nested summary. Analytics also accepts a single evaluation `.json`. It exports collector averages/failure/review rates, task averages, reason counts, score and hand-visibility histograms (with missing counts), automated/human decision disagreement, score mean absolute error, and VLM/human task-correctness disagreement. Rates include their comparison counts and are `null` when no matching human evidence exists. These are rates on the reviewed subset, not unbiased estimates of the whole dataset. Latest human review by timezone-aware timestamp is used per video; the last automated JSONL record per content ID is used. Human review appends are intended for a single writer; use a database/service for concurrent reviewers.

## Cache and operational limits

`.qc-cache/{technical,visibility,semantic}/` stores validated stage results keyed by content hash, relevant configuration, metadata and adapter identity. VLM prompt/sampling changes invalidate semantic results; business-rule changes re-score cached evidence without repeating VLM calls. Invalid cache entries are recomputed. Adapter failures and corrupted/infrastructure technical results are not cached. Bump adapter `cache_id` whenever inference behavior changes. Bump `CACHE_VERSION` after metric-algorithm changes. Simultaneous processes may compute the same missing entry twice; writes are atomic, but this MVP has no distributed inference lock.

This is a runnable heuristic baseline, not a calibrated payment classifier. Blur varies with texture; exposure depends on the scene; phase correlation can miss rotation and confound foreground/intentional motion; static scenes resemble freezes; variable-frame-rate files can resemble dropped frames; codec concealment can hide damage. Sparse images cannot establish uninterrupted hand-object contact. Thresholds should be calibrated against labeled clips for each task/camera before automatic pay decisions. For stronger evidence, validate the MediaPipe adapter on egocentric footage, add temporal VLM windows, and compare against independent human ratings using the calibration commands. The hand-only adapter leaves independent object detection unavailable, so the existing acceptance safeguard still requires REVIEW unless another detector supplies that evidence.
