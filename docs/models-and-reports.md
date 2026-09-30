# Real models and inspectable evidence

## Install and run

```sh
pip install -e '.[models,test]'
python cli.py download-hand-model
python cli.py evaluate --video example.mp4 --task "put cup in drawer" \
  --expected-object cup --collector-id collector-17 \
  --detector mediapipe --provider openai --config config.yaml \
  --output output/clip.json
```

For hands without network inference, omit `--provider openai`. To evaluate folders, the same adapter/config flags work with `evaluate-folder`. `--hand-model PATH` and `--vlm-model MODEL_SNAPSHOT` override YAML values. Local `module:factory` plugins remain supported.

Dependencies remain optional: `.[hands]` installs MediaPipe, `.[vlm]` installs the OpenAI SDK, and `.[models]` installs both. The project uses one OpenCV distribution (`opencv-contrib-python`), also required by current MediaPipe. Use a fresh virtual environment when upgrading from the older headless-only installation to avoid overlapping `cv2` packages. Native codec/platform availability can vary.

The download command fetches Google's version-1 float16 hand-landmarker bundle and verifies SHA256 `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`. Evaluation never downloads weights implicitly. The adapter hashes loaded model bytes, MediaPipe version, and inference settings for cache identity.

## What the hand tracker measures

The CPU adapter uses MediaPipe Tasks `VIDEO` mode and actual sampled timestamps. Every uncached clip starts a fresh native tracker, and its resources close even after inference errors. Hand sampling defaults to **5 Hz**, separately configurable from technical sampling. Frame rows contain the exact decoded frame index, timestamp, `hand` detection flag, normalized boxes, per-hand confidence and confidence kind, handedness if available, local track ID, and normalized/pixel distance to the nearest image boundary.

**Confidence semantics:** MediaPipe's task result exposes the handedness-category score, not the underlying palm-detection/presence probability. The adapter records it as `confidence` with `confidence_kind="handedness_classification"`; `detection_confidence` remains null. The native detection, presence and tracking acceptance thresholds are recorded in configuration. The HTML label `side=...` refers to handedness confidence. These values must not be interpreted as calibrated probabilities that a hand is visible.

Boxes are clipped envelopes of the 2D landmarks, with configurable padding. No 3D landmarks or reconstruction are stored. `hands.swap_handedness` can correct a mirrored-camera label convention; verify this with your actual camera setup. Track IDs use greedy nearest-center matching within configured time/distance limits. They are local associations, not persistent identities, and may switch during crossings or occlusion.

| Metric | Definition |
| --- | --- |
| `hand_visible_ratio` | Sample fraction with at least one hand |
| `longest_hand_missing_interval` | Longest consecutive absence in seconds, using sample-and-hold intervals |
| `hand_missing_periods` | All measured hand-absence intervals |
| `manipulation_offscreen_ratio` | Fraction of sampled timeline with hand/object absence; **hand-absence proxy only** when object detection is unavailable |
| `offscreen_evidence_source` | Identifies whether the ratio uses hand-only or hand-and-object evidence |
| `hand_tracking_continuity_score` | Fraction of adjacent sample pairs sharing at least one track ID; missing pairs count as discontinuities |
| `boundary_distance` | Minimum normalized box-to-edge margin along the corresponding image axis |
| `boundary_distance_pixels` | Minimum edge margin in original image pixels |

Tracking continuity is null without IDs or with fewer than two samples. Missing capabilities remain null; empty detection lists mean absence. Missing hands may mean occlusion, cropping or model failure rather than confirmed offscreen manipulation. The hand-only adapter supplies no independent object detector. The original object-evidence acceptance safeguard is preserved, so this adapter plus VLM alone still requires review for automatic acceptance; semantic correctness and other dimensions remain measurable and calibratable.

MediaPipe is a practical local baseline, **not yet validated for this collection domain**. Motion blur, unusual egocentric viewpoints, gloves, small hands and object occlusion need validation against human ratings. See the [official Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python) for the native runtime and output contract.

## OpenAI semantic QC

Set `OPENAI_API_KEY` outside the repository. The adapter makes a real Responses API call; it is not an offline replay. It sends ordered JPEG frames with explicit timestamps, expected task description and optional object, but no collector identity, automated score, or payment information. Existing local JSON replay and alternative providers still work.

The default is the pinned snapshot `gpt-4.1-mini-2025-04-14`, temperature 0, high image detail, 1,500 output tokens, 60-second timeout and at most two SDK retries. These settings are configurable. Temperature can be null for models that do not accept it. Temperature 0 and a pinned snapshot reduce variation but do not guarantee bit-for-bit determinism or factual correctness. An account must have access to its configured model.

The request uses a strict JSON Schema and validates the response locally with `SemanticJudgment`. The original semantic fields remain, plus required `major_occlusion: boolean`. The prompt requests direct evidence of the task, completeness, contact, visible final state and occlusions, and instructs the model not to infer unseen actions. Major occlusion adds a review reason. Refusals, missing fields, out-of-range values, invalid booleans and incomplete responses produce unavailable semantic evidence and review rather than a fabricated score. Legacy stored `Semantic` records without this new field still load with `major_occlusion=null`.

Cache keys include video hash, task context (including expected object), prompt version, sampled-frame count and adapter identity containing model snapshot/generation settings. Changing collector ID, timeout or retry policy does not rerun cached semantic inference. Scoring/payment changes reuse evidence. `store=False` is sent to the API. No API keys are written to cache or reports. The SDK reads the key only on a cache miss that requires an actual call.

The implementation follows the official [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [image input](https://developers.openai.com/api/docs/guides/images-vision), and [GPT-4.1 mini snapshot](https://developers.openai.com/api/docs/models/gpt-4.1-mini) documentation. Live availability and real task accuracy require verification with your account and labeled clips.

## Visual QC and confidence

Single evaluations write `clip.report.html`; batches write one HTML file per result in the reports directory. Reports embed JPEGs, so the annotated report itself needs no server or external assets. They show decision, overall and component scores, reasons, semantic explanation, confidence limitations and chronological timestamps. Boxes correspond to exact saved detector frame indices. Technical warnings are recomputed on those same displayed frames to avoid attaching a nearby frame's blur/exposure measurement to the wrong image. Labels include HAND MISSING, BLUR, DARK, OVEREXPOSED, HAND AT EDGE and OFFSCREEN/OCCLUSION CANDIDATE.

`report_max_frames` bounds report size (default 60). Longer clips display uniform subsamples, so the report may not show every warning; complete sampled metrics remain in JSON. All metadata and model prose are HTML-escaped. Unreadable clips still produce a diagnostic report with no frames.

`evidence_confidence` records available evidence sources, missingness/disagreement limitations, and a conservative qualitative level. **`calibrated_probability` is null** until human data supports calibration. This is intentionally distinct from quality score, handedness confidence and VLM fluency. The next step is the [blinded human calibration workflow](calibration.md), not automatic payment-threshold tuning.

## Local Qwen with Ollama

Install `pip install -e '.[hands,ollama]'`, start Ollama, then explicitly download
`ollama pull qwen3-vl:8b-instruct-q4_K_M`. Evaluate through the existing plugin interface:

```bash
python cli.py evaluate --video example.mp4 --task 'put cup in drawer' \
  --detector mediapipe --provider qc.adapters.ollama_vlm:make_provider
```

`QC_OLLAMA_URL` defaults to `http://127.0.0.1:11434`; `QC_OLLAMA_MODEL` selects the
installed model tag. Keep remote Ollama bound to localhost and use SSH forwarding.
The adapter sends chronologically ordered images with timestamps, enforces the
semantic JSON schema, validates locally, and rejects truncated responses. Defaults
are temperature 0, seed 0, 16,384 context tokens and 1,500 output tokens; these reduce
variation without guaranteeing determinism. Programmatic `OllamaConfig` overrides
these settings. Cache identity includes the installed model's content digest,
Ollama version and inference settings, alongside the pipeline's video/task/prompt
identity. The server must be reachable when constructing the provider, even when
semantic evidence is already cached, to resolve its model identity.

Run `python -m scripts.benchmark_qwen --video NEGATIVE_CONTROL.avi --output output/qwen-benchmark`
to measure 2/6/12-frame inference, GPU memory, the full hand-plus-VLM pipeline and
cache reuse. This script deliberately asks for a cup-in-drawer task; supply a known
negative control. Its timings and task rejection are integration evidence, not
estimates of accuracy on real collector demonstrations. Existing conservative
acceptance and payment rules also apply to Qwen outputs.
