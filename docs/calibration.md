# Independent human calibration

## Export

```sh
python cli.py export-review --results results.jsonl --output review-bundle \
  --percent 20 --seed pilot-1
```

Sampling is a reproducible per-clip hash lottery across all decisions, not an exact quota. Use 100 percent for a complete handoff. Exports require an empty output directory to protect completed annotations. Newly generated results retain the original video path; exports validate that the file still matches its content hash before copying. If an old result has no source path, re-evaluate it with this version (existing evidence caches can be reused where compatible).

The output separates:

- `reviewer/`: full original clips, an index, blinded per-clip HTML, `human_reviews.csv` and `rating_guide.md`.
- `analyst/`: frozen automated results and export manifest.

**Share only `reviewer/` with independent annotators.** Blinded pages contain task instructions and ordered unannotated frames but no machine scores, boxes, warnings, rejection reasons or collector identity. Reviewers must watch the original clip; frame samples cannot establish completion or uninterrupted interaction. If the browser cannot play a codec, download the supplied original and use a compatible player.

## Annotate

Keep `video_id` and `evaluation_id` unchanged. Fill:

- `reviewer_id`, timezone-aware `timestamp` (ISO 8601), optional `notes`.
- Binary `task_correct` and hard failures: `wrong_task`, `task_incomplete`, `critical_interaction_missing`, `no_meaningful_manipulation`, `severe_protocol_violation`, `corrupted_video`.
- Integer 1–5 ratings: `completeness`, `hand_visibility`, `object_visibility`, `interaction_visibility`, `technical_quality`, `overall_usability`.
- `human_decision`: PASS / REVIEW / FAIL, independently chosen.

The bundled rubric defines 1 as absent/unusable, 3 as usable with reservations, and 5 as clear/complete. Hand-object contact must actually be observable; mere co-occurrence is insufficient. `wrong_task` must be the inverse of `task_correct`; PASS cannot coexist with a hard-failure flag. Uncertainty belongs in REVIEW and notes. Fully unfilled annotation rows are counted and skipped; partially filled or invalid ratings fail validation with their row number rather than silently disappearing.

Each reviewer can work in a separate CSV copy. Concatenate the data rows (one header) to combine them. The existing `audit-add` workflow for scalar 0–100 ratings remains available; the richer calibration command expects the new exported CSV schema.

## Analyze

```sh
python cli.py calibrate \
  --annotations review-bundle/reviewer/human_reviews.csv \
  --results review-bundle/analyst/automated_results.jsonl \
  --output calibration-output
```

Outputs:

- `calibration.json`: structured statistics and denominators.
- `calibration.txt`: readable equivalent.
- `paired_reviews.jsonl`: automated metrics/decision joined with human metrics/decision, reviewer identity and timestamp.

A stable evaluation hash identifies the actual evidence and configuration, excluding evaluation time and local file path. Stale/mismatched annotations are rejected. Repeated annotations by the same reviewer use their latest timestamp. Reviewers receive equal weight within an evaluation: numeric scores are averaged, decision uses strict majority (ties/no majority become REVIEW), and binary ties are excluded from that dimension. Comparisons use one consensus per evaluation rather than counting every reviewer as a separate clip. Do not combine multiple versions of the same video's evaluation when estimating dataset-wide independence.

## Interpretation

Human 1–5 ratings map linearly to 0–100 (`(rating−1)×25`); dimensions map to 0–1. Overall correlation includes Pearson and Spearman with tied ranks, plus MAE on 0–100. Undefined correlations for fewer than two observations or constant values are null. Missing evidence is never imputed as a measured dimension score; the legacy overall evidence score still includes its explicit zero-point policy, so evidence-coverage counts and complete-evidence MAE are reported separately.

Confusion matrix rows are human decisions, columns automated decisions. Error definitions:

| Statistic | Numerator | Denominator |
| --- | --- | --- |
| False acceptance | automated PASS, human FAIL | all human FAIL |
| False rejection | automated FAIL, human PASS | all human PASS |
| Unsafe pass fraction | automated PASS, human FAIL | all automated PASS |
| Review rate | automated REVIEW | all reviewed evaluations |

REVIEW is abstention, not an automatic acceptance or rejection. Zero denominators yield null, not zero error. The full matrix preserves cases such as automated PASS/human REVIEW that these strict error definitions do not count.

For each dimension, report support count, normalized MAE, signed bias (automated minus human), correlation, and fraction differing by at least one rating point (0.25). Task correctness is binary. Completeness/contact/object use semantic evidence; hand/object ratios cap semantic visibility when independent detection exists. Technical quality uses the technical component. Hard-failure disagreement is reported separately, excluding missing automated evidence and human ties.

Acceptance associations rank raw metrics and component scores by absolute Pearson correlation with binary human PASS versus non-PASS, retaining signed correlations and support counts. This is a descriptive point-biserial association, not a causal effect or an optimized weight. Correlations can be dominated by task, camera, collector, sample selection or small counts. Inspect disagreement clips, collect representative multi-reviewer labels, and validate proposed rules on a separate holdout set.

**No calibration command modifies weights, acceptance thresholds, payment tiers or prior automated decisions.** It produces evidence for a human decision about changes; it does not claim a calibrated probability of correctness.
