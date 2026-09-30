# Public-data integration testing

The EgoDex test runner exercises real videos with MediaPipe and local Qwen. It is
an integration and diagnostic run, not a calibrated accuracy benchmark.

## Reproduce

Install the project with `pip install -e '.[models,ollama,datasets,test]'`, install
FFmpeg, download the hand model with `python cli.py download-hand-model`, and
start a private Ollama server with `qwen3-vl:8b-instruct-q4_K_M` installed.

Download the official [EgoDex test archive](https://github.com/apple-aiml-research/ml-egodex#dataset-access-and-download)
to `data/egodex/test.zip`. The archive is approximately 17.3 GB in decimal units;
leave additional room for selected clips and reports. The dataset's CC-BY-NC-ND
terms are separate from the source-code license. Videos and model weights are
not included in this repository.

```sh
python -m scripts.egodex_smoke prepare \
  --archive data/egodex/test.zip --output data/egodex-sample \
  --tasks 10 --per-task 3 --seed ego-q-public-v1

python -m scripts.egodex_smoke run \
  --dataset data/egodex-sample --output output/egodex-smoke --controls

python cli.py export-review --results output/egodex-smoke/results.jsonl \
  --output output/egodex-review
```

Preparation requires an empty output directory. Selection is deterministic and
task-stratified, using paired MP4/HDF5 entries. Original video bytes are preserved;
the manifest records their SHA-256 hashes, archive member names, annotation hashes
and selection seed. Only language attributes are read from HDF5. No hand pose,
3D reconstruction or retargeting is performed. Reversible actions use the
dataset's direction selector; ambiguous/missing descriptions fail preparation.
Literal `"None"` values in optional description attributes are treated as absent.

The default run evaluates 30 clips across 10 tasks, followed by one deliberately
mismatched instruction per task (cooking an egg on a stove). These controls reuse
the original video bytes. Check the sampled tasks before using these controls:
the expectation of rejection would be invalid for footage actually showing egg
cooking. A changed selection seed may require a different control instruction.

## Outputs and interpretation

`index.html` links to annotated per-case reports. `results.jsonl` stores full QC
outputs; `metrics.csv` stores per-case scores, timing and detector/VLM differences.
`summary.json` separates original clips from mismatched-instruction controls.
`cases.json` records exact instructions. Evidence caches support restart without
repeating completed model calls; reports and summaries are regenerated on rerun.

EgoDex task descriptions are machine-generated, so agreement with them is **not**
human task-validation accuracy. No original clip is assumed to be a PASS. A
designed wrong-instruction control is not a naturally failed demonstration.
MediaPipe sampled hand presence and Qwen's interaction-conditioned visibility
also differ in meaning; their discrepancy flags inspection, not proven error.

Existing scoring/payment thresholds and missing-object-evidence review gates
remain unchanged. All recorded payment values are recommendations or holds, and
no money is transferred. Use the blinded review export to collect independent
ratings before reporting human agreement, false acceptance/rejection rates or
changing thresholds. Do not add machine judgments to the human annotation CSV.
