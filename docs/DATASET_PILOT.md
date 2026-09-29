# VL Jev mixed decision pilot v1

This is a **metadata-first research training set**, built for a small continued-training experiment on a pretrained vision-language model. It is not a general visual pretraining corpus, a complete media download, or a calibrated model. A Modal function runs `build_pilot.py` and writes split JSONL files, `sources.lock.json`, and `stats.json` to `pilot_v1/` in the **`vl-jev-pilot-data` Modal Volume**. Every source is pinned to a Hugging Face or upstream GitHub revision. Materialized images and video live in the separate **`vl-jev-pilot-media` Modal Volume**. No dataset processing or media download is required on the workstation.

## What a row means

Each row has a `state` (empty for most visual observations), one typed `question`, a `gold.key`, and source/media provenance. The question follows the three types accepted by `experiments/modal_jev_contract.py`:

- `choice`: named, closed-list `criteria` with a gold option key.
- `noul`: a true/false decision. The gold key is `true` or `false`; the model should eventually report `P(true)`.
- `score`: 2–10 ordered criteria. The gold key is the correct level index; a model should report the full level distribution and expected score.

There are **no null gold labels**. “Unknown” or abstention is not yet a validated training class; it needs deliberately selected ambiguous media with human adjudication. The JSONL does not contain fitted probabilities. A one-hot gold label is an outcome observation, not a claimed confidence value.

The builder deterministically shuffles Choice options per row so a particular letter does not reveal the answer. A trainer may reshuffle the training options again each epoch while keeping the key/meaning mapping intact. Preserve exactly the supplied option names at inference. Do not turn Score into a binary success label or interpret its expected value as a probability of correctness.

## Coverage

| Area | Source and conversion | Label quality / media |
| --- | --- | --- |
| Football events | Soccer Events goal, yellow-card, red-card clips → Choice | Publisher labels; direct MP4 |
| Basketball | BARD one-shot clips → Noul shot-made and Score points (0–3) | Captions converted to weak labels; direct MP4 |
| Horse racing and other sports/actions | UCF101 HorseRace/HorseRiding and 14 other actions → Choice | Original action labels; two large tar shards; separate groups by source video |
| Specific horse race outcome | Wikimedia Belmont Stakes clip → Choice | One public-domain-marked QA example, not enough for a horse-racing outcome capability |
| Driving | Automingo five-frame scenes → Noul or Choice | Publisher answers; selected Parquet rows |
| Generic images | PixMo Count → Choice or Score | Object-count labels; external Flickr image URL and hash |
| General video | Prelinger Moments short event segments → Choice among caption-derived alternatives | Public-domain-marked footage, generated weak captions |
| Visual tool choice | VC-Tooler-SFT image, user prompt, available tool names → Choice | Synthetic trajectories; selected Parquet rows |
| Text-only preservation | SNLI inference → Choice/Noul, original Banking77 intents → Choice, SST-5 sentiment → Score | Source labels; no media |

The question mix is intentionally heterogeneous. That tests whether a pretrained backbone can learn a typed decision interface across modalities. It does **not** establish that one small adapter can master every listed skill. We will compare each task family separately and with a pooled aggregate.

## Splits and leakage

The builder uses deterministic source groups: football match, basketball game, driving scene, UCF101 source-video group, archive film, image hash, or text instance/premise. Official source test rows remain test rows where available; other groups are assigned to train, development, calibration, or test. A duplicate group is kept only in its most protected split. `stats.json` shows exact counts after this filter.

Use `dev` for model/epoch choice, `calibration` only for fitting probabilities or a decision threshold, and `test` once for final comparison. Do not train on the `test` JSONL. For the actual evaluation, verify image and video perceptual duplicates across sources too; the current script catches exact source-group reuse but cannot identify copied media across repositories.

Banking77 is deliberately sampled across all 77 intents, and basketball points are sampled across the four possible outcomes. These are pilot distributions, not natural deployment frequencies. A deployed probability needs a fresh calibration set drawn from the actual use case, or an explicit prior correction.

## Calibration and small-run acceptance

The first comparison should be frozen pretrained model versus a small supervised adapter/head trained on `train`, with text-only replay. Save the complete per-option logits. Fit a temperature (and, if justified, a prior adjustment) on `calibration` **after** the model is chosen on `dev`. On `test`, report accuracy and log loss/Brier score by type and source; for Choice also top-answer calibration, and for Noul the true-event reliability curve. For Score, report mean absolute error and a proper distribution score such as ranked probability score. Show bin counts and uncertainty intervals: small bins cannot support a precise “90% confidence” claim. Compare text-only results before and after training to detect forgetting.

The initial result should be called an improvement only when held-out decision quality and probability quality improve on the intended task without a material drop in text decisions. Supervised learning with a proper scoring loss is the first experiment; this dataset alone does not reproduce TypeSafe's unpublished RLCD method.

## Rights and quality gates

Treat the complete pilot as **research only**. The football source expressly restricts use to non-commercial research. BARD says CC BY 4.0, but that does not settle NBA broadcast rights. UCF101 and external Flickr images likewise need footage/image rights review. VC-Tooler-SFT is CC BY-NC 4.0. Prelinger and the selected Wikimedia horse clip carry public-domain markings, but confirm the exact source item before distribution. SST-5 does not declare a license in its Hugging Face metadata. Do not redistribute fetched media or use this mixed set for a commercial model until each source is cleared.

Before a training run, materialize only selected rows on Modal, verify the fetched bytes and media lengths, check that the event is visible in sampled frames, then human-review a stratified subset of weak labels and every evaluation example. PixMo's `image_sha256` does **not** equal the SHA-256 of the sampled URL's downloaded bytes; the materializer records the actual downloaded hash separately. Captions can be wrong, outside the chosen time segment, or solvable from a watermark/scoreboard instead of the video action. The UCF101 tar member path resolver has been checked on one HorseRace clip; further rows still need spot checks. This is why the artifact is a reproducible **candidate set**, not yet a certified training corpus.

## Build

```sh
modal run experiments/dataset/modal_build_pilot.py
modal run experiments/dataset/modal_materialize_pilot.py --smoke
```

These commands only dispatch Modal functions. The builder itself refuses to run outside the mounted Modal data Volume. It checks that source repositories still match pinned revisions, then reads annotations via pinned files or Hugging Face Dataset Viewer APIs. Banking77 comes from its [original 77-intent CSVs](https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data), because sparse viewer pages missed most intents. It refuses a changed repository revision instead of silently using new rows. Source-level checkpoints are in `pilot_v1/.build_cache/` in `vl-jev-pilot-data`; clear a source cache in that Volume after changing its conversion logic.

The materializer reads the data Volume and writes selected media to `vl-jev-pilot-media`. The smoke run verified direct sports MP4, a cropped archive clip, a five-image driving scene, a generic image, and an embedded horse-racing QA video. A separate one-row run verified a UCF101 HorseRace clip from its large tar shards. UCF101 extraction requires `--include-shards` and streams about 3.5 GB of source tar data; batch several UCF rows in one Modal call.

After an inference run, save one JSON line per dataset row as `{"id":"...","logits":{"A":0.4,"B":-0.2}}` (the keys depend on the question). Then fit and inspect temperature scaling using only the calibration split:

```sh
modal run experiments/dataset/modal_evaluate_pilot.py \
  --calibration-file predictions/calibration.jsonl \
  --test-file predictions/test.jsonl \
  --output-file reports/calibration.json
```

All three paths above are relative to `vl-jev-pilot-data` and are read or written inside Modal. A synthetic-logit smoke check of the evaluator passed; it does not represent a trained model result.

The report checks exact ID/option coverage, shows before/after log loss and Brier score, reliability bins with sample counts, and Score error/distribution metrics. It cannot make an unreviewed label correct or prove tight calibration from a small bin.

## Source pages

- [Soccer Events](https://huggingface.co/datasets/infactory-ai/soccer-events), [BARD](https://huggingface.co/datasets/GabrieleGiudici/BARD), [UCF101](https://huggingface.co/datasets/guyuchao/UCF101)
- [Automingo](https://huggingface.co/datasets/ibarcelo/Automingo_dataset), [PixMo Count](https://huggingface.co/datasets/allenai/pixmo-count), [Prelinger Moments](https://huggingface.co/datasets/davanstrien/prelinger-moments)
- [VC-Tooler-SFT](https://huggingface.co/datasets/5551z/VC-Tooler-SFT), [Wikimedia video QA](https://huggingface.co/datasets/Reubencf/Adaption-video-qa-diverse-topics)
- [SNLI](https://huggingface.co/datasets/stanfordnlp/snli), [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets), [SST-5](https://huggingface.co/datasets/SetFit/sst5)
- [VSTAT](https://huggingface.co/datasets/nyu-visionx/vstat) is a valuable horse-racing and temporal-video **test** benchmark; keep it out of training.
