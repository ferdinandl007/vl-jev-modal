# VL Jev on Modal

Research code for typed decisions over text, images, image sequences, and video. It evaluates open multimodal models and trains the published Jev-Omni decision head on a mixed dataset. All model execution, dataset ingestion, feature extraction, training, and evaluation run in [Modal](https://modal.com/). The local CLI only submits jobs and displays small status results.

This repository contains **code and documentation only**. It contains no model weights, checkpoints, extracted features, training examples, images, videos, or credentials. Those artifacts stay in Modal Volumes or at their original sources.

## Current result

The `general-head-v2` run finished on Modal on 2026-09-30. It trained the 256-slot Jev-Omni decision head on 47,680 mixed training examples, including 6,787 videos, with a **frozen multimodal backbone**. Model code and weights were fetched on Modal from [`akhilaaa3/Jev-Omni`](https://huggingface.co/akhilaaa3/Jev-Omni) at revision `5addda86ddee081a68fb067477ea100c221b8917`. This is supervised head training, not full-backbone fine-tuning or a new pretrained VLM.

| Held-out v2 test set | Correct / total | Accuracy |
| --- | ---: | ---: |
| All decisions | 7,983 / 9,284 | 85.99% |
| Vision decisions | 5,677 / 6,754 | 84.05% |
| Image decisions | 2,361 / 2,827 | 83.52% |
| Video decisions | 3,214 / 3,782 | 84.98% |
| Text decisions | 2,306 / 2,530 | 91.15% |

The pooled vision result exceeds the 83% target. The equal-family vision average is only **72.96%**, however, and GUI next-click accuracy is **30.36%** (34/112). This test mix is dominated by available labeled datasets and does not establish 83% on every visual task or real-world inputs. The text-retention guard passed on development data. These are within-dataset closed-choice results, not a comparison to JevBench, MVBench, GPT models, or a deployed system. The selected `uniform` checkpoint has SHA-256 `e7771cb40f7cc41b5847e2a4880fb498572c86128c5d20cd0cce4f880211f780`; it remains in the owner's Modal Volume and is not published here.

The earlier `general-head-v1` run trained on 31,112 examples and scored 82.81% vision (2,148/2,594) and 90.95% text (2,301/2,530) on its held-out test set. The v2 test composition differs substantially, so the two pooled vision percentages are not a like-for-like measure of improvement.

## Repository map

- [`experiments/modal_jev_omni_sports_train.py`](experiments/modal_jev_omni_sports_train.py): frozen-feature extraction, decision-head training, dev selection, held-out evaluation, and checkpoint validation.
- [`experiments/modal_general_v2_pipeline.py`](experiments/modal_general_v2_pipeline.py): detached Modal coordinator for v2 merge, packed feature extraction, training, and evaluation.
- [`experiments/dataset/`](experiments/dataset/): pinned source builders, media materializers, split checks, and dataset audits.
- [`experiments/modal_status_probe.py`](experiments/modal_status_probe.py): compact read-only status from Modal Volumes.
- Other `experiments/modal_*.py` files: model, decision-contract, and video probes.
- [`docs/DATASET_PILOT.md`](docs/DATASET_PILOT.md), [`docs/CONFIDENCE.md`](docs/CONFIDENCE.md), and [`docs/TRAINING_AND_RL_PLAN.md`](docs/TRAINING_AND_RL_PLAN.md): dataset provenance and research plans. Plans are not claims of completed experiments.

## Run on Modal

Install and authenticate the [Modal CLI](https://modal.com/docs/guide) and run commands from this repository's root. Use your own Modal workspace and budget. Source repositories are pinned in the builders; some require access or may have changed availability. The scripts create and use named Modal Volumes in that workspace.

```sh
# Inspect source availability and dataset state.
modal run -m experiments.dataset.modal_build_general_v1 --mode status
modal run -m experiments.dataset.modal_build_ucf101_full --mode status
modal run -m experiments.dataset.modal_merge_general_v2 --mode status

# When the documented source datasets and materialized media are ready,
# run the v2 pipeline as a detached Modal job.
modal run --detach -m experiments.modal_general_v2_pipeline

# Read compact results without fetching checkpoints or media locally.
modal run -m experiments.modal_status_probe
```

The v2 coordinator expects the v1 general dataset, human-action dataset, and all 13,320 UCF101 videos to be materialized first. The individual builders in `experiments/dataset/` perform those steps; inspect their `main()` modes before launching a full build. `modal run --detach` keeps the remote function active after the submitting terminal disconnects. Do not use `modal volume get` if you want all artifacts to remain in Modal.

## Data and model rights

The **MIT license applies only to code and documentation in this repository**. It does not grant rights to third-party datasets, footage, images, model weights, or model code. The mixed dataset is marked research-only because source terms and underlying media rights differ; review each source before redistribution or commercial training. In particular, no fetched third-party media or generated labels are published here. See [dataset notes](docs/DATASET_PILOT.md).

Jev-Omni is an independent upstream model. This project is not affiliated with TypeSafe AI and does not reproduce TypeSafe's unpublished model or training method. Closed-choice logits and confidence concentration are not calibrated probabilities without a held-out calibration study.
