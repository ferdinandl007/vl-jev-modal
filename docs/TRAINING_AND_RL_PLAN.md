# General multimodal decision training and action learning

28 September 2026. All dataset/media/model access, inference, and training in
this project must run on Modal. Local files here are orchestration code and
reports only. The working data and checkpoints are private Modal Volumes.

## What is running

The pinned base is `akhilaaa3/Jev-Omni` at
`5addda86ddee081a68fb067477ea100c221b8917`. The completed sports
experiment trained its 256-slot decision head with a frozen 12B multimodal
backbone; it was a small baseline, not proof of a general gain. The current
general build adds all usable training rows from A-OKVQA and ScienceQA, all
materializable rows from the existing mixed Jev pilot, and Webintosh
screenshot-to-click choices. The builder retains upstream validation/test
splits and removes train rows colliding with a held-out source group, exact
decision, or image hash. It records all exclusions and split hashes.

| Source | Train inventory | Task | Split and caveat |
| --- | ---: | --- | --- |
| [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) | 17,056 raw | Image commonsense choice | Labeled validation; official test labels withheld. |
| [ScienceQA](https://huggingface.co/datasets/Gisiyuan/ScienceQA) | 12,726 raw | Image and text science choice | Official validation/test; original ScienceQA terms are noncommercial. |
| [Webintosh](https://huggingface.co/datasets/Chengheng/Webintosh) | 2,343 constructed choices | Screenshot to next-click region | Source trajectories separated among train/validation/test. Labels are action targets, but this is an offline single-step proxy. |
| Existing pilot | 2,877 rows across all splits | Video, images, text, sports, driving questions | Visual rows require verified media; some weak labels are unreviewed. |

The A-OKVQA viewer skips a few invalid or duplicate options. The official
unlabeled A-OKVQA test partition is not a scored test. The verified v1
manifest has **31,112 train, 5,694 dev, 525 calibration, and 5,124 test**
decisions. Of its train rows, 24,154 are images, 6,721 text, 209 video, and 28
image sequences. Full counts and split hashes are in
`vl-jev-general-v1-data/v1/summary.json`.

ScienceQA exposes no problem ID; many questions reuse the same wording with
different images and choices. Its group key uses official split plus row index,
and the merge still excludes train rows with held-out identical image hashes
or exact decisions. The final manifest excluded 1,563 train rows on image
overlap; it also counted 439 exact-decision collisions, which can overlap the
same rows. The earlier question-text group policy incorrectly discarded over
7,000 source-training rows and has been replaced before feature extraction.

## Sports and human-action expansion

The next frozen manifest, `v2`, adds source-labeled action decisions to the
general set. `experiments/dataset/modal_merge_general_v2.py` merges these after
both media jobs finish; it removes the small UCF101 pilot sample in favor of the
full source and keeps original-video/image groups out of weaker splits.

| Source | Verified source inventory | Treatment |
| --- | ---: | --- |
| [UCF101](https://huggingface.co/datasets/guyuchao/UCF101) | 13,320 videos, 101 classes | Pinned revision `057753e5d0709d3f5b8104a803b91a420a069103`; 6,729 grouped train, 1,379 dev, 1,516 calibration, 3,696 official test; eight-way related-action choices. Media materialization on Modal is in progress. |
| [Human Action Recognition](https://huggingface.co/datasets/Bingsu/Human_Action_Recognition) | 12,600 genuinely labeled source-train images | Pinned revision `6c1a1284eb3557055d7c57b91cd7e68e3252b32c`; 111 exact duplicates collapsed and 31 near-duplicate pairs grouped, yielding 9,990 train / 1,239 dev / 634 calibration / 626 internally held-out test. Its 5,400 nominal source-test labels are all placeholder class 0 and never scored or trained on. Full Modal image build completed. |
| [FineGym skeleton](https://huggingface.co/datasets/Lozumi/FineGym-skeleton) | Zero admitted rows | The pinned annotation file returned HTTP 401 from Modal despite public metadata. Do not count it as training data. |

For UCF101, the source class is the choice target and curated related classes
form alternatives; the same class/video label is not a separate visual
adjudication. For the static human images, source labels also need a stratified
visual review before claiming high-quality held-out scores. The uploader's
ODbL declaration does not independently clear upstream Kaggle/DPhi images for
commercial use. Source images and video remain on Modal.

## Training-method assessment

The current frozen-feature/head experiment is the cheapest meaningful control,
but it is **not** an established optimum. Jev-Omni's reference loader takes one
last-token hidden state and maps it through a 256-position linear head. The
visual and language backbone receives no gradients here. This cannot teach
new visual features, and position logits can be sensitive to option order.
The earlier nine-football-clip probe selected the last option in all 27
permuted calls; the 72-row sports head gain was 72.9% to 75.0% on 48 dev
permutations, with unreviewed source labels. Those results justify testing
order robustness and visual label quality explicitly.

Run the pinned frozen-head baseline on the full manifest, then compare these
controlled alternatives on the **same** input revision and dev sets:

1. Uniform full-pass head training versus square-root inverse-source loss
   weighting. Each still consumes every train row per epoch. Report micro and
   equal-family vision accuracy, log loss, Brier score, and per-source counts.
2. Option-order augmentation or an option-conditioned scalar scorer, with
   untouched choice-order permutations. This addresses the slot-bias failure.
3. A small multimodal LoRA pilot on the Gemma backbone plus head, with the
   same train/dev examples and compute accounting. Test vision-language
   projection and selected language attention modules first; verify a visual
   perturbation changes the loss. Full 30k LoRA training is justified only
   after the pilot beats head-only on held-out vision tasks without damaging
   text tasks.
4. Keep text decision rehearsal during visual adaptation. Select with a text
   regression gate and group-macro vision score, not aggregate accuracy alone.
   Use calibration rows only after model selection. Score the untouched test
   once, with source, family, modality, option-order, and confidence results.

The user's 83% is a target for general held-out vision accuracy, not a measured
result or a football-only target. Publish the exact denominator and task mix.
The model's original card reports 86.15% group accuracy on JevBench's matched
231 public decisions and 53.10% on its MVBench evaluation, using different
metrics and protocols. An independent public JevBench run reports GPT-5.6 Luna
at 206/231 (89.18%) and Jev 1.13.0 at 200/231 (86.58%). These numbers cannot
be compared to the current mixed-dataset accuracy or treated as this
fine-tuned checkpoint's score. Run the same pinned JevBench public subset
before/after; never train on those tasks.

Sources: [Jev-Omni model card](https://huggingface.co/akhilaaa3/Jev-Omni),
[Open-Jev JevBench audit](https://github.com/Zefan-Cai/Open-Jev/blob/main/docs/jevbench-public.md),
[upstream JevBench](https://github.com/fstandhartinger/jevbench).

## Driving and computer-use extensions

- [URJC CARLA expert racing](https://huggingface.co/datasets/urjc-deepracer/carla-expert-racing)
  is an Apache-2.0 simulator dataset with front camera, steering, braking,
  speed, maneuver, and experiment IDs. Audit whether experiments are disjoint
  across its published splits before adding a discrete left/straight/right/
  brake decision task. Neighboring video frames must stay in one split.
- [CARLA autopilot images](https://huggingface.co/datasets/immanuelpeter/carla-autopilot-images)
  has 56.2k train / 4.8k validation / 7.2k test frames, a run-level split,
  controls and telemetry, but the dataset is about 189 GB. A modest, documented
  source sample is a feasibility study; claiming full-source training would
  require all usable train rows.
- [DrivingVQA](https://huggingface.co/datasets/WaltonFuture/DrivingVQA)
  has 3,142 train and 789 test image questions including turn and right-of-way
  choices. Its repository does not declare a license in its metadata; verify
  rights and image/group overlap before mixing it into training.
- [Webintosh](https://huggingface.co/datasets/Chengheng/Webintosh) supplies
  click targets in screenshots with trajectory-disjoint splits. The current
  constructed task scores one next-click choice among four regions. It does
  not establish multi-step task success. The card says its element
  instructions were produced with GPT-5.6 Luna, so this dataset is not an
  independent clean comparison to Luna.
- [WebSTAR](https://huggingface.co/datasets/microsoft/WebSTAR) and
  [ProCUA-SFT](https://huggingface.co/datasets/nvidia/ProCUA-SFT) have
  screenshot-action trajectories for later sequential training. Their large
  media archives need selective Modal-only ingestion and a separate run.

## RL problem

Static labeled multiple-choice rows already have the correct target and a
differentiable cross-entropy or Brier objective. Sampling answers and assigning
reward 1 for a correct label adds variance without an environment benefit.
Proper scoring rules can improve probability behavior, but calling that
TypeSafe's undisclosed RLCD would be unjustified.

The actual RL extension is a **constrained sequential policy**:

1. On each screenshot or driving observation, generate a finite, valid action
   set. For GUI: visible clickable elements plus scroll/type/stop. For CARLA:
   lane/route-conditioned steering/braking actions. Jev-Omni scores that set;
   a deterministic executor carries out the chosen action. Coordinates and
   actions must be validated against the current observation.
2. Begin with behavior cloning on demonstration trajectories, using the source
   split by task/trajectory or driving run. Evaluate offline action accuracy
   and actual environment success separately.
3. Run episodes in a Modal-hosted simulator or browser environment. BrowserGym
   exposes observations, actions, and per-step reward; OSWorld has
   execution-based desktop success checks. CARLA gives route progress,
   collision, traffic-rule, and completion outcomes. Keep test scenarios
   untouched.
4. Optimize expected episode return with an on-policy policy-gradient method,
   a value baseline, and a KL penalty to the supervised reference policy.
   Reward verified task completion/route progress; penalize collisions,
   illegal/invalid actions and excessive steps. Preserve text decision
   rehearsal and compare pre/post text benchmarks. Recalibrate decision
   probabilities on held-out data after policy training.

No RL job is active yet. The browser/desktop and CARLA environment adapters,
outcome verifiers, and baseline success measurements must precede one.
Sources: [BrowserGym](https://github.com/ServiceNow/BrowserGym),
[OSWorld](https://os-world.github.io/),
[TRL vision GRPO guidance](https://huggingface.co/docs/trl/grpo_trainer).

## Modal commands

```sh
modal run experiments/dataset/modal_build_general_v1.py --mode status
modal run experiments/dataset/modal_build_general_v1.py --mode finalize
modal run experiments/modal_jev_omni_sports_train.py --mode general-source-smoke
modal run experiments/modal_jev_omni_sports_train.py --mode general-extract --run-name general-head-v1 --shards 4
modal run experiments/modal_jev_omni_sports_train.py --mode general-train --run-name general-head-v1 --epochs 3 --weighting uniform
modal run experiments/modal_jev_omni_sports_train.py --mode general-train --run-name general-head-v1 --epochs 3 --weighting sqrt_inverse_source
modal run experiments/dataset/modal_merge_general_v2.py --mode status
```

After choosing one checkpoint on development results, extract
`calibration,test` features with `--mode general-extract --splits
calibration,test` and score the held-out split with `--mode general-evaluate
--splits test --weighting CHOSEN`. Do not interpret the train count as an
accuracy measurement.
