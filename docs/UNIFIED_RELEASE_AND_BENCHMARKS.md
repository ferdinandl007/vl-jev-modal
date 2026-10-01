# Unified model and benchmark audit — 2026-09-30

All dataset ingestion, media handling, feature extraction, training, inference,
and saved results run on Modal. This repository contains code and documentation.
The complete inventory is in Modal Volume `vl-jev-benchmark-audit`,
`inventory.json`; source split counts below were read from the actual manifests.

## Actual training coverage

The public v3 head used 47,680 v2 examples and 7,962 GUI examples. The unified v4
candidate continues that single head with **all 55,642 old examples plus 3,358
visual-category SoccerNet examples: 59,000 total**. It does not combine separately
trained head weights or select a different model per task. The upstream merged
Gemma 4 12B backbone remains frozen in this post-training stage. Its upstream
decision configuration already lists merged LoRA training; it is not the raw
Google instruction checkpoint.

| Source in current mixture | Training rows | What it supports and what remains uncertain |
| --- | ---: | --- |
| A-OKVQA | 17,020 | Image multiple choice with commonsense; not all answers are supported by pixels alone. |
| ScienceQA | 10,902 | Mixed text/diagram/image science questions; knowledge tasks remain retention data, not the visual-evidence primary score. |
| Human Action Recognition | 9,990 | Static action classification. Internally held-out source-train images; 5,400 placeholder-labeled source-test images were excluded. |
| UCF101 | 6,725 | Videos with original action labels and related-action alternatives. Includes basketball, cricket and tennis classes; not official 101-way action-recognition scoring. |
| Multimodal Mind2Web | 5,253 | Screenshot, goal/history and four named controls. Target selection, not full-page retrieval, value generation or task completion. |
| BrowserGym MiniWoB action-only trajectories | 2,709 | Accessibility-tree click selection with teacher-generated labels. |
| SoccerNet SN-VQA-2026 direct visual categories | 3,358 | Action, game state, camera class/switch, replay, jersey color, score/time and multi-view foul questions. Categories alone do not certify every sampled clip contains sufficient evidence. |
| Webintosh | 2,343 | Screenshot to click-region choices. Offline derived control-selection task. |
| Banking77, SNLI, SST-5 | 474 | Text intent, inference and ordered sentiment retention. |
| PixMo Count | 95 | Static object counting with source labels. |
| Prelinger Moments | 61 | Caption-derived video alternatives; weak labels need visual review. |
| VC-Tooler | 41 | Image/tool-selection supervision from synthetic trajectories. |
| Automingo | 28 | Ordered five-image driving scenes and publisher answers. Preserve scene order. |
| Wikimedia race example | 1 | One race-outcome example; cannot establish an outcome-prediction capability. |

The earlier pilot also contains Soccer Events and BARD basketball clips. These
contribute 23 and 47 rows respectively to the v2 test, **zero v2 training rows**.
BARD's caption-derived shot/points labels are weak. FineGym skeleton contributes
zero admitted rows because the pinned annotation file was inaccessible. There
is no admitted dedicated ice-hockey decision corpus; UCF101 field hockey is a
different sport. Dedicated subtle cricket/tennis decisions are also not yet a
verified capability. The superseded v1 manifest had 203 media hashes across
splits; v2 has zero recorded source-group or exact media-hash collisions.

Mixed source permissions remain research-only. Dataset metadata licenses do not
independently clear all underlying media. The release redistributes model
artifacts and aggregate measurements, not datasets, labels or footage.

## Selection and actual current results

The audit found three exact marked-screenshot hashes shared between Mind2Web
train and development. Their questions, option lists and answers also match;
goals/history can differ. V4 excludes all three development rows, leaving
2,105 GUI development examples. Two training screenshots also occur in the
official test-task source split; the clean primary GUI diagnostic therefore
uses 1,084 of 1,086 derived decisions. Original v3 results remain historical and
are not retroactively renamed leakage-free.

V4 selects on the equally weighted general visual-family average, GUI-source
average and SoccerNet visual-family average. Gates require improved SoccerNet
development accuracy, text/vision/GUI accuracy within one percentage point of
v3, and general log loss no more than 0.03 worse. Tests and external benchmarks
are first read after development selection. Epoch 3 passed. Checkpoint SHA-256:
`e0bfd32690d4656238e29e402b01772ca93e97602c12875b84f61793eef169c6`.

| Measurement | v3 | Unified v4 |
| --- | ---: | ---: |
| General development | 6,868/7,813 = 87.90% | 6,860/7,813 = 87.80% |
| GUI development, screenshot overlap excluded | 1,663/2,105 = 79.00% | 1,672/2,105 = 79.43% |
| Labeled visual SoccerNet development | 414/847 = 48.88% | 419/847 = 49.47% |
| Unchanged internal general test | 8,051/9,284 = 86.72% | 8,051/9,284 = 86.72% |
| Usable SoccerNet public test, all categories | 235/496 = 47.38% | 237/496 = 47.78% |
| Clean derived Mind2Web test, screenshot overlap excluded | 852/1,084 = 78.60% | 858/1,084 = 79.15% |

The visual development gain is five answers on a set used for epoch selection;
it is an optimistic diagnostic, not a statistically established gain. Public
SoccerNet test categories are absent, so its visual-modality subtotal must not
be called a verified visual-only category score. Four malformed source-test
answer keys are excluded from our 496 denominator; this is not an exact official
500-row challenge submission. The separate all-category SoccerNet head's 59.7%
result includes knowledge tasks and is not this unified checkpoint.

## External benchmark protocol

### Completed measurements and release status

The full pinned source splits completed with all 5,934 questions per full-scope
model/representation run. The following results are **micro accuracy under
the declared fast decision protocol**, not native-generation leaderboard scores.

| Model, ordered frames | BLINK validation (1,901) | TempCompass MC (1,580) | TempCompass yes/no (2,453) |
| --- | ---: | ---: | ---: |
| Upstream Jev-Omni | 995 = 52.34% | 1,075 = 68.04% | 1,722 = 70.20% |
| General v3 | 1,028 = 54.08% | 1,081 = 68.42% | 1,751 = 71.38% |
| Unified v4 | 1,026 = 53.97% | 1,082 = 68.48% | 1,745 = 71.14% |
| Google Gemma 4 12B IT | 1,162 = 61.13% | 1,065 = 67.41% | 1,773 = 72.28% |
| Qwen3.5-4B | 1,196 = 62.91% | 1,055 = 66.77% | 1,751 = 71.38% |
| Qwen3.5-9B | 1,253 = 65.91% | 1,132 = 71.65% | 1,842 = 75.09% |

Decider-2B scored 481/793 = 60.66% on the eligible single-image BLINK subset.
That is a different denominator from the full BLINK table; a direct ranking
requires all models' scores on those same IDs. The paired comparison job did
not finish with a verified result, so no bootstrap significance claim is made.

| Unified v4 representation | TempCompass MC | TempCompass yes/no |
| --- | ---: | ---: |
| Sixteen ordered midpoint images | 1,082/1,580 = 68.48% | 1,745/2,453 = 71.14% |
| Same images with actual timestamps | 1,094/1,580 = 69.24% | 1,742/2,453 = 71.02% |
| Native video processor, sixteen frames / 70 soft tokens | 1,051/1,580 = 66.52% | 1,673/2,453 = 68.20% |

Native video changes sampling/token budgets and includes redundant initial
decoding in this harness. It did not improve this checkpoint's accuracy.
An MP4 container itself does not add information. Original time/order should
be retained for continuous sequences; separate camera views should not be
assigned invented temporal continuity.

Measured v4 median request latency was 111 ms on BLINK and about 1,102 ms
on the sixteen-frame temporal tasks, including preprocessing and scoring.
These do not establish single-digit-millisecond fresh-media decisions.
The fully conditioned head is much cheaper, but changing the question or
options requires a new backbone forward in the current implementation.

On the pinned 231 public JevBench tasks, upstream/v3/v4 scored 197/206/205
correct: 85.28% / 89.18% / 88.74% micro accuracy. V4's group macro was 86.67%;
Choice 123/139, Noul 68/74, Score exact class 14/18. Original source scoring,
label order and native typed prompts were used. This is not the current
sealed composite, and the eighteen Score questions provide limited evidence.

The [complete unified v4 package](https://huggingface.co/ferdinandl007/jev-omni-unified-v4)
is public, containing 23,919,549,408 bytes of backbone weights plus the
selected head and typed adapter. Anonymous complete-model loading and all
three typed output structures were verified on Modal at revision
`7a86292f3b72ea29b8a96d9332e4ef2011d0c8f3`. Head SHA-256:
`e0bfd32690d4656238e29e402b01772ca93e97602c12875b84f61793eef169c6`.
A temporary Modal workspace-disabled error interrupted the run; access
recovered, aggregation and paired analysis completed, and publication plus
anonymous verification succeeded. This release contains v4 training;
it does not yet include the 117,330-example text expansion.

The text expansion's 117,330 training examples and all feature packs completed.
The subsequent 176,330-example candidate training has no verified completion
report. It does not inherit v4's external scores. Training resumed from the
saved features after repairing repeated MultiNLI source-ID handling.
No local dataset/model fallback is authorized.

`experiments/modal_official_benchmarks.py` materializes source labels and media
from pinned revisions. Benchmark datasets are evaluation-only and have no
synthetic labels. The initial complete standard-source suite is:

| Benchmark | Exact scope | Purpose |
| --- | --- | --- |
| [BLINK](https://huggingface.co/datasets/BLINK-Benchmark/BLINK) | All 1,901 labeled validation questions, all 14 tasks | Spatial relations, depth, counting, correspondence and multi-view visual reasoning. Validation, not hidden test. |
| [TempCompass](https://github.com/llyx97/TempCompass) MC | All 1,580 multiple-choice test questions | Action, direction, speed, order and changes. |
| TempCompass yes/no | All 2,453 test questions | Binary temporal decisions. Not the benchmark's caption-generation suite. |

Pinned baselines: upstream Jev-Omni, public general v3, unified v4, original
Google Gemma 4 12B IT, Qwen3.5-4B and Qwen3.5-9B. The native
[decider-2b-vision](https://huggingface.co/Mapika/decider-2b-vision) adapter is also
evaluated on BLINK's eligible **single-image subset only**; the release is not a
native multi-image or video decision model. Compare all other systems on those
same eligible IDs before making a decider comparison. Its native answer-slot
prompt/readout is preserved, rather than treating its weights as a chat model.

Primary comparisons use batch one, H100, BF16, SDPA, thinking disabled, original
questions/options, and the same 16 midpoint video frames. Jev uses its native
decision head; ordinary VLMs score allowed letter-token logits in one forward
pass. Report accuracy and per-task macro averages separately, failures in the
denominator, model/dataset revisions, raw option scores, and request latency.
This is a declared **fast decision protocol on standard datasets**. It is not
identical to published model-card results using generation, thinking, different
frames, resolutions or hardware, and does not establish a leaderboard rank.

Qwen's FLA 0.5.2 kernel is installed; its causal-convolution fallback remains
recorded. First-use kernel compilation can enter an initial request's latency;
the aggregate tail includes that cost. These eager-runtime measurements are
not a comparison of each vendor's best optimized production server. Record
decode/preprocess/transfer/forward separately before claiming a production
latency advantage. Head time is not full multimodal decision time.

The exact-byte overlap audit compared benchmark media with 45,126 unique
recorded general/GUI training hashes and found zero matches. This does not
detect transcoding, crops or near duplicates, and upstream backbone pretraining
is unknown. SoccerNet source-media identity overlap was filtered separately.

[MVBench](https://github.com/OpenGVLab/Ask-Anything/blob/main/video_chat2/MVBENCH.md)
is a useful next standard comparison, but all 20 tasks and their videos must be
available before calling a score full MVBench. Upstream Jev-Omni's reported 53.1%
covers 14 tasks/2,786 questions. A subset must retain its own task list and
denominator. Video-MME's full media is much larger; do not call a short-clip or
small-frame diagnostic its full official score. Public JevBench's 231 text
decisions can measure Choice/Noul/Score retention, but cannot establish a vision
ranking or its current sealed composite.

## Video representation and next training decision

The current feature cache decodes clips into ordered images and does not insert
timestamps or multi-clip boundaries. V4 replay uses that same representation
so cached features remain valid. Three fixed-checkpoint benchmark runs compare:

1. Ordered image frames, as used in training.
2. The same sampled frames with their source timestamps and one continuous-clip
   boundary stated in the input.
3. Native video processor input, 16 frames and 70 soft tokens per frame. The
   processor's sampling can differ from midpoint sampling; it is a representation
   and budget comparison, not an isolated encoding-container experiment.

Native video processing passed a six-question adapter check and was faster
there. The sample is too small for an accuracy or latency advantage claim.
Do not infer motion from multi-camera stills, attach invented FPS to unrelated
images, or call ordered frame position an actual timestamp. MP4 packaging alone
does not add spatial information. Training on timeline-aware features requires
re-extracting features; existing image-feature caches cannot substitute for it.

If the full benchmarks confirm little transfer from head-only training, the next
controlled experiment is LoRA on selected shared Gemma transformer/projection
layers with general/GUI/visual SoccerNet replay. That can change the evidence
representation without adding an agent loop or generated reasoning at inference.
Compare it against the locked head-only baseline using the same input budget.
Full-parameter continued training and full head rebuilding are distinct options;
pretraining a 12B backbone from scratch is not supported by a 59,000-row mixture.
Backbone-training modes are proposed next experiments, not completed runs.

## Release gate

Publish one consolidated Transformers checkpoint plus the selected head,
processor, upstream notices, Choice/Noul/Score loader, precise source provenance,
dataset audit and aggregate benchmark results. Copy the pinned upstream merged
backbone on Modal and replace only its head. This creates one downloadable model
without asking users to assemble separate head repositories. Existing releases
remain immutable. Anonymous readback must match the selected head hash and
exercise all three typed outputs before marking the consolidated release ready.
Benchmark status and limitations belong in the model card; no SOTA or hidden-test
claim is permitted from the internal 86.72% score.

### Recover completed benchmark measurements

Run `python3 -m modal run -m experiments.modal_official_benchmarks --mode aggregate`
to validate complete saved prediction coverage and regenerate the aggregate
without repeating inference. Then `--mode analyze` runs paired cluster
bootstrap comparisons. Both completed after Modal access recovered. V4's
accuracy differences against v3 include zero in their 95% bootstrap intervals
on all three primary benchmarks; these results do not establish a broad gain.
