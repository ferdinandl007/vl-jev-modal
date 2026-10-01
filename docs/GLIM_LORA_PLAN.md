# Glim 9B: Qwen LoRA pilot

User direction, 1 October 2026: switch the next candidate to Qwen3.5-9B and
train a balanced text, image and temporal mixture. Published Glim 12B remains
the verified Gemma-based release; its results do not transfer to this candidate.

## Bounded pilot

- Base: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Training: 960 examples, 320 per modality. Separate development and test
  samples each contain 32 per modality. Frozen manifests live on Modal.
- Rank 8, alpha 16, dropout 0.05; language self-attention Q/K/V/output
  projections only. Vision encoder stays frozen. One epoch, learning rate
  0.00002, accumulation 8, BF16, gradient checkpointing.
- Text uses an Ollama-shaped single-question schema and A–Z answer logits.
  Choice, Noul and Score are trained through candidate-token classification.
- Video uses four ordered frames decoded from real clips, with timestamps and
  duration. This is a declared frame-sequence extension, not stock Ollama
  video input or a native-video processor evaluation.
- All downloads, preparation, training and artifacts stay on Modal.
- One A100 80GB job, at most 90 minutes (roughly $5 maximum compute budget).
  Workspace usage cap is $30; paid spend cap is $0. Full retraining is not
  automatically launched.

Implementation: `experiments/modal_glim_qwen_lora.py`. Deployment:
`glim-qwen9b-lora-pilot`. Initial training call:
`fc-01M3TX30PPB02FXCQ3XX9HA492`.

## Evaluation and publication gates

A completed pilot is a candidate, not a promoted release. Derived pilot
samples establish basic performance only. Run the selected candidate on exact
public benchmark subsets, preserve split and source overlap checks, and reuse
published competitor scores only when protocols match. Do not rerun competitor
models or infer improvement from existing Qwen base scores.

Save the adapter before evaluation; record dependencies, targeted modules,
manifest hashes and actual results. Check finite losses and gradients. A
runtime limit saves a partial adapter and does not mark the run complete.

Ollama publication requires conversion/import and actual runtime verification.
The current stock decision API supports text; vision/video need a separate
explicit media interface. Neither a LoRA file nor our API adapter proves
`ollama pull` compatibility. See [compatibility notes](OLLAMA_COMPATIBILITY.md).

The existing large source mixture and 117,330 new text rows remain available
for a later budgeted expansion. This initial balanced sample does not claim
full-corpus training, calibrated confidence or state-of-the-art accuracy.

## Expanded curated run (approved 1 October 2026)

The user approved 20,000 unique examples and a faster GPU, then requested a
larger SoccerNet share informed by existing measured weaknesses. Source family
weights use those aggregate results; individual benchmark/test examples remain
reserved. This is heuristic curation with original source labels, not a claim
that every retained visual question has received human sufficiency review.

Implementation: `experiments/modal_glim_curated.py`; run:
`qwen9b-curated-20k-v3`. The earlier v1 preparation excluded multi-asset replay
and legacy cached counting images and was never trained. V2 checked additional media support; v3 freezes the
final difficulty/diversity mix and stricter knowledge exclusions. V3 preserves up to
eight referenced media assets per question, separate clip timelines and four
ordered timestamped frames per clip. It includes only already saved Modal
media. Curated image inputs have a 262,144-pixel budget; individual video frames have
a 65,536-pixel budget. Source grouping, exact media hashes and material identities protect
held-out data; universal match-level or near-duplicate separation is not
certified.

- 8,400 text: 2,000 routing, 1,700 evidence/answerability, 3,000 human-rated
  Score and 1,700 derived message queues.
- 6,600 images: approximately 3,000 GUI, 1,500 explicit visual-cue questions,
  900 actions/counting and all eligible direct-visual SoccerNet images.
- 5,000 video: all eligible direct-visual SoccerNet clips up to 2,200, then
  UCF101 to fill the remaining allocation. Scarcity reallocations are recorded.
- Choice alternatives are shuffled without changing original values or labels.
- Initial base is the pinned Qwen3.5-9B checkpoint, not the pilot adapter;
  hold-outs and results are therefore attributable to this curated training.
- Rank 8; BF16; H100; microbatch 2 and effective batch 8. One epoch gives
  2,500 optimizer updates. Vision encoder remains frozen.
- Hard function timeout 13,800 seconds, internal budget 13,200 seconds;
  conservative compute bound about $18 including allocated CPU/memory.
- After warm-up, measure throughput and project completion cost. Save a partial
  adapter and stop if completing the run with evaluation reserve would exceed
  the internal runtime allocation. Account limits remain $30 usage / $0 spend.

The pilot finished in 974.9 seconds. Its 32-example-per-modality test diagnostics
were text 26/32, image 21/32 and video 32/32. These small derived samples do not
establish improvement over the base or a public benchmark rank. Larger public
benchmark evaluation and actual Ollama conversion/import remain release gates.

### Frozen v3 inventory and submission

The CPU preparation completed with 20,000 unique training decisions, 192
development and 192 test diagnostics. Actual training: 8,400 text, 6,600 image,
5,000 video. All 3,358 direct-visual SoccerNet training rows are admitted:
800 action, 560 jersey colour, 476 score/time, 386 game state, 320 camera class,
320 camera switching, 318 replay and 178 multiple-view foul recognition.
This is about 17% of the mixture. All 95 cached PixMo Count examples are retained.

Source and difficulty balance uses a deterministic alternating mix of close
candidate overlap and random diversity within source/family/label strata.
Explicit visual-cue image questions exclude obvious explanation/speculation
wording; this remains a heuristic filter rather than human visual review.

Training SHA256:
`3e572ee30969244427e34f00e17fdeba3b8871fe910fe3bb3ddc5e2ff7d93f96`.
Development SHA256:
`8d996acf8b3ae08232c91705da5ef4c6787c5c8001fd6bb3b8f29dcbf10b2d8a`.
Test SHA256:
`e0eeef21506f4c7ac483326a22067f0b0db9b7cf5242efe8ba34de12e6f31d02`.

Modal deployment: `glim-qwen9b-curated`. Training call:
`fc-01M3TYJAN9CEH9KJ0PB3XJ7N3S`. Credit read before submission: $1.60 metered,
$0 billed, $28.40 remaining. The launcher checks the manifest and refuses a
duplicate submission. `launch.json`, progress, budget-stop and completion reports
are saved in `glim-qwen-lora/qwen9b-curated-20k-v3` on Modal.

### H100 compiler repair

The first H100 call failed on its first backward pass: FLA explicitly refuses
the known incorrect gated-delta backward kernel on Hopper with Triton 3.4–3.7.0.
No optimizer update completed. The corrected image pins Triton 3.7.1, the
version recommended by the kernel guard ([upstream issue](https://github.com/fla-org/flash-linear-attention/issues/640)).
This explicitly overrides Torch 2.10's bundled Triton 3.6 metadata pin for the
FLA custom kernels; actual training forwards/backwards must pass finite-loss
and gradient checks. The dependency versions are recorded in the final report.
No incorrect-kernel guard is disabled.

Remaining credits were checked again before the single restart: $1.71 metered,
$0 billed. Current training call: `fc-01M3TYZ9R2F2J7DNSHK1DYQGAC`.
The original failed call is retained in the launch audit. The dataset and
runtime/credit allocation remain unchanged.

### Input budget verification and stable training

A second early call stopped at the context-size guard. In Transformers 5.17,
legacy processing kwargs can replace an explicit `processor_kwargs` dictionary.
Putting padding inside that dictionary preserves both padding and image limits.
The shared renderer was checked on Modal CPU with ten two-example batches,
including text, ordinal Score, counting, GUI and five-asset replay inputs.
All passed; longest tested input was 1,809 tokens, with finite image tensors.
No media were dropped to make the checks pass.

Current training call: `fc-01M3TZC3MY7N5DTKNRN76HER92`. It passed 224 training
examples and 28 optimizer updates with finite losses/gradients. Early speed
following the first 160 examples: 3.49 examples/second, projecting 5,875 seconds
and approximately $7.60 in training compute before evaluation. This is an early
estimate, not a finished-run measurement or a controlled GPU-only comparison;
batching and the curated input distribution also differ from the A100 pilot.
The conservative $18 training allocation and account credit caps remain in force.
Latest billing read: $1.92 metered, $0 billed.
