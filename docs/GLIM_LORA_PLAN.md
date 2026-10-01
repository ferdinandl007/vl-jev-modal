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
