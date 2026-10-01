# Glim 12B: mixture first, then LoRA

User direction, 1 October 2026: finish a balanced mixture of text, vision and
temporal tasks, then consider backbone LoRA. This is an experiment plan;
no new backbone adapter has been trained or published by this project yet.
The upstream backbone already contains merged LoRA training. Our v4 and the
active text expansion train the decision head while freezing that backbone.

## Why this is the next candidate

Head training can change the readout but cannot learn new backbone features.
The measured BLINK deficit and weak SoccerNet gains justify testing backbone
adaptation after establishing consistent input-only labels and holdouts.
LoRA reduces training memory and parameter count; it does not reduce the
12B backbone's inference size or guarantee faster decisions or better scores.

## Freeze the mixture

- Text: routing, absence/none, evidence sufficiency and ordinal rating.
- Vision: visual relations, source-grounded sports questions and GUI choices.
- Temporal: genuine ordered clips, motion/change/event order, preserving real
  timestamps. Do not infer continuity between unrelated views or train on
  answers that require unseen future frames or outside facts.
- Retain prior skills through balanced source/modality sampling and replay.
- Preserve held-out source groups before training. Public benchmark IDs are
  reserved and never used to choose adapters or epochs.
- Audio is not covered by the completed external benchmark suite; do not
  advertise an audio improvement from these experiments.

The existing full mixture and new text sources remain audited in the dataset
notes. The 117,330 new text rows outweigh the earlier 59,000 mixture, so
uniform sampling of all rows is not automatically balanced by modality.

## First LoRA pilot

Use one small adapter (initial rank 8 or 16) on a selected set of decoder
attention projections and train the existing decision head jointly. Record
all target modules, adapter rank/alpha/dropout, sampling weights, base/head
revisions and trainable parameter count. Start with the vision encoder frozen;
consider projection/encoder adaptation only if the measured failure cases
justify it. More rank or full retraining is a later experimental choice.

Cached frozen-backbone features cannot train LoRA: the pilot must use original
text/media with gradient-enabled backbone forwards. That changes compute cost.
Run a small, explicitly budgeted pilot before attempting the whole corpus;
do not automatically launch full retraining under the $30 credit allowance.

Select on development retention checks, then evaluate the selected Glim model
once per reserved public benchmark protocol. Reuse published competitor
scores; match splits, IDs, criteria, frame sampling and scoring when comparing.
Publish the new adapter/merged checkpoint and its own results. Existing v4
scores do not transfer to a newly trained or quantized artifact.

## Ollama requirement

A Gemma LoRA alone does not port the native head into Ollama's token-scoring
runtime. Faithful native-head runtime support and a token-scoring/distilled
edition are different publication routes. See `OLLAMA_COMPATIBILITY.md`.
