# Glim 12B: Ollama and decision API compatibility

Research verified 1 October 2026 against official Ollama sources.

## What is available today

[Ollama 0.35 introduced `/v1/systemone`](https://ollama.com/blog/ollama-now-supports-jev-style-decision-models)
with named Choice, Noul and Score questions. The initial models are Nimble 9B,
Tev1 4B and Tev1 0.8B. The [published implementation](https://github.com/ollama/ollama/blob/main/decision/systemone.go)
compiles question schemas into prompts, assigns A–Z answer codes, and scores
candidate token logits. Its request supports text state (including serialized
objects/arrays), 1–64 questions and 2–26 candidates. The current decision request
has no native image/video/audio field.

Glim uses a merged Gemma 4 12B backbone plus a separately trained `head.pt`
and its native feature/prompt path. Importing the backbone using a Modelfile
would not automatically execute this head. API shape compatibility does not
establish architecture or output equivalence. We have not published a stock
`ollama pull` artifact and must not advertise one yet.

## Faithful compatibility paths

| Route | Preserves native head | Stock Ollama | Required work |
| --- | --- | --- | --- |
| `/v1/systemone` service wrapping Glim | Yes | Compatible client API; separate runtime | Native classifier, HTTP route, SDK verification |
| Native Ollama/llama.cpp head implementation | Yes if reproduced exactly | Requires runtime support | Export head tensors, reproduce feature extraction/pooling/head/masking and prompt formatting, wire scoring dispatch, validate parity before quantization |
| Train/distill a token-scoring edition | A different model | Can use supported token-scoring runtime | Train on Ollama's exact compiled prompts and A–Z outputs; evaluate and label separately |
| Import only the Gemma backbone | No | Potential backbone import | Would omit our head training; cannot inherit Glim scores |

The first route has a strict adapter in `experiments/systemone_adapter.py`.
It preserves native inference, option order, absence options, Noul P(true),
Score expected level and entropy confidence. It validates the full request
before inference. The common supported range is 2–26 Choice options and
2–10 Score levels; broader Ollama Score requests are explicitly rejected.
The adapter does not invent token usage. It is not an HTTP server or a claim
of full TypeSafe SDK integration verification. Native inference quality and
latency remain those of the published model. Vision stays on the original
multimodal interface until a declared media extension is implemented.

## Publication sequence

1. Publish the named complete Glim checkpoint with native head and provenance.
2. Publish a separately labeled API adapter without changing model weights.
3. Implement and verify a runtime port or train a token-scoring edition before
   listing a stock Ollama artifact. Export/quantization and parity checks run
   on Modal; no laptop model download or inference.
4. Publish results for the actual exported artifact, precision and protocol.
   Do not copy BF16 native-head scores onto a different or quantized backend.

## Public comparison suite, with no competitor reruns

Ollama's launch uses Bespoke's [public benchmark suite](https://github.com/bespokelabsai/nimble/blob/main/docs/PUBLIC_BENCHMARKS.md):
**3,880 human-labeled decisions across 13 subsets**, with exact IDs and
checksums committed. It includes MASSIVE English/German routing, BoolQ,
SQuAD answerability, PAWS, MultiNLI, Civil Comments, Aegis2, HelpSteer2,
SummEval relevance/consistency, PubMedQA and VitaminC. Rebuild those exact
subsets and run only Glim. Compare published scores using the same criteria,
reference labels and scoring. Keep those IDs out of future training and
report any inherited overlap. Source training families in our new text corpus
must be disclosed; source test examples are reserved, not training data.

This is the next planned standard text evaluation, not a completed Glim result.
Our existing public JevBench 231 and BLINK/TempCompass results remain separate.
