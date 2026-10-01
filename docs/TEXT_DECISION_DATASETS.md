# Large text corpus for Choice, Noul and Score

30 September 2026. All source downloads, normalization, feature extraction,
training and evaluation run on Modal. No source texts, media or weights are
stored in this checkout.

## Sources admitted

Status on 1 October 2026: corpus build and packed feature extraction completed.
Modal access recovered after a temporary workspace-disabled error. Training
resumed after fixing a loader assumption: MultiNLI pair IDs can repeat for
distinct inputs, so saved features are verified using each pack's manifest
hash, model revision, row position, source ID and target. No integrity checks
are bypassed. Training was subsequently stopped at the user's request to pause Modal spending.
The candidate has no verified completed training report;
do not claim that the new data is already in a published checkpoint.

The frozen `text-decisions-v2` build contains **117,330 new training decisions**,
8,308 development decisions, 5,604 reserved calibration decisions and 6,337
test diagnostics. The feature/training pipeline consumes this frozen manifest
with all 59,000 existing v4 training examples as replay: **176,330 examples**
per epoch. A newly selected head must pass development retention gates before
being considered a release candidate. These counts do not mean that a new
checkpoint has completed training or passed an external benchmark.

| Source | Source training inventory inspected | Admitted training decisions | Use |
| --- | ---: | ---: | --- |
| [MASSIVE 1.1](https://github.com/alexa/massive) | 34,542 in en-US, de-DE and zh-CN | 30,325 | Intent Choice, close alternatives and none-of-these cases. |
| [CLINC150, plus config](https://huggingface.co/datasets/clinc/clinc_oos) | 15,250 | 13,672 | Broad intent routing and out-of-scope cases. |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 10,003 | 8,811 | Fine-grained customer-message routing. |
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | 392,702 | 29,997 | Three-way entailment / unresolved / contradiction Choice, and explicitly defined binary evidence-sufficiency Noul. |
| [SQuAD 2](https://huggingface.co/datasets/rajpurkar/squad_v2) | 130,319 | 19,997 | Noul: can the supplied passage answer the question? Includes unanswerable questions. |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | 35,331 | 11,528 | Human-rated 0–4 Score for coherence or verbosity. Uses the supplied prompt and response; excludes factual-correctness ratings that may need outside evidence. |
| Derived message queues | Composed from admitted intent records | 3,000 | Choice: oldest message matching an intent, or none; Noul: any matching message exists. |

MASSIVE's million-example headline includes parallel translations across 52
languages. These are not a million independent English decisions. Source IDs
group the three admitted languages together so translations cannot cross
our internal train/development/calibration splits.

The three existing operational text sources—BANKING77, SNLI and SST-5—supply
474 examples to the older general training mixture. ScienceQA also contributes
text questions; 474 is not the total number of text inputs in v4.

## Label and input contract

Source labels are reused without model-generated relabeling. Intent choices
use eight candidates, including an explicit `none_of_these` option. Candidate
selection prioritizes labels sharing words with the source intent, then adds
other alternatives. In 20% of in-scope cases the source intent is omitted;
the correct decision is therefore none of the supplied candidates. Source
out-of-scope records also target none of the listed intents.

Noul is a binary probability, not a null or abstention value. A MultiNLI
unresolved claim remains distinct in three-way Choice. The Noul question
explicitly asks whether the supplied premise **establishes** the claim;
both contradiction and insufficient evidence answer false for that gate.

Message queues contain three to five original messages, visible IDs and ages.
The rule requests a particular intent and, for Choice, the oldest matching
message. Source human intent labels and visible ages determine the composed
answer. Labels are never inserted into the individual message text. These are
derived message-routing exercises, not a newly human-annotated email corpus or
gold decisions about real business urgency. Queue constituents come only from
their already protected split.

Score retains ordered human ratings rather than inventing percentages. The
training loss includes an ordinal cumulative-distribution term alongside
cross entropy. Evaluation reports expected-level mean absolute error as well
as exact-class accuracy. Human ratings remain subjective; they are not exact
truth or a guarantee of calibration.

The new text feature prompt matches the public typed adapter's key-prefixed
option format. Training preserves a single shared decision head. Multiple
named API questions currently take one backbone forward per question.

## Separation and benchmark scope

The builder pins source revisions, hashes downloaded files and split
manifests, and protects normalized text and source groups across splits.
It groups MultiNLI by premise, SQuAD by article and HelpSteer by prompt.
It removes new training records matching prior internal held-out text or
reserved public diagnostic states. New development/calibration records that
match previously trained v4 text are also excluded. Exact duplicate decisions
are removed. Unknown upstream pretraining and semantic near duplicates remain
outside what this audit can certify.

These bounded intent tasks are **not** the official 60-, 77- or 150-intent
classification benchmark. SQuAD answerability is not extractive EM/F1. The
6,337-row test is a declared derived diagnostic, not a full-source leaderboard
score. Full source test/validation files remain available on Modal for later
source-native evaluations, with inherited training overlap disclosed.

[JevBench](https://github.com/fstandhartinger/jevbench)'s 231 public original,
easy and hard tasks remain evaluation-only. The pinned source scoring code
is used without modification. Choice and Score retain original label order;
Noul false/true map to source no/yes. This does not evaluate the current
sealed composite or establish its rank.

[Fast Decisions](https://huggingface.co/datasets/fastino/fast-decisions) is
especially relevant to email triage, ticket routing, human handoff and
multi-label gates. Its public files contain 1,700 **development** examples,
100 in each of 17 domains. The publisher's benchmark uses a separate held-out
test. Public development examples remain reserved diagnostics; scores on
them must not be compared directly with its published test leaderboard.

ProofWriter and the inspected Enron-spam mirror were not admitted because a
source-specific data license was not confirmed. Yelp's large rating corpus
has academic-use restrictions; the initial Score expansion uses permissively
licensed human HelpSteer ratings. Raw Enron emails do not supply human gold
labels for routing, reply obligations or list selection.

## Modal execution

```sh
# Reproduce or inspect the frozen corpus without downloading it locally.
modal run -m experiments.modal_text_decisions --mode build
modal run -m experiments.modal_text_decisions --mode status

# Evaluate the current upstream/v3/v4 heads on public typed tasks.
modal run --detach -m experiments.modal_text_decisions --mode jevbench

# Extract packed features and train one candidate with full v4 replay.
modal run --detach -m experiments.modal_text_decisions --mode pipeline
```

Dataset: Modal Volume `vl-jev-text-decisions`, directory `text-decisions-v2`.
Candidate: `vl-jev-general-v1-training/general-head-v5-text-decisions`.
Its model selection uses text-family macro accuracy, with at most 0.5 percentage
point regression in general text/vision development accuracy, one percentage
point in GUI/Soccer development accuracy, and no more than 0.02 worsening in
Score expected-level MAE. Public benchmarks do not select epochs. A candidate
does not inherit v4's external benchmark scores.
