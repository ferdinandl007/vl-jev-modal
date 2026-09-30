# GUI decisions: choose controls, not coordinates

The current `gui_next_click` task asks Jev-Omni to choose one of four screen coordinates from a broad trajectory goal. The v2 head reached 51/155 (32.9%) on development and 34/112 (30.4%) on test. Its source, [Webintosh](https://huggingface.co/datasets/Chengheng/Webintosh), also records a specific instruction for each clicked element. Some recorded actions do not follow from the broad goal alone, so this score mixes missing intent, visual grounding, and action selection.

## Runtime contract

Give the model the screenshot, the user's goal, recent actions, and a list of available controls. A control has an ephemeral ID, role, accessible name, and optional window or section context. The decision is an operation plus an element ID, such as `{"operation":"click","element_id":"e17"}`. The executor resolves the ID against the current accessibility tree or parser output. It checks that the ID still exists before executing. It can use a bounding box internally if the control came from vision, but the model does not need to produce a coordinate.

Use the accessibility tree when available. For controls missing from the tree, [Microsoft OmniParser v2](https://huggingface.co/microsoft/OmniParser-v2.0) detects interactable regions and captions icons; draw numbered marks on the image and pass the corresponding element labels to the model. The MIT `icon_detect_v3` detector and MIT caption model are the permissive combination documented by Microsoft. Keep the parser's box-to-ID map inside the executor.

The Jev-Omni head currently accepts at most ten choices. On screens with more controls, retrieve a shortlist from the full element list using goal, history, role, name, and section, then ask the multimodal model to choose among those controls. Measure shortlist recall separately; a perfect decision head cannot recover a missing target.

## Evidence from Modal development probes

| Input and task | Result | Interpretation |
| --- | ---: | --- |
| Broad Webintosh goal + screenshot + coordinate options | 51/155, 32.9% | Existing v2 development score; next action may be underdetermined. |
| Exact local instruction + screenshot + coordinate options | 70/155, 45.2% | Oracle intent improves grounding, but still leaves coordinates difficult. |
| Exact local instruction + screenshot + four distinct accessible labels | 61/67, 91.0% | Semantic element selection works on the named subset. |
| Same semantic choices without screenshot | 63/67, 94.0% | The named subset is mostly solvable from text; it is not a vision benchmark. |
| OmniParser region contains target center | 38/40, 95.0% | Early detector coverage only; this does not measure box quality. Mean 127.5 regions per screenshot. |
| Target center in detector's ten highest-confidence regions | 10/40, 25.0% | Detector confidence alone is a poor shortlist policy. |

The named-element filter kept only 67 of 155 Webintosh development rows; 88 had at least one unnamed candidate. These are small diagnostic samples, not a general computer-use success rate. The OmniParser center-coverage metric can count overly large boxes as hits, so it is an optimistic detector diagnostic.

### Task-conditioned Mind2Web pilot

The Modal pilot used one pinned Multimodal Mind2Web training shard. It kept 150 train and 59 development click decisions, split by task ID (34 train tasks, 10 development tasks). The model input was the task, up to five prior actions, four named candidate controls, and optionally the screenshot. The correct action description was excluded. All four controls received the same style of visual mark; the gold slot was rotated deterministically.

| Input on 59 development decisions | Published general v2 head | Pretrained head |
| --- | ---: | ---: |
| Named controls only | 47/59 (79.7%) | 48/59 (81.4%) |
| Screenshot + named controls | 49/59 (83.1%) | 48/59 (81.4%) |
| Screenshot with all four controls marked + named controls | 52/59 (88.1%) | 48/59 (81.4%) |

The marked image gained five correct decisions over labels alone for the published head. This is a one-shard development pilot with ten tasks and four preselected candidates. It does not measure recall from the full page, typing, dropdown selection, live task completion, or generalization to official Mind2Web test sites/domains. The 150 new training rows have **not** been used to train a new head.

## Next benchmark

[Multimodal Mind2Web](https://huggingface.co/datasets/osunlp/Multimodal-Mind2Web) is a better action-selection source: each step has a screenshot, task, prior action sequence, operation, a positive element, and negative element candidates. It provides 7,775 train actions and separate test-task (1,339), test-website (1,019), and test-domain (4,060) actions. We should reserve official tests and split train by task ID for development. The correct `target_action_reprs` must never appear in the input; only earlier actions may be history. The dataset card labels this research use under OpenRAIL.

Measure four things separately:

1. **Candidate recall:** fraction of gold controls present in the accessibility/parser output and in the top ten shortlist.
2. **Action accuracy:** correct element ID and operation, conditioned on the gold control being present. Report click, type, and select separately.
3. **Image contribution:** compare screenshot + element labels with element labels alone on the same examples. Also test marked screenshots when labels are ambiguous or missing.
4. **Task success:** run the resulting agent in [BrowserGym](https://github.com/ServiceNow/BrowserGym), which exposes screenshots and accessibility trees, then report completed tasks and recovery after invalid or stale element IDs.

Train a GUI action adapter on task + history + screenshot + control list, with explicit replay of the existing text, image, and video mix. Select on task-grouped development data; score the official Mind2Web and BrowserGym holdouts only after selection. The existing 83% general vision target remains a separate guardrail and should not be conflated with GUI action accuracy.
