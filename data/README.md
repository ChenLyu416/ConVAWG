# Data directory

The ConVAWG dataset is **not** stored in this repository. It is released under
a data-use agreement (`../DATA_USE_AGREEMENT.md`) on Hugging Face:

https://huggingface.co/datasets/ChenL49/ConVAWG

Complete the agreement form on that page (access is granted on acceptance),
then download it here:

```bash
huggingface-cli login
huggingface-cli download ChenL49/ConVAWG --repo-type dataset --local-dir data/ConVAWG
```

```
data/ConVAWG/
├── README.md                              # dataset card and record structure
├── DATA_USE_AGREEMENT.md
├── scenarios/<scenario_id>.json           # Stage 1: personas, profiles, crime types, narrative
├── event_graphs/<scenario_id>.json        # Stage 2: events and relations
├── expanded_event_graphs/<scenario_id>.json   # Stage 2: sub-events and timestamps
├── chat_scripts/<scenario_id>.json        # Stage 2: one plan per conversation
└── dialogues/<scenario_id>_dialogues.json # final dialogues (200 scenarios, 6,171 events)
```

`dialogues/` is self-contained. The other directories are the planning
metadata, in the formats the corresponding pipeline stages write, so a later
stage can be re-run from them. To fetch the dialogues only, add
`--include "dialogues/*" "*.md"` to the download command.

`Experiments/Downstream_tasks/escalation_forecasting` and
`synthetic_transfer` read `data/ConVAWG/dialogues/` by default. The
`toxic_classification` and `persona_relation` tasks are built from the
pipeline's stage 6 and stage 5 outputs respectively; pass those directories
explicitly (see each task's `build_dataset.py --help` and config).

## Field reference

All files are keyed by `scenario_id` (the file name). `event_id` links the
levels: `E3` is an event in `event_graphs/`; `E3.2` is its second sub-event in
`expanded_event_graphs/`; `E3.2.pre`, `E3.2.during` and `E3.2.post` are
conversations planned before, during and after an offline sub-event; an ID
without a suffix is a sub-event that itself takes place online. Entries of
`chat_scripts[]` and `dialogues[]` with the same `event_id` are the plan and
the realisation of one conversation, in the same order.

### `scenarios/<id>.json`

| Field | Description |
|---|---|
| `persona_id`, `persona_text` | The persona seed |
| `victim_profile` | Name, age, gender, occupation, ethnicity, vulnerabilities, support seeking |
| `perpetrator_profile` | Name, age, gender, occupation, ethnicity, relationship to the victim, motivations, history, characteristics |
| `third_party_actors` | Other characters with their role, relationship and actions |
| `crime_types` | CPS-aligned crime labels |
| `time_duration` | Time span covered by the scenario |
| `scenario_description` | Narrative description of the scenario |
| `generation_metadata` | Generator model, backend and timestamp |

### `event_graphs/<id>.json`

| Field | Description |
|---|---|
| `scenario_id`, `victim_name`, `perpetrator_name`, `third_party_names` | Scenario key and cast |
| `events[]` | `event_id`, `description`, `agent` and `patient` (name and role), `action`, `time`, `timestamp`, `location`, `attributes` (polarity, modality, time hint, crime-type hints), and the source sentence (`text_span`, `evidence`) |
| `relations[]` | `src_event_id`, `dst_event_id`, `relation_type` (`ESCALATES_TO`, `TEMPORAL_BEFORE`, `CAUSES`, `ENABLES`, `EVIDENCE_OF`, `CONTRASTS_WITH`), `confidence`, `evidence` |
| `metadata` | Crime types, time span and automatic quality checks |

### `expanded_event_graphs/<id>.json`

| Field | Description |
|---|---|
| `original_events`, `original_relations` | The event graph, repeated |
| `expanded_events[]` | Per event: `parent_event`, `parent_goal`, `sub_events[]` (`sub_event_id`, `location`, `location_type`, `agent`, `patient`, `action`, `description`, `crime_type`) and `sub_relations[]` |
| `timeline_span_days` | Length of the scenario timeline in days |
| `metadata` | As above, plus the fictional timestamp assigned to each event |

### `chat_scripts/<id>.json`

`chat_scripts[]`, one entry per conversation:

| Field | Description |
|---|---|
| `event_id`, `timestamp` | Link to the dialogue, and the fictional date and time |
| `event_type` | `online_chat` (the sub-event is itself a conversation) or `online_chat_expansion` (a conversation around an offline sub-event) |
| `expansion_purpose`, `original_physical_event` | For expansions: `planning`, `during_event` or `aftermath`, and the sub-event concerned |
| `platform`, `platform_description` | Type of messaging channel |
| `participants` | Sender and receiver, each with `name` and `role` |
| `conversation_context` | `action`, `description`, `purpose`, `tone`, `emotional_state`, `escalation_level` (0–4), `crime_type` |
| `content_guidance` | Main points, suggested length, style notes and red flags for each role |
| `coherence_requirements` | For expansions: summaries of earlier conversations (`previous_context`), information to convey, follow-up hooks |
| `conversation_summary` | For `online_chat` events: narrative outline of the conversation |

`previous_context` repeats earlier summaries in every later plan, which makes
this the largest directory (about 115 MB).

### `dialogues/<id>_dialogues.json`

| Field | Description |
|---|---|
| `scenario_id`, `victim_name`, `perpetrator_name`, `third_party_names` | Scenario key and cast |
| `vawg_scenarios` | The scenario, identical to `scenarios/<id>.json` |
| `num_events` | Number of dialogue events |
| `dialogues[]` | `event_id`, `timestamp`, `participants`, `conversation_context`, `enforced_dialogue` (text before toxicity injection, one `Speaker: message` per line) and, where applied, `caa_rewritten_dialogue` (text after injection) |

The final text of an event is `caa_rewritten_dialogue` when present, otherwise
`enforced_dialogue`. `conversation_context.escalation_level` is `null` for
conversations that do not involve the perpetrator; the level planned for the
underlying event is in the matching chat script.

Everything under `data/` except this file is git-ignored.
