# Plan: Tool-Calling Verification Experiment for SceneIQ

## Task Description
Run a tool-assisted verification experiment on the SceneIQ Qwen3-VL-2B pipeline across 8,613 samples. The fine-tuned model generates predictions (scene, action, waypoints) at temperature 0, then the **base** Qwen3-VL-2B-Instruct model verifies those predictions using 4 statistical tools built from training data. Test 6 experimental conditions (baseline, prior-only, confusion-only, all-tools, staged-prediction, oracle) and measure whether tool-assisted verification can correct the model's systematic biases — specifically the 12.5x over-prediction of incident_zone (46.8% predicted vs 3.7% true rate).

## Objective
When complete, we will have:
1. A SQLite database at `tool_calling_experiment/tool_calling.db` containing every prediction, tool call, tool response, revision decision, and final answer for every sample under every condition
2. Net flip analysis: how many samples does each tool condition save vs break compared to greedy baseline
3. Per-tool effectiveness: which tools the model actually responds to (conditional revision probability)
4. Oracle ceiling: maximum possible F1 if tools were perfect — determines whether to pursue tool-calling further
5. Revision accuracy: when the model changes its answer after tool feedback, is the new answer better?
6. Latency measurements per condition
7. A comprehensive per-sample report and insight analysis (same framework as self-consistency experiment)

## Problem Statement
From the self-consistency experiment, we know:
- **46.9% of errors are high-bias** (model confidently wrong, agreement >= 90%)
- **Only 0.03% are high-variance** (model uncertain)
- Self-consistency voting is net-harmful (-62 flip balance)
- The dominant error is **nominal -> incident_zone** (3,404 samples, 50.6% of all nominal predictions)
- Root cause: `nominal_triggers` fine_class (2,801 samples, only 19.0% accuracy)
- The model's ~50% F1 is near its ceiling for sampling-based approaches

The hypothesis: **tool-assisted verification can correct systematic biases that sampling cannot**, because tools provide external information (base rates, co-occurrence statistics) that the model lacks.

## Research Findings (from Phase 0 Investigation)

### vLLM Tool Calling API
- **Offline API**: `llm.chat(messages, sampling_params, tools=tools)` at `vllm/entrypoints/llm.py:887`
- **Multi-turn**: Call `llm.chat()` -> parse tool calls from raw text -> append `role: "tool"` message -> call `llm.chat()` again
- **No automatic parsing in offline mode**: Must parse `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` from generated text manually
- **Batch support**: Pass list of conversations to `llm.chat()`
- **Example**: `examples/offline_inference/chat_with_tools.py` demonstrates full loop
- **Tool parser for online server**: Use `hermes` parser (not `qwen3_xml` or `qwen3_coder` which are for Coder models)

### Qwen3-VL Tool Format
- Chat template at `/fsx/mketkar-sceneiq/checkpoints/checkpoint_006/chat_template.jinja` has **full tool support**:
  - Tools rendered in `<tools></tools>` XML tags in system prompt
  - Model outputs: `<tool_call>{"name": "fn", "arguments": {...}}</tool_call>`
  - Tool responses: `<tool_response>...</tool_response>` tags in user role
- Special tokens: `<tool_call>` (151657), `</tool_call>` (151658), `<tool_response>` (151665), `</tool_response>` (151666)
- **CRITICAL**: Fine-tuned model CANNOT do tool calling — catastrophic forgetting from SceneIQ training. Must use **base Qwen3-VL-8B-Instruct** for all tool-calling phases. The larger model is a better tool caller than the 2B variant.

### Model Loading Strategy
- **8x H200 GPUs available** (144 GB each) — use separate GPUs for each model, all loaded simultaneously
- Fine-tuned model (prediction): `CUDA_VISIBLE_DEVICES=0` → `/fsx/mketkar-sceneiq/checkpoints/checkpoint_006/` (~4.6 GB)
- Base 2B verifier: `CUDA_VISIBLE_DEVICES=1` → `/fsx/models/Qwen3-VL-2B-Instruct` (~4.6 GB)
- Base 8B verifier: `CUDA_VISIBLE_DEVICES=2` → `/fsx/models/Qwen3-VL-8B-Instruct` (~16 GB)
- **Why test BOTH 2B and 8B**: The 2B-vs-8B comparison isolates how much verifier model size matters for tool-calling effectiveness. If 8B dramatically outperforms 2B on tool use, that tells us the limiting factor is model capability, not tool design. If both perform similarly, the tools themselves are the bottleneck.
- All three models can be loaded simultaneously on separate GPUs — no sequential unloading needed
- **Download steps required**: Both base model weights need to be downloaded from HuggingFace to `/fsx/models/` before inference (see Task 0)
- **FSx striping**: All model directories must be striped across all OSTs (`lfs setstripe -c -1`) before downloading weights. This parallelizes reads across storage targets for fast model loading. Existing files can be re-striped with `lfs migrate -c -1`.

### Data Available for Tool Construction
All from self-consistency DB (`self_consistency_experiment/self_consistency.db`):

**Ground Truth Class Distribution:**
| Scene Class | Count | Percentage |
|---|---|---|
| nominal | 6,728 | 78.1% |
| flagger | 660 | 7.7% |
| flooded | 618 | 7.2% |
| incident_zone | 322 | 3.7% |
| mounted_police | 272 | 3.2% |

**Prediction Distribution (Greedy):**
| Predicted | Count | Percentage |
|---|---|---|
| incident_zone | 4,032 | 46.8% |
| nominal | 3,714 | 43.1% |
| flagger | 457 | 5.3% |
| flooded | 293 | 3.4% |
| mounted_police | 105 | 1.2% |

**Confusion Matrix (GT -> Predicted, major errors):**
- nominal -> incident_zone: **3,404** (50.6% of nominal, THE dominant error)
- flagger -> incident_zone: 221 (33.5% of flagger)
- flooded -> incident_zone: 163 (26.4% of flooded)
- mounted_police -> incident_zone: 78 (28.7% of mounted_police)
- Total incident_zone over-predictions from non-incident classes: 3,866

**Fine Class Root Cause:**
- `nominal_triggers` fine_class: 2,801 samples, only 19.0% accuracy
- These are nominal scenes that contain visual triggers (traffic cones, barriers, etc.) that the model misclassifies as incidents

**Scene-Action Co-occurrence (15 unique GT combinations):**
- Only incident_zone has lane changes (lc_left, lc_right)
- nominal and mounted_police only have (null, null) actions
- flooded: slowdown+null, stop+null, proceed+null
- flagger: stop+null, slowdown+null, proceed+null

**HLA Thresholds (from waypoint_hla.py):**
- stop_speed=0.3 m/s, hard_decel=-1.5 m/s^2, slow_decel=-0.5 m/s^2
- proceed_accel=0.5 m/s^2, lane_width=0.2 m

**Verifier Model Tool Calling Capability:**
- Using 8B (not 2B) as verifier — significantly better at tool calling
- Qwen3-0.6B achieves 68% on BFCL single-turn; 8B should be substantially better
- 8B can handle 1-4 tools with flat schemas and single/multi-turn flows
- Still may struggle with complex multi-step tool chains — keep tools simple

## Solution Approach

### Key Design Decisions

1. **Pipeline A (Predict-then-Verify) is primary** — Fine-tuned model predicts at temp=0, base model verifies with tools. This allows batching both phases separately for maximum throughput.

2. **Pipeline B (Staged Prediction) is secondary** — Base model only, multi-turn with tools. Tests whether tools can replace fine-tuning entirely.

3. **Oracle condition runs FIRST** — If the ceiling is low (< 0.70 F1), abort the full experiment. This saves potentially days of wasted compute.

4. **Sequential model loading** — Load fine-tuned model, run all predictions, unload, load base model, run all verifications. No multi-GPU needed.

5. **Same SQLite + analysis framework** as self-consistency experiment — Enables direct comparison.

6. **4 statistical tools** built from self-consistency DB data — No external data needed.

7. **Parse tool calls manually** in offline mode — Regex for `<tool_call>...</tool_call>` XML blocks.

### Tool Definitions

```python
# Tool 1: Prior Distribution Check
TOOL_PRIOR_CHECK = {
    "type": "function",
    "function": {
        "name": "check_scene_prior",
        "description": "Check how common a predicted scene type is in the training data. Returns the base rate of the predicted scene and the most common scene overall. Use this to verify whether a rare prediction is plausible.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_scene": {
                    "type": "string",
                    "enum": ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"],
                    "description": "The predicted scene classification to check"
                }
            },
            "required": ["predicted_scene"]
        }
    }
}

# Tool 2: Scene-Action Compatibility
TOOL_SCENE_ACTION = {
    "type": "function",
    "function": {
        "name": "check_scene_action_compatibility",
        "description": "Check whether a predicted action is compatible with a predicted scene type based on historical co-occurrence data. Returns whether this combination has been observed and what actions are typical for this scene.",
        "parameters": {
            "type": "object",
            "properties": {
                "scene": {
                    "type": "string",
                    "enum": ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]
                },
                "long_action": {
                    "type": "string",
                    "enum": ["stop", "slowdown", "proceed", "null"],
                    "description": "Longitudinal action prediction"
                },
                "lat_action": {
                    "type": "string",
                    "enum": ["lc_left", "lc_right", "null"],
                    "description": "Lateral action prediction"
                }
            },
            "required": ["scene", "long_action", "lat_action"]
        }
    }
}

# Tool 3: Waypoint Feasibility Check
TOOL_WAYPOINT_CHECK = {
    "type": "function",
    "function": {
        "name": "check_waypoint_feasibility",
        "description": "Check whether predicted waypoint deltas are within the typical range for a given scene-action combination. Returns feasibility assessment and typical waypoint statistics.",
        "parameters": {
            "type": "object",
            "properties": {
                "scene": {"type": "string", "enum": ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]},
                "long_action": {"type": "string", "enum": ["stop", "slowdown", "proceed", "null"]},
                "first_waypoint_x": {"type": "number", "description": "First waypoint x-delta"},
                "first_waypoint_y": {"type": "number", "description": "First waypoint y-delta"}
            },
            "required": ["scene", "long_action", "first_waypoint_x", "first_waypoint_y"]
        }
    }
}

# Tool 4: Confusion Risk Detector
TOOL_CONFUSION_CHECK = {
    "type": "function",
    "function": {
        "name": "check_confusion_risk",
        "description": "Check whether a predicted scene type is commonly confused with another class. Returns the historical confusion rate and what class it is most often confused with. High-risk predictions should be double-checked carefully.",
        "parameters": {
            "type": "object",
            "properties": {
                "predicted_scene": {
                    "type": "string",
                    "enum": ["nominal", "flooded", "incident_zone", "mounted_police", "flagger"]
                }
            },
            "required": ["predicted_scene"]
        }
    }
}
```

### Tool Implementations

```python
CLASS_FREQUENCIES = {
    "nominal": 0.781, "flagger": 0.077, "flooded": 0.072,
    "incident_zone": 0.037, "mounted_police": 0.032,
}

CONFUSION_PAIRS = {
    "incident_zone": {
        "confused_with": "nominal", "error_rate": 0.506,
        "note": "incident_zone is predicted 12.5x more often than it actually occurs. "
                "46.8% of all predictions are incident_zone but only 3.7% of ground truth is. "
                "Most incident_zone predictions are actually nominal scenes with visual triggers "
                "(traffic cones, barriers). Look carefully: are there actual emergency vehicles, "
                "crashes, or road closures? If not, this is likely nominal."
    },
    "flooded": {
        "confused_with": "nominal", "error_rate": 0.264,
        "note": "26.4% of flooded scenes are misclassified. Look for standing water on the road surface."
    },
    "flagger": {
        "confused_with": "incident_zone", "error_rate": 0.335,
        "note": "33.5% of flagger scenes are confused with incident_zone. Look for a human flagging traffic."
    },
    "mounted_police": {
        "confused_with": "incident_zone", "error_rate": 0.287,
        "note": "28.7% of mounted_police scenes are confused with incident_zone. Look for horses."
    },
}

# Co-occurrence matrix: {scene: {(long, lat): count, ...}}
COOCCURRENCE = {
    "nominal": {("null", "null"): 6728},
    "incident_zone": {
        ("slowdown", "null"): 107, ("stop", "null"): 72, ("slowdown", "lc_left"): 46,
        ("proceed", "null"): 32, ("slowdown", "lc_right"): 29, ("stop", "lc_left"): 18,
        ("proceed", "lc_left"): 9, ("stop", "lc_right"): 5, ("proceed", "lc_right"): 4,
    },
    "flooded": {("slowdown", "null"): 348, ("stop", "null"): 171, ("proceed", "null"): 99},
    "flagger": {("stop", "null"): 391, ("slowdown", "null"): 190, ("proceed", "null"): 79},
    "mounted_police": {("null", "null"): 272},
}
```

### Verification Prompt Template (Pipeline A)

```
You are a driving scene verification system. A prediction model has analyzed dashcam images and made predictions about the scene. Your job is to verify these predictions using the provided tools, then output your corrected predictions.

The model predicted:
- Scene type: {predicted_scene}
- Longitudinal action: {predicted_long_action}
- Lateral action: {predicted_lat_action}

Use the available tools to check whether these predictions are reasonable. After checking, provide your final corrected predictions in this exact format:

FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REVISED: <yes|no>
REASON: <brief explanation>
```

## Relevant Files

### Existing Source Files
- `/workspace/vllm/self_consistency_experiment/run_self_consistency.py` — Inference script template (adapt for prediction phase)
- `/workspace/vllm/self_consistency_experiment/self_consistency.db` — Source data for tool construction + greedy baseline results
- `/workspace/vllm/self_consistency_experiment/compute_majority_votes.py` — Pattern for post-processing analysis
- `/workspace/vllm/self_consistency_experiment/generate_sample_report.py` — Pattern for per-sample JSONL generation
- `/workspace/vllm/self_consistency_experiment/generate_insights_report.py` — Pattern for insight report generation
- `/workspace/vllm/examples/offline_inference/chat_with_tools.py` — vLLM tool calling example (multi-turn pattern)
- `/fsx/mketkar-sceneiq/checkpoints/checkpoint_006/chat_template.jinja` — Qwen3-VL chat template with tool support
- `/fsx/mketkar-sceneiq/checkpoints/checkpoint_006/` — Fine-tuned model checkpoint
- `/fsx/mketkar-sceneiq/datasets/` — Local MDS dataset (8,613 samples)

### New Files to Create
- `tool_calling_experiment/create_db.py` — SQLite schema creation
- `tool_calling_experiment/build_tools.py` — Build tool statistics from self-consistency DB
- `tool_calling_experiment/tools.py` — Tool definitions and implementations (the 4 Python functions)
- `tool_calling_experiment/run_prediction.py` — Phase 1: Fine-tuned model generates predictions
- `tool_calling_experiment/run_verification.py` — Phase 2: Base model verifies with tools (Pipeline A)
- `tool_calling_experiment/run_staged.py` — Pipeline B: Base model with staged tool use
- `tool_calling_experiment/run_oracle.py` — Oracle condition: tools return ground truth
- `tool_calling_experiment/parse_tool_calls.py` — Parse `<tool_call>` XML from model output
- `tool_calling_experiment/compute_analysis.py` — Post-processing: flip analysis, revision rates, per-tool effectiveness
- `tool_calling_experiment/generate_sample_report.py` — Per-sample JSONL report
- `tool_calling_experiment/generate_insights_report.py` — Human-readable insight report
- `tool_calling_experiment/tool_calling.db` — SQLite database (created at runtime)

## SQLite Schema Design

```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

-- One row per experimental condition
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT,
    condition_name TEXT NOT NULL,  -- 'baseline', 'prior_only', 'confusion_only', 'all_tools', 'staged', 'oracle'
    pipeline TEXT NOT NULL,        -- 'A' (predict-then-verify) or 'B' (staged)
    predictor_model TEXT,          -- path to fine-tuned model (NULL for Pipeline B)
    verifier_model TEXT,           -- path to base model
    tools_enabled TEXT,            -- JSON list of enabled tool names
    temperature_predict REAL DEFAULT 0.0,
    temperature_verify REAL DEFAULT 0.0,
    seed INTEGER DEFAULT 42,
    total_samples INTEGER,
    total_wall_time_s REAL,
    predict_wall_time_s REAL,
    verify_wall_time_s REAL,
    throughput_samples_per_s REAL,
    status TEXT NOT NULL DEFAULT 'running'
);

-- One row per sample per condition
-- Stores both original prediction and final (possibly revised) prediction
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    sample_id INTEGER NOT NULL,
    chum_uri TEXT,
    -- Original prediction (from fine-tuned model or base model Turn 1)
    original_scene TEXT,
    original_long_action TEXT,
    original_lat_action TEXT,
    original_generated_text TEXT,
    original_scene_correct INTEGER,
    -- Final prediction (after tool verification, may be same as original)
    final_scene TEXT,
    final_long_action TEXT,
    final_lat_action TEXT,
    final_generated_text TEXT,     -- full verification model output
    final_scene_correct INTEGER,
    -- Revision tracking
    was_revised INTEGER,           -- 1 if final != original
    scene_was_revised INTEGER,     -- 1 if scene changed specifically
    revision_reason TEXT,          -- model's stated reason for revision
    -- Ground truth
    scene_type_gt TEXT,
    long_action_gt TEXT,
    lat_action_gt TEXT,
    odd_label TEXT,
    fine_class TEXT,
    location TEXT,
    -- Flip analysis vs greedy baseline
    original_flipped_correct INTEGER,   -- original wrong, final correct
    original_flipped_incorrect INTEGER, -- original correct, final wrong
    -- Timing
    predict_time_ms REAL,
    verify_time_ms REAL,
    total_time_ms REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, sample_id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_exp ON predictions(experiment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_revised ON predictions(experiment_id, was_revised);
CREATE INDEX IF NOT EXISTS idx_predictions_flip ON predictions(original_flipped_correct, original_flipped_incorrect);

-- One row per tool call per sample per condition
-- A sample may have 0-4 tool calls depending on condition
CREATE TABLE IF NOT EXISTS tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    sample_id INTEGER NOT NULL,
    tool_call_order INTEGER NOT NULL,  -- 0, 1, 2, 3 (order within this sample)
    tool_name TEXT NOT NULL,           -- 'check_scene_prior', 'check_confusion_risk', etc.
    tool_arguments_json TEXT,          -- JSON of arguments passed
    tool_result_json TEXT,             -- JSON of tool response
    -- Did the model revise after this specific tool call?
    model_revised_after INTEGER,       -- 1 if model changed prediction after seeing this result
    revised_field TEXT,                -- 'scene', 'long_action', 'lat_action', or NULL
    old_value TEXT,                    -- value before revision
    new_value TEXT,                    -- value after revision
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id, sample_id, tool_call_order)
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_exp ON tool_calls(experiment_id, sample_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_revised ON tool_calls(model_revised_after);

-- Aggregate metrics per condition
CREATE TABLE IF NOT EXISTS condition_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    -- Accuracy
    scene_accuracy REAL,
    scene_macro_f1 REAL,
    -- Revision stats
    n_revised INTEGER,
    revision_rate REAL,              -- n_revised / total
    revision_accuracy REAL,          -- of revised samples, fraction where new answer is correct
    -- Flip analysis
    n_saves INTEGER,                 -- original wrong -> final correct
    n_breaks INTEGER,                -- original correct -> final wrong
    net_improvement INTEGER,         -- saves - breaks
    -- Per-tool effectiveness
    tool_prior_call_count INTEGER,
    tool_prior_revision_rate REAL,   -- P(revise | tool_prior called)
    tool_confusion_call_count INTEGER,
    tool_confusion_revision_rate REAL,
    tool_scene_action_call_count INTEGER,
    tool_scene_action_revision_rate REAL,
    tool_waypoint_call_count INTEGER,
    tool_waypoint_revision_rate REAL,
    -- Latency
    mean_predict_time_ms REAL,
    mean_verify_time_ms REAL,
    mean_total_time_ms REAL,
    p95_total_time_ms REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(experiment_id)
);
```

## Implementation Phases

### Phase 0: Oracle Ceiling Test (RUN FIRST)
Run the oracle condition before building real tools. If the ceiling is < 0.70 F1, abort.

### Phase 1: Foundation (Schema + Tools + Scripts)
Build the DB schema, tool implementations, and all inference scripts.

### Phase 2: Prediction Phase
Run the fine-tuned model on all 8,613 samples (reuse greedy baseline from self-consistency if possible, or re-run).

### Phase 3: Verification Phase
Run the base model with tools across all 6 conditions.

### Phase 4: Analysis & Reporting
Compute flip analysis, revision rates, per-tool effectiveness, generate insight report.

## Team Orchestration

- You operate as the team lead and orchestrate the team to execute the plan.
- You're responsible for deploying the right team members with the right context to execute the plan.
- IMPORTANT: You NEVER operate directly on the codebase. You use `Task` and `Task*` tools to deploy team members to do the building, validating, testing, deploying, and other tasks.

### Team Members

- Builder
  - Name: foundation-builder
  - Role: Create SQLite schema, tool implementations, tool call parser, and all inference scripts
  - Agent Type: builder
  - Resume: true
  - Background: true

- Builder
  - Name: analysis-builder
  - Role: Create analysis script, per-sample report generator, and insight report generator
  - Agent Type: builder
  - Resume: true
  - Background: true

- Runner
  - Name: oracle-runner
  - Role: Run the oracle ceiling test FIRST. If F1 < 0.70, report back and halt.
  - Agent Type: runner
  - Resume: true
  - Background: false (critical gate -- must review before proceeding)

- Runner
  - Name: inference-runner
  - Role: Run all 6 experimental conditions sequentially on GPU
  - Agent Type: runner
  - Resume: true
  - Background: false (GPU runs need monitoring)

- Runner
  - Name: analysis-runner
  - Role: Execute analysis scripts after inference completes
  - Agent Type: runner
  - Resume: true
  - Background: true

- Builder
  - Name: insight-analyst
  - Role: Opus-powered per-sample insight generator (same framework as self-consistency)
  - Agent Type: builder
  - Resume: true
  - Background: false (final deliverable)

- Validator
  - Name: result-validator
  - Role: Validate DB integrity, row counts, and statistical analysis outputs
  - Agent Type: validator
  - Resume: true
  - Background: true

## Step by Step Tasks

### 0. Download Base Models & Optimize FSx Striping (2B + 8B)
- **Task ID**: download-base-models
- **Depends On**: none
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true (can run alongside build tasks)
- **Step A**: Set up FSx directory striping for max parallel I/O before downloading (stripe across all OSTs so large safetensor files load fast):
  ```bash
  # Set directory stripe to all OSTs — new files written here get max parallel reads
  lfs setstripe -c -1 /fsx/models/Qwen3-VL-2B-Instruct/
  lfs setstripe -c -1 /fsx/models/Qwen3-VL-8B-Instruct/
  ```
- **Step B**: Download 2B base model:
  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-2B-Instruct', local_dir='/fsx/models/Qwen3-VL-2B-Instruct')"
  ```
- **Step C**: Download 8B base model:
  ```bash
  python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/fsx/models/Qwen3-VL-8B-Instruct')"
  ```
- **Step D**: Re-stripe the fine-tuned checkpoint for fast loading (if not already striped):
  ```bash
  # Check current striping
  lfs getstripe /fsx/mketkar-sceneiq/checkpoints/checkpoint_006/model.safetensors
  # If stripe_count is low (e.g., 1 or 5), migrate to all OSTs
  lfs migrate -c -1 /fsx/mketkar-sceneiq/checkpoints/checkpoint_006/model.safetensors
  ```
- **Step E**: Verify striping applied and models load fast in vLLM:
  ```bash
  # Confirm striping
  lfs getstripe /fsx/models/Qwen3-VL-2B-Instruct/*.safetensors
  lfs getstripe /fsx/models/Qwen3-VL-8B-Instruct/*.safetensors
  # Test loading
  CUDA_VISIBLE_DEVICES=1 python -c "from vllm import LLM; llm = LLM(model='/fsx/models/Qwen3-VL-2B-Instruct', trust_remote_code=True, max_model_len=4096, enforce_eager=True); print('2B loaded OK')"
  CUDA_VISIBLE_DEVICES=2 python -c "from vllm import LLM; llm = LLM(model='/fsx/models/Qwen3-VL-8B-Instruct', trust_remote_code=True, max_model_len=4096, enforce_eager=True); print('8B loaded OK')"
  ```
- Record actual model paths in `tool_calling_experiment/config.json` for scripts to reference
- **Note**: `lfs` commands must be run on the host node that mounts FSx, not inside containers

### 1. Create SQLite Schema
- **Task ID**: create-schema
- **Depends On**: none
- **Assigned To**: foundation-builder
- **Agent Type**: builder
- **Parallel**: true
- Create `tool_calling_experiment/create_db.py` with the schema above
- Run it to create the empty DB

### 2. Build Tool Implementations
- **Task ID**: build-tools
- **Depends On**: none
- **Assigned To**: foundation-builder
- **Agent Type**: builder
- **Parallel**: true
- Create `tool_calling_experiment/tools.py`:
  - Tool definitions (OpenAI format JSON)
  - Tool implementations (Python functions)
  - Load statistics from self-consistency DB at import time
  - Oracle tool variants that accept ground truth
- Create `tool_calling_experiment/build_tools.py`:
  - Query self-consistency DB for class frequencies, confusion matrix, co-occurrence, waypoint stats
  - Save as JSON for tools.py to load (avoids DB dependency at inference time)
- Create `tool_calling_experiment/parse_tool_calls.py`:
  - Parse `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` from generated text
  - Handle malformed output gracefully (2B model may produce invalid JSON)
  - Return list of (tool_name, arguments) tuples

### 3. Build Prediction Script (Phase 2)
- **Task ID**: build-prediction-script
- **Depends On**: create-schema
- **Assigned To**: foundation-builder
- **Agent Type**: builder
- **Parallel**: true
- Create `tool_calling_experiment/run_prediction.py`:
  - Load fine-tuned model, run greedy inference on all 8,613 samples
  - Parse scene, action, waypoints from output
  - Store in predictions table with original_* fields
  - This is essentially the same as run_self_consistency.py with n=1, temp=0
  - OR: copy greedy results from self-consistency DB directly (faster)
  - CLI: `--reuse-greedy` flag to copy from self-consistency DB instead of re-running

### 4. Build Verification Script (Pipeline A)
- **Task ID**: build-verification-script
- **Depends On**: build-tools
- **Assigned To**: foundation-builder
- **Agent Type**: builder
- **Parallel**: true
- Create `tool_calling_experiment/run_verification.py`:
  - CLI: `--condition` (prior_only, confusion_only, all_tools, oracle) + `--verifier-model` (path) + `--gpu-id` (int, default 1)
  - Set `CUDA_VISIBLE_DEVICES={gpu_id}` at script start
  - Load specified base model (2B or 8B) on the designated GPU
  - For each sample: construct verification prompt with image + predictions + tool definitions
  - Use `llm.chat(messages, tools=tools)` for batched inference
  - Parse tool calls from output, execute tools locally, feed results back
  - Parse final predictions from model's revised output
  - Store in predictions table (final_* fields) and tool_calls table
  - `--condition oracle` replaces tool implementations with GT-returning variants
  - Handle multi-turn: first call may generate tool calls, second call generates final answer
  - Batch strategy: run first turn for all samples, collect tool calls, execute, run second turn
  - Experiment ID encodes verifier size: `tc_{condition}_{verifier_size}_{timestamp}` (e.g., `tc_all_tools_8b_20260216_120000`)

### 5. Build Staged Prediction Script (Pipeline B)
- **Task ID**: build-staged-script
- **Depends On**: build-tools
- **Assigned To**: foundation-builder
- **Agent Type**: builder
- **Parallel**: true
- Create `tool_calling_experiment/run_staged.py`:
  - Base model only, no fine-tuned model
  - Multi-turn: Turn 1 (scene + check_confusion_risk), Turn 2 (action + check_scene_action), Turn 3 (waypoints + check_waypoint)
  - Each turn is a separate `llm.chat()` call with tools
  - Parse structured output from each turn
  - Store results in same predictions + tool_calls tables

### 6. Build Analysis + Report Scripts
- **Task ID**: build-analysis
- **Depends On**: create-schema
- **Assigned To**: analysis-builder
- **Agent Type**: builder
- **Parallel**: true (can be built while other scripts are being built)
- Create `tool_calling_experiment/compute_analysis.py`:
  - Flip analysis: per condition, count saves/breaks/net
  - Revision rate: per condition, how often does model revise?
  - Revision accuracy: when revised, is new answer better?
  - Per-tool effectiveness: conditional probability of revision given each tool's output
  - Per-class breakdown: which scene classes benefit most from tools?
  - Latency analysis: mean/p95/p99 per condition
  - Store aggregates in condition_metrics table
  - Print comprehensive stdout summary
- Create `tool_calling_experiment/generate_sample_report.py`:
  - Same JSONL format as self-consistency experiment
  - Per sample: original prediction, each tool call and result, final prediction, revision decision, GT, flip status
  - Diagnostic tags: tool_saved, tool_broke, revised_correctly, revised_incorrectly, ignored_tool_warning, all_tools_agreed, etc.
- Create `tool_calling_experiment/generate_insights_report.py`:
  - Same 10-section format as self-consistency SAMPLE_INSIGHTS_REPORT.md
  - Sections adapted for tool calling: "Samples Where Tools Saved", "Samples Where Tools Broke", "Samples Where Model Ignored Tool Warning", "Per-Tool Effectiveness", etc.

### 7. Validate All Scripts
- **Task ID**: validate-scripts
- **Depends On**: all build tasks (1-6)
- **Assigned To**: result-validator
- **Agent Type**: validator
- **Parallel**: false
- Verify all scripts compile and have consistent schema usage
- Verify tool definitions match between tools.py and verification scripts
- Verify parse_tool_calls handles edge cases

### 8. Build Tool Statistics from Data
- **Task ID**: build-tool-stats
- **Depends On**: build-tools, validate-scripts
- **Assigned To**: inference-runner (or analysis-runner)
- **Agent Type**: runner
- **Parallel**: false
- Run: `python tool_calling_experiment/build_tools.py`
- Verify output JSON has all required statistics
- This must complete before any verification runs

### 9. Run Prediction Phase (reuse greedy)
- **Task ID**: run-predictions
- **Depends On**: build-tool-stats, build-prediction-script
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: false
- Run: `CUDA_VISIBLE_DEVICES=0 python tool_calling_experiment/run_prediction.py --reuse-greedy`
- Copies greedy baseline from self-consistency DB into tool_calling DB predictions table
- Verify: 8,613 rows in predictions with original_* fields populated

### 10. Run Oracle-2B (CRITICAL GATE)
- **Task ID**: run-oracle-2b
- **Depends On**: run-predictions, download-base-models
- **Assigned To**: oracle-runner
- **Agent Type**: runner
- **Parallel**: true (can run in parallel with oracle-8b on different GPU)
- Run: `python tool_calling_experiment/run_verification.py --condition oracle --verifier-model /fsx/models/Qwen3-VL-2B-Instruct --gpu-id 1`
- Measure oracle F1 with 2B verifier
- Report: scene accuracy, revision rate, net flips

### 11. Run Oracle-8B (CRITICAL GATE)
- **Task ID**: run-oracle-8b
- **Depends On**: run-predictions, download-base-models
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true (runs on GPU 2 while oracle-2b runs on GPU 1)
- Run: `python tool_calling_experiment/run_verification.py --condition oracle --verifier-model /fsx/models/Qwen3-VL-8B-Instruct --gpu-id 2`
- Measure oracle F1 with 8B verifier
- **GATE CHECK**: If BOTH oracle F1s < 0.70, **STOP**. If either >= 0.70, **PROCEED**.
- Report: scene accuracy, revision rate, net flips, comparison 2B vs 8B

### 12. Run Prior-Only-2B (Pipeline A)
- **Task ID**: run-prior-only-2b
- **Depends On**: run-oracle-2b (must pass gate)
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true (can run in parallel with 8B variant on different GPU)
- Run: `python tool_calling_experiment/run_verification.py --condition prior_only --verifier-model /fsx/models/Qwen3-VL-2B-Instruct --gpu-id 1`
- Only Tool 1 (check_scene_prior) enabled

### 13. Run Prior-Only-8B (Pipeline A)
- **Task ID**: run-prior-only-8b
- **Depends On**: run-oracle-8b (must pass gate)
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true (runs on GPU 2 alongside 2B on GPU 1)
- Run: `python tool_calling_experiment/run_verification.py --condition prior_only --verifier-model /fsx/models/Qwen3-VL-8B-Instruct --gpu-id 2`

### 14. Run Confusion-Only-2B (Pipeline A)
- **Task ID**: run-confusion-only-2b
- **Depends On**: run-prior-only-2b
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true
- Run: `python tool_calling_experiment/run_verification.py --condition confusion_only --verifier-model /fsx/models/Qwen3-VL-2B-Instruct --gpu-id 1`

### 15. Run Confusion-Only-8B (Pipeline A)
- **Task ID**: run-confusion-only-8b
- **Depends On**: run-prior-only-8b
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true
- Run: `python tool_calling_experiment/run_verification.py --condition confusion_only --verifier-model /fsx/models/Qwen3-VL-8B-Instruct --gpu-id 2`

### 16. Run All-Tools-2B (Pipeline A)
- **Task ID**: run-all-tools-2b
- **Depends On**: run-confusion-only-2b
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true
- Run: `python tool_calling_experiment/run_verification.py --condition all_tools --verifier-model /fsx/models/Qwen3-VL-2B-Instruct --gpu-id 1`

### 17. Run All-Tools-8B (Pipeline A)
- **Task ID**: run-all-tools-8b
- **Depends On**: run-confusion-only-8b
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true
- Run: `python tool_calling_experiment/run_verification.py --condition all_tools --verifier-model /fsx/models/Qwen3-VL-8B-Instruct --gpu-id 2`

### 18. Run Staged-2B (Pipeline B)
- **Task ID**: run-staged-2b
- **Depends On**: run-all-tools-2b
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true
- Run: `python tool_calling_experiment/run_staged.py --verifier-model /fsx/models/Qwen3-VL-2B-Instruct --gpu-id 1`

### 19. Run Staged-8B (Pipeline B)
- **Task ID**: run-staged-8b
- **Depends On**: run-all-tools-8b
- **Assigned To**: inference-runner
- **Agent Type**: runner
- **Parallel**: true
- Run: `python tool_calling_experiment/run_staged.py --verifier-model /fsx/models/Qwen3-VL-8B-Instruct --gpu-id 2`

### 20. Run Analysis
- **Task ID**: run-analysis
- **Depends On**: run-staged-2b, run-staged-8b (all conditions complete)
- **Assigned To**: analysis-runner
- **Agent Type**: runner
- **Parallel**: false
- Run compute_analysis.py, generate_sample_report.py
- Key output: 2B vs 8B comparison across all conditions
- Verify all outputs

### 21. Generate Insight Report
- **Task ID**: generate-insights
- **Depends On**: run-analysis
- **Assigned To**: insight-analyst
- **Agent Type**: builder (opus)
- **Parallel**: true (can run alongside stats)

### 22. Statistical Deep-Dive
- **Task ID**: stats-deep-dive
- **Depends On**: run-analysis
- **Assigned To**: stats-analyst
- **Agent Type**: experiment_stats_analyst
- **Parallel**: true (can run alongside insights)

### 23. Final Validation
- **Task ID**: validate-all
- **Depends On**: generate-insights, stats-deep-dive
- **Assigned To**: result-validator
- **Agent Type**: validator
- **Parallel**: false

## Experimental Conditions Summary

Test both 2B and 8B as verifiers to measure how much model size matters for tool-calling effectiveness.

| Condition | Pipeline | Predictor | Verifier | Tools | Primary Question |
|---|---|---|---|---|---|
| 0: Baseline | -- | Fine-tuned 2B | -- | None | Existing greedy results |
| 1a: Oracle-2B | A | Fine-tuned 2B | Base 2B | GT tools | Ceiling with 2B verifier |
| 1b: Oracle-8B | A | Fine-tuned 2B | Base 8B | GT tools | Ceiling with 8B verifier |
| 2a: Prior-only-2B | A | Fine-tuned 2B | Base 2B | Tool 1 | Does base rate info fix incident_zone? (2B) |
| 2b: Prior-only-8B | A | Fine-tuned 2B | Base 8B | Tool 1 | Does base rate info fix incident_zone? (8B) |
| 3a: Confusion-only-2B | A | Fine-tuned 2B | Base 2B | Tool 4 | Does confusion awareness help? (2B) |
| 3b: Confusion-only-8B | A | Fine-tuned 2B | Base 8B | Tool 4 | Does confusion awareness help? (8B) |
| 4a: All-tools-2B | A | Fine-tuned 2B | Base 2B | Tools 1-4 | Full verify-and-revise (2B) |
| 4b: All-tools-8B | A | Fine-tuned 2B | Base 8B | Tools 1-4 | Full verify-and-revise (8B) |
| 5a: Staged-2B | B | -- | Base 2B | Tools 1-4 | Can tools replace fine-tuning? (2B) |
| 5b: Staged-8B | B | -- | Base 8B | Tools 1-4 | Can tools replace fine-tuning? (8B) |

**Key comparison axis**: For every condition, 2B vs 8B verifier tells us:
- Does the 8B model follow tool guidance better (higher revision rate)?
- Does the 8B model produce more accurate revisions?
- Is the oracle ceiling higher with 8B (can it use perfect info better)?
- At what point does verifier size stop mattering?

**GPU assignment**: Fine-tuned model on GPU 0, Base 2B on GPU 1, Base 8B on GPU 2. All three can be loaded simultaneously.

## Acceptance Criteria

1. **SQLite DB created** at `tool_calling_experiment/tool_calling.db` with schema matching design
2. **Both base models downloaded** and verified loading on separate GPUs (2B on GPU 1, 8B on GPU 2)
3. **Oracle ceilings measured** for both 2B and 8B — F1 reported, go/no-go decision documented
4. **11 conditions completed** (baseline + 5 conditions × 2 verifier sizes, or fewer if oracle gate fails)
5. **Per-condition metrics**: scene_accuracy, macro_f1, revision_rate, revision_accuracy, net_improvement
6. **2B vs 8B comparison**: for each condition, direct comparison of verifier model sizes
7. **Per-tool effectiveness**: conditional revision probability for each tool, stratified by verifier size
8. **Net flip analysis**: per condition, saves vs breaks vs baseline
9. **Per-sample report data** with full tool call traces
10. **Insight report** with elementary-teacher explanations, including 2B-vs-8B analysis
11. **Decision**: is tool-calling viable for this model/task? Does verifier size matter? Clear recommendation.

## Validation Commands

```bash
# Verify DB
python3 -c "
import sqlite3
conn = sqlite3.connect('tool_calling_experiment/tool_calling.db')
for row in conn.execute('SELECT experiment_id, condition_name, scene_accuracy, revision_rate, n_saves, n_breaks, net_improvement FROM condition_metrics'):
    print(row)
conn.close()
"

# Verify tool call traces
python3 -c "
import sqlite3
conn = sqlite3.connect('tool_calling_experiment/tool_calling.db')
print('Tool calls per condition:')
for row in conn.execute('SELECT experiment_id, tool_name, COUNT(*), AVG(model_revised_after) FROM tool_calls GROUP BY experiment_id, tool_name'):
    print(row)
conn.close()
"

# Verify scripts compile
for f in create_db build_tools tools parse_tool_calls run_prediction run_verification run_staged run_oracle compute_analysis generate_sample_report; do
    python -m py_compile tool_calling_experiment/$f.py && echo "OK: $f" || echo "FAIL: $f"
done

# Verify reports
wc -l tool_calling_experiment/sample_report_data.json
wc -l tool_calling_experiment/TOOL_CALLING_INSIGHTS_REPORT.md
```

## Decision Points

After running all conditions, one of these outcomes:

1. **Oracle ceiling low (< 0.70 F1)**: STOP. Model cannot self-correct even with perfect info. Need retraining.
2. **Oracle high, real tools don't help**: Tools aren't informative enough. Iterate on tool design.
3. **Real tools help but model rarely revises**: Model ignores tool output. Consider rule-based post-processing instead.
4. **Real tools help and model revises correctly**: Viable path. Optimize which tools matter most.

## Notes

- **Base model download**: `Qwen/Qwen3-VL-2B-Instruct` weights (~4.6 GB) may need to be downloaded from HuggingFace. Tokenizer is already cached locally.
- **Multi-turn latency**: Each verification turn re-tokenizes the full conversation. Pipeline A roughly doubles inference time. Pipeline B roughly triples it.
- **2B model tool calling quality**: Expect imperfect JSON in tool calls. Parser must handle malformed output gracefully.
- **VLLM_BATCH_INVARIANT=1**: Set for prediction phase (deterministic forward pass). Verification phase doesn't need it since we're not comparing across batch sizes.
- **Reuse greedy baseline**: The self-consistency DB already has 8,613 greedy predictions. Copy them rather than re-running.
