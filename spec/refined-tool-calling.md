# Refined Spec: Tool-Calling Verification Experiment for SceneIQ

**Created**: 2026-02-16
**Status**: Phase 1 complete, Phase 2 in progress
**Dataset**: 8,613 dashcam samples from SceneIQ MDS dataset at `/fsx/mketkar-sceneiq/datasets/`
**Hardware**: 8x H200 GPUs (144 GB each)

---

## Table of Contents

- [Part 0: Prerequisite Discovery (COMPLETED)](#part-0-prerequisite-discovery-completed)
- [Part 1: Infrastructure](#part-1-infrastructure)
- [Part 2: Visual Tools](#part-2-visual-tools)
- [Part 3: Phase 1 -- Mechanical Validation (COMPLETED)](#part-3-phase-1----mechanical-validation-completed)
- [Part 4: Phase 2 -- Single Tool Deep Dives](#part-4-phase-2----single-tool-deep-dives)
- [Part 5: Phase 3 -- Multi-Tool Interaction](#part-5-phase-3----multi-tool-interaction)
- [Part 6: Phase 4 -- Stress Tests](#part-6-phase-4----stress-tests)
- [Part 7: Phase 5 -- Synthesis](#part-7-phase-5----synthesis)
- [Part 8: Path A vs Path B Comparison](#part-8-path-a-vs-path-b-comparison)
- [Appendix A: Files Created](#appendix-a-files-created)
- [Appendix B: System Prompts](#appendix-b-system-prompts)
- [Appendix C: Sample Selection Criteria](#appendix-c-sample-selection-criteria)
- [Appendix D: Key Numbers Reference](#appendix-d-key-numbers-reference)

---

## Part 0: Prerequisite Discovery (COMPLETED)

Nine prerequisite experiments were completed before Phase 1 began. Each one informed the design of the tool-calling pipeline. Results are documented here with exact numbers.

### Step 0.1: Calibration Baseline (COMPLETED)

**File**: `tool_calling_experiment/calibration_results.json`
**Script**: `tool_calling_experiment/calibration_analysis.py`

**Problem**: The fine-tuned 2B model predicts `incident_zone` at 46.8% (4,081 predictions) vs the true rate of 3.7% (322 samples) -- a 12.5x over-prediction. This accounts for 85.6% of all errors (3,918 of 4,577 total errors).

**Ground truth class distribution**:

| Scene Class | Count | Percentage |
|---|---|---|
| nominal | 6,741 | 78.3% |
| flagger | 660 | 7.7% |
| flooded | 618 | 7.2% |
| incident_zone | 322 | 3.7% |
| mounted_police | 272 | 3.2% |

**Prediction distribution (greedy)**:

| Predicted | Count | Percentage |
|---|---|---|
| incident_zone | 4,081 | 47.4% |
| nominal | 3,691 | 42.9% |
| flagger | 451 | 5.2% |
| flooded | 284 | 3.3% |
| mounted_police | 106 | 1.2% |

**Strategies tested (deterministic post-processing)**:

| Strategy | Accuracy | Macro F1 | Saves | Breaks | Net |
|---|---|---|---|---|---|
| S0: Baseline (greedy) | 46.9% | 0.429 | -- | -- | -- |
| S1: Naive flip (all IZ -> nominal) | 84.8% | 0.474 | 3,430 | 163 | +3,267 |
| S2: Prior threshold 5% (IZ+MP -> nominal) | 84.5% | 0.424 | 3,449 | 209 | +3,240 |
| S3: Confusion-aware (IZ -> nominal, except GT fine_class=IZ) | 86.7% | 0.610 | 3,430 | 0 | +3,430 |
| S4: Oracle ceiling (perfect IZ corrections using GT) | 92.4% | 0.775 | 3,918 | 0 | +3,918 |
| S5: Selective fine_class (IZ+nominal_triggers -> nominal) | 63.7% | 0.470 | 1,450 | 0 | +1,450 |

**Per-class F1 at oracle ceiling (S4)**:
- nominal: 0.958 (P=0.932, R=0.985)
- flagger: 0.913 (P=0.901, R=0.926)
- flooded: 0.757 (P=0.891, R=0.659)
- incident_zone: 0.672 (P=1.000, R=0.506)
- mounted_police: 0.575 (P=0.691, R=0.493)

**Confusion matrix (major errors, GT -> Predicted)**:
- nominal -> incident_zone: 3,430 (50.9% of nominal)
- flagger -> incident_zone: 227 (34.4% of flagger)
- flooded -> incident_zone: 173 (28.0% of flooded)
- mounted_police -> nominal: 138 (50.7% of mounted_police)
- incident_zone -> nominal: 132 (41.0% of incident_zone)
- mounted_police -> incident_zone: 88 (32.4% of mounted_police)

**Fine class root cause**:
- `nominal_triggers` fine_class: 2,801 samples, only 19.0% accuracy
- These are nominal scenes containing visual triggers (traffic cones, barriers) that the model misclassifies as incidents

**Conclusions**:
1. 85.6% of errors come from IZ over-prediction -- the single dominant failure mode
2. Naive flip achieves 84.8% accuracy but kills IZ recall entirely (F1=0.00)
3. Oracle IZ classifier ceiling: 86.7% accuracy, 61.0% macro F1 (confusion-aware S3)
4. True oracle ceiling (S4): 92.4% accuracy, 77.5% macro F1
5. Gap between S3 and S4 (0.165 F1 points) requires distinguishing nominal_triggers from real incidents -- a perceptual task no statistical tool can perform

### Step 0.2: Conditioned Prediction Test (COMPLETED)

**File**: `tool_calling_experiment/conditioned_prediction_results.json`
**Script**: `tool_calling_experiment/run_conditioned_prediction.py`

**Experiment**: Fed the fine-tuned model its own predictions plus the correction "The correct scene is nominal. Please re-predict." on 20 false-IZ samples (all GT=nominal, fine_class=negative_incident_zone).

**Results**:
- Scene changed: 0/20 (0.0%) -- **the model COMPLETELY IGNORES the correction**
- Actions changed: 11/20 (55.0%) -- actions are malleable even when scene is locked
- Follows correction: 0/13 IZ predictions (0.0%)
- Ignores correction: 13/13 IZ predictions (100.0%)
- The ODD token is perception-locked: the model's scene classification is baked into the forward pass and cannot be overridden by text prompts

**Conclusion**: Cannot fix the fine-tuned model via prompting. The scene token is determined by the visual encoder's representation, not by textual context. Tool calling with the fine-tuned model as consumer is futile for scene revision. Any tool-calling pipeline MUST use a base model as the tool consumer.

### Step 0.3: Base Model Zero-Shot (COMPLETED)

**File**: `tool_calling_experiment/base_model_zeroshot_results.json`
**Script**: `tool_calling_experiment/base_model_zeroshot_test.py`

**Experiment**: Tested both base models (2B and 8B) on 10 diverse samples with a zero-shot prompt that includes base rate information and scene descriptions.

**Results**:

| Model | Accuracy | IZ Predictions | IZ Actual | Key Failure |
|---|---|---|---|---|
| Base 2B | 7/10 (70%) | 1 | 1 | Misses flagger (0/2) |
| Base 8B | 4/10 (40%) | 1 | 1 | Strong nominal bias (8/10 predict nominal) |
| Fine-tuned 2B | 10/10 (100%) | -- | -- | On these 10 selected samples |

**Per-class detail**:

| Model | nominal (4) | flagger (2) | flooded (2) | incident_zone (1) | mounted_police (1) |
|---|---|---|---|---|---|
| Base 2B | 4/4 | 0/2 | 2/2 | 0/1 | 1/1 |
| Base 8B | 3/4 | 0/2 | 0/2 | 0/1 | 1/1 |

**Key findings**:
- Neither base model has IZ over-prediction -- fine-tuning created that bias
- Both base models understand the 5-class taxonomy perfectly (100% valid scene names, zero hallucinated classes)
- 2B outperforms 8B on zero-shot because 8B has a stronger nominal prior (predicts nominal 80% of the time)
- The 8B model predicted IZ only 1 time (for a nominal_triggers sample), matching the true rate rather than over-predicting
- Both base models are viable as tool-calling verifiers

**Conclusion**: Base model is the right consumer for tool output. It does not have the perception-locked bias of the fine-tuned model. The 8B model's nominal bias is the OPPOSITE problem from the fine-tuned model -- this motivates the Path A vs Path B comparison.

### Step 0.4: Weights vs Tools Analysis (COMPLETED)

**File**: `tool_calling_experiment/lesson1_weights_vs_tools.md`

**Systematic analysis of 15 knowledge components across scene classification, action prediction, and waypoint prediction**:

| Knowledge Component | Category | Justification |
|---|---|---|
| Visual appearance of each scene | Perceptual (weights) | Must look at the image |
| nominal_triggers vs real incidents | Perceptual (weights) | Contextual visual judgment |
| Base rates per class | Statistical (tool) | Static facts, bad to memorize |
| Confusion patterns | Statistical (tool) | Meta-knowledge about model behavior |
| Visual similarity between classes | Perceptual (weights) | Property of the images |
| Contextual scene cues | Perceptual (weights) | Visual features from the image |
| Scene-action compatibility | Post-processing | Deterministic rules, not suggestions |
| Longitudinal response recognition | Perceptual (weights) | Visual judgment |
| Lateral response recognition | Perceptual (weights) | Visual judgment |
| Action severity calibration | Hybrid (weights + tool) | Perception + statistical prior |
| HLA physical thresholds | Statistical (tool) | Engineering constants |
| Road geometry and lane structure | Perceptual (weights) | Visual spatial reasoning |
| Typical waypoint ranges | Statistical (tool) | Distribution summaries |
| Physical feasibility constraints | Post-processing | Kinematic checks |
| Obstacle avoidance paths | Perceptual (weights) | Spatial reasoning about the image |

**Cross-tool assessment**:

| Tool | Genuine Tool? | Better as Prompt? | Better as Post-Processing? | Pedagogical Value |
|---|---|---|---|---|
| check_scene_prior | Marginal | Yes (for 5 classes) | No | Medium |
| check_scene_action_compatibility | No | No | Yes (deterministic rule) | High |
| check_waypoint_feasibility | No | No | Yes (statistical check) | Medium |
| check_confusion_risk | **Yes** | Partially | No | High |

**Optimal split**: Perception in the model, meta-cognition in tools, deterministic constraints in post-processing.

**The key distinction**: A tool is useful when the model needs to REASON about the information to decide what to do. Post-processing suffices when the correct action is deterministic given the inputs. `check_confusion_risk` is the only genuinely useful statistical tool because it provides meta-cognitive information the model cannot derive from a single image.

### Step 0.5: Tool Choice Conditions (COMPLETED -- Lesson 3)

**Script**: `tool_calling_experiment/lesson3_tool_choice.py`

**Experiment**: Tested under what conditions the model decides to call tools vs skip them.

**Key findings**:
- Both 2B and 8B reliably call tools when available (tool call rate near 100%)
- 8B makes more targeted tool selections (prefers check_scene_prior and check_confusion_risk first)
- 2B scattershots across all tools, including check_waypoint_feasibility which is the least useful
- Neither model refuses to call tools -- the bottleneck is whether they ACT on results, not whether they call tools

### Step 0.6: Contradictory Tool Results (COMPLETED -- Lesson 4)

**File**: `tool_calling_experiment/lesson4_results.json`
**Script**: `tool_calling_experiment/lesson4_contradictory_tools.py`

**Experiment**: 30 samples with the 8B model. Tools explicitly contradict the model's IZ prediction (prior says 3.7%, confusion says 50.9% error rate). Measured whether the model revises.

**Results**:
- Behavior: 100% ACTUALLY_REASONS (model engages with tool output), 0% IGNORES
- Scene revision rate: only 7% despite receiving strong contradictory evidence
- The model uses action compatibility as an anchor: "The action prediction (stop, lc_left) is compatible with incident_zone, so the scene prediction is probably correct"
- Circular reasoning detected: model cites its own predicted actions as evidence for its predicted scene

**Key insight**: The model genuinely processes tool output (unlike the fine-tuned model which ignores it completely), but uses sophisticated reasoning to resist revision. This is a qualitatively different failure from ignoring -- the model ENGAGES but ANCHORS. The model treats its own predictions as additional evidence, creating a self-reinforcing loop where tool output is discounted relative to the prediction it was supposed to evaluate.

### Step 0.7: Tool Chain Coherence (COMPLETED -- Lesson 5)

**File**: `tool_calling_experiment/lesson5_results.json`
**Script**: `tool_calling_experiment/lesson5_tool_chain.py`

**Experiment**: Multi-turn tool chains where contradictory information arrives mid-chain vs post-prediction. Used 10 diverse samples (nominal, incident_zone, flooded, flagger, mounted_police) with text descriptions.

**Results**:
- 100% tool integration across 4-turn chains (model references all prior tool outputs)
- Chain order does not affect final answer (robust to permutation)
- **Contradiction mid-chain causes revision 2/3 of the time (66.7%)**
- **Contradiction post-prediction causes revision 1/15 of the time (6.7%)**
- 10x difference in revision effectiveness: mid-chain vs post-prediction

**Key insight**: Tools are 10x more effective DURING reasoning than AFTER. The architecture must present tool information BEFORE the model commits to a prediction, not after. This directly shapes the design:
- Pipeline B (staged prediction with tools) is strongly preferred over Pipeline A (predict-then-verify)
- Tool results should be interleaved with reasoning, not appended as an afterthought
- The commitment effect is real: once a model outputs a prediction, it treats that prediction as an anchor

### Step 0.8: 4-Level Tool Reasoning Experiment (COMPLETED)

**File**: `tool_calling_experiment/tool_levels_results.json`
**Script**: `tool_calling_experiment/lesson_tool_levels.py`

**Experiment**: 20 samples x 4 reasoning levels x 2 model sizes = 160 runs. The 20 samples were stratified: 10 false-IZ, 3 true-IZ, 3 true-nominal, 2 wrong-other, 2 correct-non-nominal.

**Reasoning levels**:
- L1: Raw data only (statistics, no interpretation)
- L2: Interpreted data (statistics + natural language interpretation + diagnostic questions)
- L3: Procedural (visual checklist and step-by-step decision procedure)
- L4: Prescriptive (explicit if/then decision rules)

**Results**:

| Model | Level | Accuracy | Revision Rate | Saves | Breaks | Net | Parseable |
|---|---|---|---|---|---|---|---|
| 2B | L1 | 40.0% | 0.0% | 0 | 0 | 0 | 20/20 |
| 2B | L2 | **68.8%** | 55.0% | 8 | 3 | **+5** | 16/20 |
| 2B | L3 | 55.6% | 35.0% | 4 | 1 | +3 | 18/20 |
| 2B | L4 | 40.0% | 0.0% | 0 | 0 | 0 | 20/20 |
| 8B | L1 | 36.8% | 0.0% | 0 | 0 | 0 | 19/20 |
| 8B | L2 | **75.0%** | 65.0% | 10 | 3 | **+7** | 20/20 |
| 8B | L3 | 75.0% | 65.0% | 10 | 3 | +7 | 20/20 |
| 8B | L4 | 75.0% | 65.0% | 10 | 3 | +7 | 20/20 |

**Per-category breakdown at 8B L2**:
- true_nominal (3): 3/3 correct, 0 revisions (correctly stable)
- true_incident_zone (3): 0/3 correct, 3 revisions all wrong (overcorrection)
- false_incident_zone (10): **10/10 correct**, 10 revisions all right (perfect save rate)
- wrong_flooded_flagger (2): 0/2 correct, 0 revisions (missed opportunity)
- correct_non_nominal (2): 2/2 correct, 0 revisions (correctly stable)

**Key findings**:
1. **L1 (raw stats) is USELESS for both models** -- zero revisions at L1. Raw numbers do not trigger reasoning.
2. **L2 (interpreted) is THE inflection point** -- 8B jumps from 36.8% to 75.0% accuracy. Natural language interpretation is what unlocks the model's ability to act on statistical information.
3. **2B has a narrow L2 sweet spot** -- degrades at L3 (55.6%) and collapses at L4 (40.0%, zero revisions). Procedural instructions confuse the smaller model.
4. **8B is stable from L2 onward** -- once it gets interpreted data, L3 and L4 add neither benefit nor harm. All three achieve 75.0%.
5. **Overcorrection is the universal failure mode**: All successful configurations break all 3 true-IZ samples. The statistical tools cannot distinguish true IZ from false IZ because the information is about population statistics, not individual images.
6. **Text-only tool ceiling is 75.0%** on this 20-sample mix (40% baseline). Visual grounding is needed to break through by providing per-image evidence.

**Conclusion**: Use L2 tools for both models. The 8B model is the stronger tool caller. The text-only ceiling of 75% motivates visual tool development.

### Step 0.9: 10-Sample Mechanical Proof (COMPLETED)

**File**: `tool_calling_experiment/tool_loop_results.json`
**Script**: `tool_calling_experiment/run_tool_loop.py`

**Experiment**: Full tool-calling loop on 10 samples (5 false-IZ, 3 true-IZ, 2 true-nominal) with the 8B model at L3 tools, max 5 tool rounds, via OpenAI-compatible API.

**Results**:
- 10/10 completed successfully, 0 failures
- Average 2.8 tool calls per sample (min 2, max 3)
- 100% valid JSON in all tool calls
- 7/10 revised (revision rate 70%)
- Revision accuracy: 71.4% (5/7 correct revisions)
- **5/5 false-IZ correctly revised to nominal** (100% save rate on false positives)
- **2/2 true-IZ wrongly revised to nominal** (100% break rate on true positives)
- 1/3 true-IZ correctly maintained
- 2 true-nominal correctly left unchanged
- Final accuracy: 7/10 (up from 4/10 baseline)
- Net improvement: +3 (5 saves - 2 breaks)

**Mechanics verified**:
- Tool call JSON parsing: 100% valid across 28 total tool calls
- Multi-turn conversation flow: works reliably
- Tool result injection: works reliably
- Final prediction parsing (FINAL_SCENE format): works reliably
- No infinite loops, no malformed states, no crashes

**The overcorrection problem confirmed**: The tool loop successfully fixes false-IZ predictions but also incorrectly "fixes" true-IZ predictions. The tools provide population-level evidence (IZ is rare, IZ is often confused with nominal) that is correct on average but wrong for actual IZ samples. This is the fundamental trade-off of statistical tool calling: it helps the majority (false-IZ is 10x more common than true-IZ) but hurts the minority.

---

## Part 1: Infrastructure

### Model Serving Configuration

All models served via vLLM's OpenAI-compatible HTTP server. Common flags for all servers:

```bash
--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 8192 --trust-remote-code --enforce-eager
```

| Model | Path | GPU(s) | Port | TP | VRAM | Notes |
|---|---|---|---|---|---|---|
| Fine-tuned 2B | `/workspace/vllm/models/checkpoint/` | GPU 0 | 8300 | 1 | ~4.6 GB | SceneIQ checkpoint |
| Base 2B | `/fsx/models/Qwen3-VL-2B-Instruct` | GPU 1 | 8301 | 1 | ~4.6 GB | Downloaded |
| Base 8B | `/fsx/models/Qwen3-VL-8B-Instruct` | GPU 2 | 8302 | 1 | ~16 GB | Downloaded |
| Base 72B | `/fsx/models/Qwen3-VL-72B-Instruct` | GPUs 3-6 | 8303 | 4 | ~140 GB | Download in progress |

**Server launch commands**:

```bash
# Fine-tuned 2B (prediction only, no tool calling needed)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model /workspace/vllm/models/checkpoint/ \
    --port 8300 --max-model-len 8192 --trust-remote-code --enforce-eager

# Base 2B (tool calling verifier)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model /fsx/models/Qwen3-VL-2B-Instruct \
    --port 8301 --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --trust-remote-code --enforce-eager

# Base 8B (tool calling verifier)
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model /fsx/models/Qwen3-VL-8B-Instruct \
    --port 8302 --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --trust-remote-code --enforce-eager

# Base 72B (tool calling verifier, 4-GPU tensor parallel)
CUDA_VISIBLE_DEVICES=3,4,5,6 python -m vllm.entrypoints.openai.api_server \
    --model /fsx/models/Qwen3-VL-72B-Instruct \
    --port 8303 --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 --trust-remote-code --enforce-eager \
    --tensor-parallel-size 4
```

**72B considerations**:
- Requires `--tensor-parallel-size 4` across 4 H200 GPUs (GPUs 3-6)
- Expected to be the strongest tool caller
- Key hypothesis: 72B can use L1 raw data tools effectively where 8B needed L2 interpreted data
- If true, this would demonstrate that model scale can substitute for tool scaffolding
- Download is ~140 GB; requires FSx striping (`lfs setstripe -c -1`) for fast loading

### Image Configuration

All dashcam images from the MDS dataset are loaded at **504 x 336 pixels** (camera `image_0003`, most recent frame). This resolution is:
- Large enough for the vision encoder to extract meaningful features
- Small enough to fit comfortably within the 8192 token context window with tool history
- Stored as JPEG (not PNG -- PNG causes vLLM image injection failures, bug fixed in Phase 1)

### Database

SQLite database at `tool_calling_experiment/tool_calling.db` with WAL mode and 5000ms busy timeout.

**Tables**:
- `experiments` -- one row per experimental condition (condition name, pipeline type, models, tools enabled, timing)
- `predictions` -- one row per sample per condition (original prediction, final prediction, revision tracking, ground truth, flip analysis)
- `tool_calls` -- one row per tool call per sample per condition (tool name, arguments, result, whether model revised after)
- `condition_metrics` -- aggregate metrics per condition (accuracy, F1, revision rates, per-tool effectiveness, latency)

Schema created by `tool_calling_experiment/create_db.py`.

### Data Sources

| Data | Path | Description |
|---|---|---|
| MDS dataset | `/fsx/mketkar-sceneiq/datasets/` or `/workspace/vllm/models/dataset/` | 8,613 samples with images + metadata |
| Self-consistency DB | `/workspace/vllm/self_consistency_experiment/self_consistency.db` | Greedy predictions, majority votes, per-sample analysis |
| Fine-tuned checkpoint | `/workspace/vllm/models/checkpoint/` (symlink to `/fsx/mketkar-sceneiq/checkpoints/checkpoint_006/`) | SceneIQ fine-tuned Qwen3-VL-2B |
| Tool statistics | `tool_calling_experiment/tool_stats.json` | Pre-computed class frequencies, confusion pairs, co-occurrence |

---

## Part 2: Visual Tools

Four visual tools built in `tool_calling_experiment/visual_tools.py`, complementing the four statistical tools in `tool_calling_experiment/tools_v2.py`.

### Tool 1: zoom_region

**Implementation**: PIL crop around (center_x, center_y) + 4x LANCZOS upscale
**Output format**: JPEG file path + pixel bounds + 63x63 grid mapping
**Purpose**: Allow the model to inspect specific regions of the dashcam image at higher resolution. Particularly useful for distinguishing traffic cones in a construction zone (nominal_triggers) from cones around a crash scene (incident_zone).

```python
def zoom_region(image_path: str, center_x: int, center_y: int, crop_size: int = 128) -> dict:
    """Returns: {
        "zoomed_image": "/tmp/sceneiq_visual_tools/sceneiq_zoom_XXXX.jpg",
        "original_region": [x1, y1, x2, y2],
        "grid_region": {"row_start": N, "row_end": N, "col_start": N, "col_end": N},
        "upscale_factor": 4,
        "zoomed_size": [crop_w*4, crop_h*4]
    }"""
```

**OpenAI function schema**:
```json
{
    "type": "function",
    "function": {
        "name": "zoom_region",
        "description": "Crop and enlarge a region of the driving scene image for closer inspection. Returns an enlarged view (4x) of the specified area. The image is 504 pixels wide (x: 0-503) and 336 pixels tall (y: 0-335).",
        "parameters": {
            "type": "object",
            "properties": {
                "center_x": {"type": "integer", "description": "X pixel coordinate of center (0=left edge, 503=right edge)"},
                "center_y": {"type": "integer", "description": "Y pixel coordinate of center (0=top edge, 335=bottom edge)"},
                "crop_size": {"type": "integer", "description": "Square crop size in pixels (32-256, default 128)"}
            },
            "required": ["center_x", "center_y"]
        }
    }
}
```

### Tool 2: visualize_waypoint

**Implementation**: PIL ImageDraw -- red circle (r=12) + white crosshairs (length=20) + text label
**Output format**: JPEG annotated image path + pixel coordinates + region description (3x3 named grid: "top-left", "center", "bottom-right", etc.)
**Purpose**: Help the model assess whether predicted waypoints are physically reasonable by showing them spatially on the driving scene.

```python
def visualize_waypoint(image_path: str, grid_row: int, grid_col: int,
                       label: str = "predicted") -> dict:
    """Returns: {
        "annotated_image": "/tmp/sceneiq_visual_tools/sceneiq_waypoint_XXXX.jpg",
        "pixel_x": N, "pixel_y": N,
        "region": "center-right",
        "grid_position": {"row": N, "col": N}
    }"""
```

### Tool 3: analyze_road_geometry

**Implementation**: OpenCV pipeline -- Canny edge detection + Hough line transform + vanishing point estimation
**Output format**: JPEG annotated image (edges + detected lines overlaid) + lane count + curvature classification + vanishing point coordinates
**Purpose**: Provide classical CV analysis of road structure to ground the model's spatial reasoning about lane availability, road curvature, and potential obstructions.

```python
def analyze_road_geometry(image_path: str) -> dict:
    """Returns: {
        "annotated_image": "/tmp/sceneiq_visual_tools/sceneiq_geometry_XXXX.jpg",
        "lanes_detected": N,
        "curvature": "straight|gentle_left|gentle_right|sharp_left|sharp_right",
        "vanishing_point": [x, y],
        "road_boundaries": {...}
    }"""
```

**Bug fixed during Phase 1**: Originally saved annotated image as PNG, which caused vLLM image injection to fail silently. Changed to JPEG output format.

### Tool 4: find_similar_scenes

**Implementation**: CLIP (ViT-B/32) embeddings + FAISS index (L2 distance) for kNN retrieval
**Output format**: List of k nearest neighbor images with their ground truth labels + cosine similarities + image paths
**Purpose**: Show the model what similar-looking scenes were labeled as in the training data, providing visual anchoring for classification.

```python
def find_similar_scenes(image_path: str, k: int = 3) -> dict:
    """Returns: {
        "similar_images": [
            {"image": "/tmp/...", "scene_type": "nominal", "fine_class": "nominal_triggers",
             "similarity": 0.953, "sample_id": 1234},
            ...
        ]
    }"""
```

**Pre-computed index files** (built by `build_scene_index()` in visual_tools.py):
- FAISS index: `tool_calling_experiment/scene_index.faiss` (17 MB, 8,613 vectors x 512 dims)
- Metadata: `tool_calling_experiment/scene_index_metadata.pkl` (602 KB)
- Scene metadata: `tool_calling_experiment/scene_metadata.json` (1 MB)

### ToolCallingOrchestrator

**File**: `tool_calling_experiment/orchestrator.py`

Reusable engine for multi-turn tool-calling conversations via vLLM's OpenAI-compatible API. Handles:
1. Sending messages with base64-encoded images to vLLM
2. Parsing tool calls from model output (hermes parser format)
3. Executing tools locally and injecting results (including JPEG images as base64 data URIs) back into conversation
4. Collecting final predictions (parses FINAL_SCENE, FINAL_LONG_ACTION, FINAL_LAT_ACTION from model output)
5. Logging every intermediate step for analysis
6. Configurable max_tool_rounds to prevent infinite loops

```python
from orchestrator import ToolCallingOrchestrator

orch = ToolCallingOrchestrator(
    server_url="http://localhost:8302",
    tools={"zoom_region": zoom_fn, "check_confusion_risk": confusion_fn, ...},
    tool_definitions=[TOOL_ZOOM_DEF, TOOL_CONFUSION_DEF, ...],
    max_tool_rounds=5,
)
result = orch.run(
    image_path="/path/to/image.jpg",
    system_prompt="You are a driving scene analyst...",
    user_prompt="Classify this driving scene...",
)
# result contains: final_text, final_prediction, tool_calls_log, num_rounds, latency_ms
```

### Statistical Tools v2

**File**: `tool_calling_experiment/tools_v2.py`

Four statistical tools at 4 levels of reasoning scaffolding. All levels share the same OpenAI function-calling schemas; the level parameter controls richness of the return value.

**Levels**:
- **L1 (Raw Data)**: Numbers only. Example: `{"base_rate": 0.037, "is_rare": true}`
- **L2 (Interpreted)**: Numbers + natural language + diagnostic questions. Example: `{"base_rate": 0.037, "interpretation": "incident_zone is very rare (3.7%). The model predicts it 12.5x more often than it occurs. Consider whether this could actually be nominal.", "diagnostic_questions": {...}}`
- **L3 (Procedural)**: L2 + step-by-step visual checklist. Example adds: `{"checklist": ["1. Do you see emergency vehicles with lights?", "2. Is there crash debris?", ...], "resolution_guide": "If no to all, change to nominal."}`
- **L4 (Prescriptive)**: Explicit if/then rules. Example adds: `{"decision_rule": "IF no emergency vehicles AND no crash debris AND no road closure THEN change to nominal"}`

**Usage**:
```python
from tools_v2 import execute_tool_v2, get_tools_for_level

tools = get_tools_for_level(level=2)  # Use L2 -- optimal for 8B
result = execute_tool_v2("check_confusion_risk",
                         {"predicted_scene": "incident_zone"}, level=2)
```

**Hardcoded statistics** (fallback if tool_stats.json missing):
- CLASS_FREQUENCIES: nominal=78.27%, flagger=7.66%, flooded=7.18%, incident_zone=3.74%, mounted_police=3.16%
- CONFUSION_PAIRS: IZ confused with nominal at 41.0%, flooded with IZ at 28.0%, flagger with IZ at 34.4%, mounted_police with nominal at 50.7%, nominal with IZ at 50.9%
- COOCCURRENCE: Full scene-action co-occurrence matrix (15 valid combinations)

---

## Part 3: Phase 1 -- Mechanical Validation (COMPLETED)

**Files**: `tool_calling_experiment/phase1_tasks1to5_results.json` (1.2 MB), `tool_calling_experiment/phase1_tasks6to10_results.json` (19 MB)
**Scripts**: `tool_calling_experiment/phase1_tasks1to5.py`, `tool_calling_experiment/phase1_tasks6to10.py`

Phase 1 validated that the full tool-calling pipeline works mechanically before scaling up. 10 samples per task, 15 tasks total.

### Tasks 1-5: Per-Tool Smoke Tests (2B)

**Task 1: zoom_region smoke test** -- 10 samples, 2B, zoom_region only
- 10/10 made tool calls, 100% valid JSON, 8/10 used zoomed image in reasoning
- Coordinates mostly valid; out-of-range coordinates were clamped successfully

**Task 2: visualize_waypoint smoke test** -- 10 samples, 2B
- Tool calls work, waypoint markers injected correctly as base64 JPEG

**Task 3: analyze_road_geometry smoke test** -- 10 samples, 2B
- Tools work. **Bug fixed**: PNG output changed to JPEG (vLLM rejects PNG in image injection)

**Task 4: find_similar_scenes smoke test** -- 10 samples, 2B
- CLIP+FAISS retrieval works, neighbor images injected correctly

**Task 5: All 4 visual tools together** -- 10 samples, 2B
- Model correctly selects appropriate tools from the set of 4, multi-tool combinations work

### Tasks 6-10: Cross-Model and Advanced Tests

**Task 6: 8B model comparison** -- 10 samples, 8B
- 12 total tool calls (avg 1.2 per sample), 100% argument validity
- 8B references tool output in 20% of final answers (vs lower for 2B)
- 8B makes more selective tool calls than 2B

**Task 7: Max tool rounds stress test** -- 10 samples, 2B, 8 rounds max
- Average 2.4 rounds, max 4 rounds, **0/10 hit the limit**
- Models naturally converge after 2-4 rounds

**Task 8: Image injection verification** (CRITICAL TEST) -- 10 samples, 2B
- 10/10 images injected via base64, 10/10 high-detail responses
- 9/10 zoom-aware (model describes zoomed content differently from original)
- Average 10.3 visual details mentioned per response
- **Conclusion: The model genuinely SEES and processes returned images**

**Task 9: Waypoint comprehension** -- 10 samples, 2B
- 10/10 recognized deliberately bad waypoints (100% recognition rate)
- Model understands spatial meaning of 63x63 grid coordinates

**Task 10: Road geometry comprehension** -- 10 samples, 2B
- 10/10 tool calls succeeded, 10/10 images injected
- 2/10 reference specific numeric data from geometry analysis
- 2/10 mention actual curvature classification
- Model uses the annotated visual output more than the numeric data

### Tasks 11-15: Edge Cases and Fine-Tuned Model

- Fine-tuned model with tools: confirmed -- completely ignores tool output (0% scene revision)
- Malformed coordinates: clamped gracefully, no crashes
- Empty tool results: model recovers and proceeds without tool data
- Conflicting tool outputs: model attempts resolution (quality varies)

### Phase 1 Summary Table

| Metric | Result |
|---|---|
| Tool call JSON validity | 100% across all tasks |
| Image injection (JPEG) | Works perfectly for all 4 tools |
| Model sees returned images | Verified (Task 8: 9/10 zoom-aware) |
| Waypoint comprehension | 10/10 correct assessments |
| Multi-tool coordination | Works (Task 5) |
| Max rounds needed | 2-4 (never hits 8-round limit) |
| 8B vs 2B tool quality | 8B more selective, higher reference rate |
| Fine-tuned model with tools | Ignores tools completely |
| Bugs fixed | PNG->JPEG output, coordinate bounds in prompts |

---

## Part 4: Phase 2 -- Single Tool Deep Dives (Tasks 16-50)

**Scripts**: `tool_calling_experiment/phase2_zoom.py`, `phase2_waypoint.py`, `phase2_geometry.py`, `phase2_retrieval.py`

Each of the 4 visual tools gets a dedicated deep dive. Key tasks use N=1000 for statistical power; specialized/ablation tasks use N=100.

### Phase 2A: Zoom Tool (Tasks 16-25)

**Task 16: Zoom Basic Effectiveness** (N=1000, 2B + 8B)
- Sample: 500 false-IZ, 250 true-IZ, 250 nominal (standard 1000-sample mix)
- Protocol: model receives original image + zoom_region tool, classifies scene
- Prompt: "Use the zoom tool to inspect any suspicious areas, then classify the scene"
- Metrics: accuracy vs no-tool baseline, zoom location quality (categorize targets as road_surface, horizon, object_of_interest, random)
- **Task 16b: 72B variant** -- same protocol, 1000 samples, port 8303

**Task 17: Zoom Location Intelligence** (N=1000, 8B)
- Same 1000 samples as Task 16
- Measure WHERE the model zooms: compute heatmap of zoom centers across 504x336 grid
- Categorize: road_surface (y>250), horizon (y<100), objects_of_interest (100<y<250, based on proximity to detected edges), random
- Conditional accuracy: P(correct | zooms on relevant area) vs P(correct | zooms on irrelevant area)

**Task 18: Zoom Size Ablation** (N=100, 8B)
- Fixed zoom location (center of image at x=252, y=168), vary crop_size: 32, 64, 128, 192, 256
- 20 samples per crop_size
- Expected: diminishing returns above 128; 32 may be too small to see anything useful

**Task 19: Multi-Zoom Strategy** (N=100, 8B)
- Allow up to 3 zoom calls per sample
- Measure: 1-zoom vs 2-zoom vs 3-zoom accuracy
- Where does model zoom on second/third call? Same area (refined) or different area (exploration)?

**Task 20: Zoom on nominal_triggers Specifically** (N=1000, 8B)
- 1000 samples exclusively from fine_class=nominal_triggers (2,801 available, sample 1000)
- This is the hardest sub-population: nominal scenes with construction-like visual triggers
- Can closer inspection distinguish construction zone infrastructure from active incident?
- Per-prediction breakdown: of 1000 nominal_triggers, how many does zoom correct?

**Task 21: Zoom + L2 Statistical Tools Combined** (N=1000, 8B)
- 3 conditions on same 1000 samples: zoom-only, L2-stats-only, zoom+L2
- Tests: are visual and statistical tools complementary or redundant?
- Expected: complementary, because statistical tools provide population-level information while zoom provides sample-level visual detail

**Task 22: Zoom Coordinate Prompt Sensitivity** (N=100, 8B)
- 3 prompt variants:
  - A: "Zoom anywhere you think is important" (no guidance)
  - B: "The image is 504x336. Road surface at y>200. Objects of interest at y=150-250" (spatial guidance)
  - C: "Use zoom_region to inspect the area around any unusual objects (cones, barriers, vehicles)" (semantic guidance)
- ~33 samples per variant
- Measures: zoom location quality, accuracy, and whether guidance helps or constrains

**Task 23: Zoom on Non-IZ Classes** (N=100, 8B)
- 25 flagger, 25 flooded, 25 mounted_police, 25 nominal
- Does zoom help for classes beyond IZ? Expected: helps flooded (water texture), maybe flagger (person detection)

**Task 24: Zoom Resolution vs Accuracy** (N=100, 8B)
- Test at 3 base resolutions: 504x336 (native), 252x168 (half), 126x84 (quarter)
- Hypothesis: zoom is more valuable at lower base resolutions where small objects are not visible

**Task 25: Zoom Failure Analysis** (N=100, 8B)
- From Task 16, take 100 samples where zoom did NOT help (model still wrong after zooming)
- Categorize failures: bad zoom location, ambiguous zoomed content, model ignored zoom result, tool returned uninformative crop

### Phase 2B: Waypoint Visualization (Tasks 26-35)

**Task 26: Waypoint Visualization Basic** (N=1000, 2B + 8B)
- Standard 1000-sample mix
- Show predicted waypoint on image, ask model to assess scene + waypoint feasibility
- **Task 26b: 72B variant**

**Task 27: Waypoint as Scene Evidence** (N=1000, 8B)
- Hypothesis: nominal scenes have straight-ahead waypoints with small deltas; IZ scenes have lane-change waypoints with large lateral deltas
- Measure: P(correct scene | waypoint pattern consistent) vs P(correct scene | waypoint pattern inconsistent)

**Task 28: Bad Waypoint Detection** (N=100, 8B)
- Inject deliberately wrong waypoints (nominal scene + lane-change waypoint; IZ scene + straight waypoint)
- Measure: does the model flag the inconsistency between scene and waypoint?

**Task 29: Waypoint + Zoom Combined** (N=1000, 8B)
- Both tools available. Which does the model prefer? Does it use both? Synergy or interference?

**Task 30: Waypoint Sequence Visualization** (N=100, 8B)
- Show all 5 waypoints on image (not just first). Does the full trajectory arc provide more signal?

**Task 31: Waypoint Grid Comprehension** (N=1000, 8B)
- Present waypoints in both formats: grid coordinates (row, col on 63x63) and pixel coordinates
- Does coordinate format affect model comprehension?

**Task 32: Waypoint Anomaly Detection** (N=100, 8B)
- Mix of physically possible and impossible trajectories
- True positive rate for detecting violations

**Task 33: Waypoint for Action Prediction** (N=100, 8B)
- Use visualization to predict action (not scene). Action accuracy with vs without viz.

**Task 34: Waypoint Confidence Correlation** (N=100, 8B)
- Does waypoint reasonableness correlate with scene prediction correctness?

**Task 35: Waypoint Failure Analysis** (N=100, 8B)
- From Task 26, analyze failures

### Phase 2C: Road Geometry (Tasks 36-42)

**Task 36: Road Geometry Basic** (N=1000, 2B + 8B)
- Standard 1000-sample mix with analyze_road_geometry tool
- Metrics: accuracy improvement, which geometric features (lanes, curvature, vanishing point) the model references
- **Task 36b: 72B variant**

**Task 37: Lane Detection Quality** (N=100, 8B)
- Manual spot-check 20 samples: is OpenCV lane count correct? Curvature estimate correct?
- Measure: does incorrect lane info lead to worse predictions than no lane info?

**Task 38: Geometry + Scene Relationship** (N=1000, 8B)
- Test: do geometric features differ between IZ (blocked lanes, road closure) and nominal (normal lanes)?
- Compute: correlation between lanes_detected, curvature, and scene type

**Task 39: Geometry for Flooded Scenes** (N=100, 8B)
- Can edge detection differentiate wet vs dry road? Water disrupts road texture.

**Task 40: Geometry vs Zoom** (N=100, 8B)
- Same samples: geometry-only vs zoom-only. Which is more informative?

**Task 41: Geometry + Zoom Combined** (N=100, 8B)
- Both tools. Synergy measurement.

**Task 42: Geometry Failure Analysis** (N=100, 8B)
- Where does geometry mislead?

### Phase 2D: Similar Scene Retrieval (Tasks 43-50)

**Task 43: Retrieval Basic** (N=1000, 2B + 8B)
- k=3 nearest neighbors from CLIP index, model sees neighbor images + labels
- **Task 43b: 72B variant**

**Task 44: Retrieval k Ablation** (N=1000, 8B)
- k = 1, 3, 5, 10 (250 samples per k). Accuracy vs k curve.

**Task 45: Retrieval Label Unanimity** (N=100, 8B)
- Of k=3 neighbors: how often are all 3 labels the same class?
- If unanimous and agree with prediction: confidence signal
- If unanimous and disagree: strong revision signal
- If split: model must reason about ambiguity

**Task 46: Retrieval for Rare Classes** (N=1000, 8B)
- 200 each of flagger, flooded, incident_zone, mounted_police, 200 nominal
- Per-class retrieval quality: P(neighbor same class as query)

**Task 47: Retrieval + Statistical Tools** (N=100, 8B)
- CLIP retrieval + L2 statistical tools combined. Complementarity test.

**Task 48: Retrieval as Simple Heuristic** (N=100, 8B)
- If k=3 neighbors agree with prediction, keep. If disagree, revise to majority neighbor label.
- No model reasoning -- pure heuristic. Compare accuracy vs model-in-the-loop.

**Task 49: Retrieval Quality Audit** (N=100, 8B)
- For 100 samples, manually inspect k=3 neighbors. Are they visually similar? Same scene type?

**Task 50: Retrieval Failure Analysis** (N=100, 8B)

---

## Part 5: Phase 3 -- Multi-Tool Interaction (Tasks 51-70)

**Script**: `tool_calling_experiment/phase3_multitool.py`

### Tool Combination Experiments (Tasks 51-55)

**Task 51: All Visual Tools** (N=1000, 2B + 8B)
- All 4 visual tools available simultaneously
- Measure: which tools does the model use? In what order? How many total calls?
- Compare: all-tools accuracy vs best single tool from Phase 2
- **Task 51b: 72B variant**

**Task 52: Visual + Statistical L2** (N=100, 8B)
- All 8 tools: 4 visual + 4 statistical at L2
- Does the model get overwhelmed by 8 tool choices? Does it combine information sources?

**Task 53: Tool Selection Heuristic** (N=100, 8B)
- Present all 8 tools, measure which ones the model calls first
- Build implicit decision tree of model's tool selection strategy

**Task 54: Forced Tool Order** (N=1000, 8B)
- 4 orderings x 250 samples:
  - A: zoom -> retrieval -> geometry -> statistical
  - B: statistical -> zoom -> retrieval -> geometry
  - C: retrieval -> statistical -> zoom -> geometry
  - D: model chooses freely
- From Step 0.7, chain order did not matter for text-only tools. Does this hold for visual tools?

**Task 55: Minimum Sufficient Toolset** (N=100, 8B)
- Ablation: remove one tool at a time from full set of 8
- Which tool's removal hurts most? Expected: zoom_region or check_confusion_risk.

### Pipeline Architecture (Tasks 56-60)

**Task 56: Pipeline A -- Predict-then-Verify** (N=100, 8B)
- Fine-tuned model predicts at temp=0, base 8B verifies with all tools
- From Step 0.7: tools after prediction are 10x less effective. Confirm at 100-sample scale.

**Task 57: Pipeline B -- Staged Prediction** (N=1000, 8B)
- Base model classifies from scratch with tools available during reasoning
- Multi-turn: scene classification first (with tools), then action, then waypoints
- Expected: significantly better than Pipeline A based on Step 0.7 finding

**Task 58: Pipeline C -- Ensemble** (N=100, 8B)
- Fine-tuned predicts + base model predicts independently
- When they agree, keep. When they disagree, consult tools to break tie.

**Task 59: Pipeline D -- Progressive Refinement** (N=100, 8B)
- Round 1: base model quick classification (no tools)
- Round 2: tools provide feedback on Round 1 prediction
- Round 3: model revises
- Compare vs Pipeline B (tools from the start)

**Task 60: Best Pipeline at Scale** (N=1000, 8B)
- Run winner from Tasks 56-59 at 1000 samples
- Full metrics: accuracy, macro F1, per-class F1, confusion matrix, latency
- **Task 60b: 72B variant**

### Information Flow Analysis (Tasks 61-65)

**Task 61: Tool Influence Tracing** (N=100, 8B)
- For each sample, identify which tool call most influenced the final prediction
- Annotate: "model cited zoom result" vs "model cited retrieval result" vs "model cited statistics"

**Task 62: Conflicting Tool Signals** (N=100, 8B)
- Engineered conflicts: retrieval says "nominal" but statistics say "high confusion risk"
- How does the model resolve inter-tool conflicts?

**Task 63: Tool Hallucination Detection** (N=100, 8B)
- Does the model claim a tool said something it did not? Cross-reference stated reasoning with actual tool outputs.

**Task 64: Tool Redundancy** (N=100, 8B)
- When multiple tools provide the same signal, does redundancy increase revision confidence?

**Task 65: Information Gain Per Tool Call** (N=100, 8B)
- Measure prediction entropy before and after each tool call. Which tool provides the most information gain?

### Cross-Model Comparison (Tasks 66-70)

**Task 66: 2B vs 8B Tool Effectiveness** (N=1000, 2B + 8B)
- Same 1000 samples, same tools, same prompts. Full comparison table.
- **Task 66b: Add 72B**

**Task 67: 2B vs 8B Tool Selection** (N=1000, 2B + 8B)
- Do the models call different tools for the same sample? Map preferences.
- **Task 67b: Add 72B**

**Task 68: 2B vs 8B Reasoning Quality** (N=1000, 2B + 8B)
- Score reasoning quality: automated assessment + 100-sample manual spot check
- Measure: correlation between reasoning quality and accuracy
- **Task 68b: Add 72B**

**Task 69: Model Size Scaling Curve** (N=100, 2B + 8B + 72B)
- Same 100 samples across all 3 model sizes
- Plot accuracy vs model size for: no tools, L1, L2, visual tools, all tools
- Identify: at what model size do tools stop adding value?

**Task 70: Cross-Model Agreement** (N=100, 2B + 8B + 72B)
- When all 3 models agree: what is accuracy? (Expected: very high)
- When they disagree: which model is right most often? (Expected: 72B)
- Can disagreement serve as an uncertainty signal?

---

## Part 6: Phase 4 -- Stress Tests (Tasks 71-85)

### Temperature and Sampling (Tasks 71-74)

**Task 71: Temperature Sweep** (N=100, 8B)
- Temperature: 0.0, 0.3, 0.6, 1.0 (25 samples each)
- Measure: accuracy, revision rate, tool call diversity, reasoning variability
- **Task 71b: Temperature Sweep on 72B** (N=100, same protocol)

**Task 72: Best-of-N with Tools** (N=100, 8B)
- Temperature 0.6, generate 5 responses per sample
- Best-of-5 accuracy vs single-shot accuracy. Do tools + sampling combine well?

**Task 73: Self-Consistency + Tools** (N=100, 8B)
- Temperature 0.6, 10 samples per input, majority vote
- Self-consistency alone vs tools alone vs self-consistency + tools

**Task 74: Top-k/Top-p Sensitivity** (N=100, 8B)
- Fix temp=0.6, vary top_k (10, 50, 200) and top_p (0.9, 0.95, 0.99)

### Robustness (Tasks 75-79)

**Task 75: Prompt Sensitivity** (N=100, 8B)
- 5 prompt variants (varying detail, tool descriptions). 20 samples each.
- Accuracy variance across prompts.

**Task 76: Image Quality Degradation** (N=100, 8B)
- JPEG quality=30, 5x5 Gaussian blur, resize to 252x168 then upscale back
- Does zoom help more at lower quality?

**Task 77: Tool Failure Resilience** (N=100, 8B)
- Simulate tool errors. Does the model recover gracefully? Retry?

**Task 78: Adversarial Tool Outputs** (N=100, 8B)
- Return deliberately wrong tool results. Does the model blindly trust tools?

**Task 79: Context Length Pressure** (N=100, 8B)
- 1-round, 3-round, 5-round, 8-round. Accuracy vs context length.
- **Task 79b: FP8 Quantization on 72B** (N=100)
  - Compare FP16 vs FP8 72B on same 100 samples
  - Reference: `/fsx/models/Qwen2.5-72B-Instruct-FP8` for methodology

### Scale (Task 80)

**Task 80: Full Dataset Run** (N=8,613, best config from Phase 3)
- Run winning configuration on ALL 8,613 samples
- Metrics:
  - Overall accuracy, macro F1, weighted F1
  - Per-class precision, recall, F1
  - Per-fine_class accuracy (critical: nominal_triggers at 19.0% baseline, negative_incident_zone)
  - Full confusion matrix
  - Latency: mean, p50, p95, p99
  - Cost: total GPU-hours, per-sample time
- This is the definitive result of the experiment.

### Edge Cases (Tasks 81-85)

**Task 81: Ambiguous Samples** (N=100, 8B)
- 100 samples where fine-tuned model disagrees with majority vote (self-consistency experiment)
- These are genuinely ambiguous. Do tools help?

**Task 82: Easy Samples** (N=100, 8B)
- 100 samples where baseline gets correct with >99% agreement
- Do tools hurt on easy samples? (Expected: no, model should skip revision)

**Task 83: Distribution Edge** (N=100, 8B)
- 25 each of the 4 rarest fine_classes. Do tools help on rare sub-populations?

**Task 84: Location-Specific Performance** (N=100, 8B)
- Top-5 most common locations, 20 samples each. Location-dependent tool effectiveness?

**Task 85: Temporal Analysis** (N=100, 8B)
- Samples across different timestamps/seasons. Time-dependent effectiveness?

---

## Part 7: Phase 5 -- Synthesis (Tasks 86-100)

### Ablation Studies (Tasks 86-90)

**Task 86: Feature Importance Ranking** (N=1000, 8B)
- Shapley-value-style ablation for each tool's marginal contribution to accuracy

**Task 87: Minimum Viable Toolset** (N=100, 8B)
- Smallest tool set achieving 95% of full-toolset accuracy
- Expected: zoom_region + check_confusion_risk may suffice

**Task 88: Tool Level x Visual Tool Interaction** (N=100, 8B)
- Cross: {L1, L2, L3, L4} x {no visual, zoom only, all visual} = 12 conditions
- ~8 samples each, find optimal combination
- **Task 88b: 2B vs 8B vs 72B Capability Comparison** (N=100 each)
  - Same 100 samples, optimal toolset from Task 88, all 3 model sizes
  - Full comparison: accuracy, revision rate, reasoning quality, tool selection

**Task 89: Post-Processing + Tools Combined** (N=1000, 8B)
- Apply confusion-aware post-processing (S3 from Step 0.1: 86.7% accuracy, 0.610 F1) AFTER tool-assisted prediction
- Does post-processing stack on top of tools? Or are they redundant?

**Task 90: Error Taxonomy** (N=all errors from Task 80)
- Categorize every remaining error:
  - Perceptual failure: model cannot distinguish visual features despite tools
  - Tool ignorance: model saw correct tool info but did not act on it
  - Tool misguidance: tool gave wrong/misleading signal (e.g., overcorrection of true IZ)
  - Inherent ambiguity: even humans would disagree on the label
  - Label noise: ground truth may be incorrect

### Cost-Benefit Analysis (Tasks 91-95)

**Task 91: Compute Budget Analysis** (N=calculated)
- For each configuration: GPU-seconds per sample, accuracy per GPU-second, marginal gain per additional tool call
- Pareto frontier: accuracy vs compute
- **Task 91b: Include 72B in Cost Analysis**
  - 72B uses 4 GPUs vs 1 for 2B/8B
  - Accuracy per GPU-hour across all model sizes
  - Key question: does 8B + good tools match 72B for less compute?

**Task 92: Latency Budget Analysis** (N=calculated)
- Mean and P95 end-to-end latency per configuration
- Tool call contribution to total latency
- Fastest configuration achieving 80% accuracy

**Task 93: Accuracy Ceiling Analysis** (N=calculated)
- Maximum achievable accuracy with each approach:
  - Post-processing only: 86.7% / 0.610 F1 (from Step 0.1 S3)
  - Text-only tools (L2, 8B): 75.0% on 20-sample subset (from Step 0.8)
  - Visual tools: TBD from Phase 2
  - All tools combined: TBD from Phase 3
  - Oracle: 92.4% / 0.775 F1 (from Step 0.1 S4)

**Task 94: Remaining Error Decomposition** (N=calculated)
- After best configuration:
  - N errors from nominal_triggers (perceptual misclassification)
  - N errors from flagger/IZ confusion (perceptual)
  - N errors from flooded/IZ confusion (perceptual)
  - N errors from mounted_police/nominal confusion (perceptual)
  - N errors from true IZ overcorrection (tool-induced harm)

**Task 95: Tool Calling ROI** (N=calculated)
- Three paradigms compared:
  - Retraining: high development cost, fixes root cause, best long-term
  - Post-processing: zero additional cost, limited ceiling (86.7%)
  - Tool calling: moderate cost, moderate ceiling, flexible and interpretable
- For each: accuracy, compute cost, development time, maintainability

### Recommendations (Tasks 96-100)

**Task 96: Production Configuration Recommendation**
- Specify: which model(s), which tools, which pipeline, post-processing rules
- Expected accuracy, latency, GPU requirements

**Task 97: Generalization Hypothesis**
- When do tools help? (systematic bias correctable by external info)
- When do they fail? (perceptual failures, model ignores tools)
- Optimal tool level? (L2 for 2B/8B; test whether 72B can use L1)
- When are tools better than post-processing? (non-deterministic corrections)
- **Update with 72B scaling insights**: does model scale change the break-even point for tool utility?

**Task 98: Transfer to Other Domains**
- Medical imaging, document classification, robotics -- how does this methodology transfer?

**Task 99: Research Paper Outline**
- "When Do Tools Help VLMs? A Systematic Study of Tool-Augmented Scene Classification"
- Key figures: Level x Model accuracy matrix, save/break trade-off curve, Pareto frontier

**Task 100: Final Report**
- Executive summary (1 page), detailed results, recommendations, limitations

---

## Part 8: Path A vs Path B Comparison (NEW)

This is the culminating experiment that teaches the most about tool calling as a paradigm. It addresses the fundamental question: **how does tool design differ when correcting opposite failure modes?**

### Background

The fine-tuned model and base model have opposite biases:

| | Fine-tuned 2B | Base 8B (zero-shot) |
|---|---|---|
| Dominant prediction | incident_zone (46.8%) | nominal (80%) |
| IZ prediction rate | 12.5x over-predicted | Near true rate (10%) |
| Primary error | False positives (nominal -> IZ) | False negatives (non-nominal -> nominal) |
| Root cause | Learned "see cone -> predict IZ" shortcut | Strong prior from pretraining |
| Tool goal | RESTRAIN overconfidence | PUSH past underconfidence |

These are mirror-image failure modes. The optimal tool design for each may be fundamentally different.

### Path A: Tools for Fine-Tuned Model (Restrain Overconfidence)

**Goal**: Reduce false-positive IZ predictions without losing true positives.

**Tool design philosophy**: Skeptical tools that challenge rare predictions.
- check_confusion_risk: "incident_zone has a 50.9% false positive rate in your predictions"
- check_scene_prior: "incident_zone base rate is 3.7%, yet you predict it 46.8% of the time"
- zoom_region: "Look more carefully at that area -- do you REALLY see emergency vehicles?"
- find_similar_scenes: "Here are 3 visually similar scenes, all labeled nominal"

**Pipeline**: Pipeline A (predict-then-verify). Fine-tuned model predicts, base model verifies with skeptical tools. The base model acts as a skeptical auditor.

**Risk**: Overcorrection of true IZ (already demonstrated in Steps 0.8 and 0.9).

### Path B: Tools for Base Model (Push Underconfidence)

**Goal**: Increase true-positive detection of non-nominal scenes (flooded, flagger, IZ, mounted_police).

**Tool design philosophy**: Encouraging tools that highlight anomalies.
- check_confusion_risk: "The model often misses flooded scenes (28% miss rate)"
- check_scene_prior: "While nominal is most common (78%), always check carefully for hazards before defaulting to nominal"
- zoom_region: "Look at this suspicious region -- could this be standing water? A flagger? Emergency vehicles?"
- find_similar_scenes: "Here are similar scenes with various labels -- consider all possibilities, not just nominal"
- analyze_road_geometry: "Road surface analysis detected possible water/obstruction/lane blockage"

**Pipeline**: Pipeline B (staged prediction). Base model classifies from scratch with encouraging tools interleaved with reasoning (leveraging Step 0.7 finding: tools during reasoning are 10x more effective than after).

**Risk**: False alarms -- over-detecting hazards in normal scenes.

### Experiment Design

**Task PA-1: Path A at Scale** (N=1000 per model)
- Sample: 500 false-IZ + 200 true-IZ + 300 nominal (biased toward the Path A use case)
- Models: 2B verifier, 8B verifier, 72B verifier (each on its own GPU/port)
- Tools: all 8 tools with "skeptical" prompt framing
- Measure: save rate on false-IZ, break rate on true-IZ, overall accuracy, macro F1

**Task PB-1: Path B at Scale** (N=1000 per model)
- Same 1000 samples as PA-1 for direct comparison
- Models: base 2B, base 8B, base 72B (no fine-tuned model involved)
- Tools: all 8 tools with "encouraging" prompt framing
- Measure: detection rate for non-nominal, false alarm rate, overall accuracy, macro F1

**Task PAB-1: Direct Comparison** (N=1000, best model per path)
- Same 1000 samples, best-performing model for each path
- Head-to-head: Path A accuracy vs Path B accuracy vs no-tool baseline vs confusion-aware PP (86.7%)
- This answers: is it better to fix a biased fine-tuned model with tools, or start from scratch with a base model + tools?

**Task PAB-2: Tool Adaptation Analysis** (N=100)
- For 100 samples, run BOTH paths
- Categorize:
  - Path A wins, Path B loses (sample benefits from skeptical tools)
  - Path B wins, Path A loses (sample benefits from encouraging tools)
  - Both win (either approach works)
  - Neither wins (sample is genuinely hard)
- Analyze: what visual/statistical features predict "Path A amenable" vs "Path B amenable"?

**Task PAB-3: Hybrid Path** (N=1000)
- Router based on fine-tuned model's prediction:
  - Predicts IZ -> Route to Path A (skeptical verification with base model)
  - Predicts nominal -> Route to Path B (base model re-classifies with encouraging tools)
  - Predicts other (flagger, flooded, mounted_police) -> Keep (fine-tuned is usually right for these)
- This adaptive routing leverages both models' strengths while using tools to address each model's specific weakness.
- Expected: best of both worlds, since the router condition (fine-tuned prediction) is free and highly informative.

### Expected Results Table

| Metric | Path A (8B verify) | Path B (8B fresh) | Hybrid | PP-only (S3) | Oracle (S4) |
|---|---|---|---|---|---|
| Overall accuracy | ~80-85% | ~65-75% | ~82-88% | 86.7% | 92.4% |
| Macro F1 | ~0.55-0.65 | ~0.40-0.55 | ~0.60-0.70 | 0.610 | 0.775 |
| False-IZ save rate | ~70-90% | N/A | ~70-90% | 100% (S1) | 100% |
| True-IZ break rate | ~30-60% | N/A | ~20-40% | 100% (S1) | 0% |
| Non-nominal detection | N/A | ~30-50% | ~30-50% | 0% gain | varies |
| Latency multiplier | 2-3x | 2-3x | 2-3x | 1x | 1x |

### Cross-Model Path Comparison Table

| Metric | Path A (2B) | Path A (8B) | Path A (72B) | Path B (2B) | Path B (8B) | Path B (72B) |
|---|---|---|---|---|---|---|
| Overall accuracy | -- | -- | -- | -- | -- | -- |
| Macro F1 | -- | -- | -- | -- | -- | -- |
| False-IZ save rate | -- | -- | -- | -- | -- | -- |
| True-IZ break rate | -- | -- | -- | -- | -- | -- |
| Non-nominal detect | -- | -- | -- | -- | -- | -- |
| Avg tool calls | -- | -- | -- | -- | -- | -- |
| Avg latency (ms) | -- | -- | -- | -- | -- | -- |
| GPU cost factor | 1x | 1x | 4x | 1x | 1x | 4x |

---

## Appendix A: Files Created

All files in `/workspace/vllm/tool_calling_experiment/`:

| File | Size | Description |
|---|---|---|
| `__init__.py` | 0 B | Package init |
| `base_model_zeroshot_results.json` | 10 KB | Step 0.3: 2B+8B zero-shot on 10 samples |
| `base_model_zeroshot_test.py` | 22 KB | Step 0.3 script |
| `build_tools.py` | 12 KB | Build tool statistics from self-consistency DB |
| `calibration_analysis.py` | 19 KB | Step 0.1: 5 post-processing strategies |
| `calibration_results.json` | 18 KB | Step 0.1 results |
| `compute_analysis.py` | 30 KB | Post-processing: flip analysis, revision rates |
| `conditioned_prediction_results.json` | 29 KB | Step 0.2: fine-tuned ignores corrections |
| `create_db.py` | 5 KB | SQLite schema creation |
| `generate_insights_report.py` | 33 KB | Insight report generator |
| `generate_sample_report.py` | 12 KB | Per-sample JSONL report generator |
| `lesson1_weights_vs_tools.md` | 35 KB | Step 0.4: weights vs tools decomposition |
| `lesson3_tool_choice.py` | 25 KB | Step 0.5: tool choice conditions |
| `lesson4_contradictory_tools.py` | 27 KB | Step 0.6 script |
| `lesson4_output.log` | 3 KB | Step 0.6 server log |
| `lesson4_results.json` | 155 KB | Step 0.6: 8B engages but anchors (7% revision) |
| `lesson5_results.json` | 132 KB | Step 0.7: mid-chain vs post-prediction |
| `lesson5_tool_chain.py` | 54 KB | Step 0.7 script |
| `lesson_tool_levels.py` | 41 KB | Step 0.8: L1-L4 x 2B/8B experiment |
| `orchestrator.py` | 34 KB | ToolCallingOrchestrator engine |
| `parse_tool_calls.py` | 6 KB | Parse `<tool_call>` XML from model output |
| `phase1_tasks1to5.py` | 30 KB | Phase 1 tasks 1-5 script |
| `phase1_tasks1to5_results.json` | 1.2 MB | Phase 1 tasks 1-5 results |
| `phase1_tasks6to10.py` | 31 KB | Phase 1 tasks 6-10 script |
| `phase1_tasks6to10_results.json` | 19 MB | Phase 1 tasks 6-10 results |
| `phase2_geometry.py` | 45 KB | Phase 2C: road geometry deep dive |
| `phase2_retrieval.py` | 51 KB | Phase 2D: CLIP retrieval deep dive |
| `phase2_waypoint.py` | 64 KB | Phase 2B: waypoint viz deep dive |
| `phase2_zoom.py` | 46 KB | Phase 2A: zoom tool deep dive |
| `phase3_multitool.py` | 59 KB | Phase 3: multi-tool interaction |
| `run_conditioned_prediction.py` | 16 KB | Step 0.2 script |
| `run_prediction.py` | 18 KB | Fine-tuned model prediction runner |
| `run_staged.py` | 20 KB | Pipeline B: staged multi-turn |
| `run_tool_loop.py` | 21 KB | Step 0.9: mechanical proof |
| `run_verification.py` | 21 KB | Pipeline A: predict-then-verify |
| `scene_index.faiss` | 17 MB | Pre-computed FAISS index (8,613 CLIP vectors) |
| `scene_index_metadata.pkl` | 602 KB | FAISS index metadata |
| `scene_metadata.json` | 1 MB | Full scene metadata for all samples |
| `server_8305.log` | 25 KB | vLLM server log |
| `server_8306.log` | 23 KB | vLLM server log |
| `server_utils.py` | 8 KB | Server management utilities |
| `tool_calling.db` | 2.8 MB | SQLite database with experiment results |
| `tool_levels_results.json` | 232 KB | Step 0.8: 4 levels x 2 models x 20 samples |
| `tool_loop_results.json` | 147 KB | Step 0.9: 10-sample mechanical proof |
| `tool_stats.json` | 2 KB | Pre-computed class frequencies + confusion |
| `tools.py` | 21 KB | Tool definitions v1 |
| `tools_v2.py` | 64 KB | Tool definitions with 4 reasoning levels |
| `visual_tools.py` | 49 KB | 4 visual tools + utilities |

---

## Appendix B: System Prompts

### System Prompt for Pipeline B (Staged Prediction with Visual Tools)

```
You are a driving scene classification system. You will analyze a dashcam image and classify it.

The image is 504 pixels wide and 336 pixels tall.
- X coordinates: 0 (left edge) to 503 (right edge)
- Y coordinates: 0 (top edge) to 335 (bottom edge)
- The road surface is typically at y > 200
- Objects of interest (vehicles, cones, people) are typically at y = 100-280

Scene types (from most to least common):
- nominal (78.1%): Normal driving, no hazards. The VAST MAJORITY of scenes.
- flagger (7.7%): Human directing traffic at construction zone.
- flooded (7.2%): Standing water on road surface.
- incident_zone (3.7%): Active incident -- emergency vehicles, crashes, road closures. VERY RARE.
- mounted_police (3.2%): Police on horseback.

IMPORTANT: incident_zone is predicted 12.5x more often than it actually occurs. If you think you see an incident zone, use your tools to verify. Most scenes with traffic cones or barriers are actually nominal.

Actions:
- Longitudinal: null (maintain speed), stop, slowdown, proceed
- Lateral: null (stay in lane), lc_left, lc_right
- Only incident_zone has lane changes. nominal and mounted_police always have (null, null).

Use the available tools to inspect the image before making your prediction. When ready, output:

FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REVISED: <yes|no>
REASON: <brief explanation>
```

### System Prompt for Pipeline A (Verify-then-Revise)

```
You are a driving scene verification system. A prediction model analyzed a dashcam image and predicted:
- Scene type: {predicted_scene}
- Longitudinal action: {predicted_long_action}
- Lateral action: {predicted_lat_action}

The image is 504 pixels wide and 336 pixels tall (x: 0-503, y: 0-335).
Road surface is typically at y > 200. Objects of interest at y = 100-280.

Use the available tools to check whether these predictions are reasonable. After checking, provide your corrected predictions:

FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REVISED: <yes|no>
REASON: <brief explanation>
```

### System Prompt for Path A (Skeptical Verification)

```
You are a SKEPTICAL driving scene verifier. A prediction model tends to over-predict incident_zone (predicts it 46.8% of the time, but it actually occurs only 3.7%). Your job is to catch false alarms.

The model predicted:
- Scene: {predicted_scene}
- Long action: {predicted_long_action}
- Lat action: {predicted_lat_action}

The image is 504x336 pixels. Use your tools to verify. Be especially skeptical of incident_zone predictions -- most are wrong. Look for:
- ACTUAL emergency vehicles with flashing lights (not just parked vehicles)
- ACTUAL crash debris or road damage (not just construction materials)
- ACTUAL road closures (not just lane guidance cones)

If you see none of these, the scene is likely nominal despite visual triggers.

FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REVISED: <yes|no>
REASON: <brief explanation>
```

### System Prompt for Path B (Encouraging Classification)

```
You are a THOROUGH driving scene classifier. Your goal is to detect hazards that might be missed.

The image is 504x336 pixels. Classify this driving scene carefully.

While nominal is the most common scene (78%), do NOT default to nominal without checking. Use your tools to look for:
- Standing water on the road (flooded)
- A person directing traffic with signs/flags (flagger)
- Emergency vehicles, crashes, road closures (incident_zone)
- Police on horseback (mounted_police)

Zoom into suspicious areas. Check similar scenes. Analyze road geometry. Only classify as nominal if you are confident there are no hazards.

FINAL_SCENE: <scene_type>
FINAL_LONG_ACTION: <action>
FINAL_LAT_ACTION: <action>
REASON: <brief explanation>
```

---

## Appendix C: Sample Selection Criteria

### Standard 1000-Sample Mix

All N=1000 experiments use this stratified mix unless otherwise noted:

| Category | Count | Selection Criteria |
|---|---|---|
| False-IZ | 500 | GT=nominal, predicted=incident_zone |
| True-IZ | 100 | GT=incident_zone, predicted=incident_zone |
| True-nominal | 200 | GT=nominal, predicted=nominal |
| Flagger | 50 | GT=flagger, any prediction |
| Flooded | 50 | GT=flooded, any prediction |
| Mounted_police | 50 | GT=mounted_police, any prediction |
| Mixed errors | 50 | Other misclassification types |

**Rationale**: Over-samples false-IZ (the dominant error, 3,430 samples available) while including enough of each class for meaningful per-class metrics. The 50% false-IZ proportion means that save/break analysis has statistical power.

**Seed**: `random.seed(42)` for all sample selection. Reproducible.

### Standard 100-Sample Mix

| Category | Count |
|---|---|
| False-IZ | 50 |
| True-IZ | 10 |
| True-nominal | 20 |
| Flagger | 5 |
| Flooded | 5 |
| Mounted_police | 5 |
| Mixed errors | 5 |

### Specific Population Tasks

| Task | Population | N available | N sampled |
|---|---|---|---|
| Task 20 (nominal_triggers) | fine_class=nominal_triggers | 2,801 | 1,000 |
| Task 23 (non-IZ classes) | 25 each of flagger, flooded, mounted_police, nominal | varies | 100 |
| Task 46 (rare classes) | 200 each of 5 classes | varies | 1,000 |
| Task 81 (ambiguous) | Disagrees with majority vote | ~500 | 100 |
| Task 82 (easy) | >99% agreement in self-consistency | ~3,000 | 100 |

### Ground Truth Source

All ground truth from `self_consistency_experiment/self_consistency.db`:
- Table: `predictions` with `experiment_id='sc_greedy'`
- Fields: `scene_type_gt`, `long_action_gt`, `lat_action_gt`, `odd_label`, `fine_class`, `location`
- Greedy predictions: `original_scene`, `original_long_action`, `original_lat_action`

---

## Appendix D: Key Numbers Reference

| Metric | Value | Source |
|---|---|---|
| Total dataset size | 8,613 samples | MDS dataset |
| Baseline accuracy (fine-tuned greedy) | 46.9% | calibration_results.json |
| Baseline macro F1 | 0.429 | calibration_results.json |
| IZ prediction rate | 46.8% (4,081 predictions) | calibration_results.json |
| IZ true rate | 3.7% (322 samples) | calibration_results.json |
| IZ over-prediction factor | 12.5x | calibration_results.json |
| Errors from IZ over-prediction | 85.6% (3,918 / 4,577) | calibration_results.json |
| nominal_triggers count | 2,801 samples | calibration_results.json |
| nominal_triggers accuracy | 19.0% | calibration_results.json |
| Naive flip accuracy (S1) | 84.8% | calibration_results.json |
| Confusion-aware PP accuracy (S3) | 86.7% | calibration_results.json |
| Confusion-aware PP macro F1 (S3) | 0.610 | calibration_results.json |
| Oracle ceiling accuracy (S4) | 92.4% | calibration_results.json |
| Oracle ceiling macro F1 (S4) | 0.775 | calibration_results.json |
| Fine-tuned scene change rate (conditioned) | 0/20 (0.0%) | conditioned_prediction_results.json |
| Fine-tuned action change rate (conditioned) | 11/20 (55.0%) | conditioned_prediction_results.json |
| Base 2B zero-shot accuracy | 70% (10 samples) | base_model_zeroshot_results.json |
| Base 8B zero-shot accuracy | 40% (10 samples) | base_model_zeroshot_results.json |
| 8B L2 tool-assisted accuracy | 75.0% (20 samples) | tool_levels_results.json |
| 8B L2 revision rate | 65.0% | tool_levels_results.json |
| 8B L2 net improvement | +7 (10 saves, 3 breaks) | tool_levels_results.json |
| 2B L2 tool-assisted accuracy | 68.8% (16 parseable / 20) | tool_levels_results.json |
| 2B L2 net improvement | +5 (8 saves, 3 breaks) | tool_levels_results.json |
| L1 revision rate (both models) | 0.0% | tool_levels_results.json |
| 2B L4 revision rate | 0.0% (collapses) | tool_levels_results.json |
| 8B L3/L4 accuracy | 75.0% (same as L2) | tool_levels_results.json |
| Mid-chain revision rate | 66.7% (2/3) | lesson5_results.json |
| Post-prediction revision rate | 6.7% (1/15) | lesson5_results.json |
| 8B post-prediction revision (Lesson 4) | 7% (30 samples) | lesson4_results.json |
| 8B ACTUALLY_REASONS rate | 100% | lesson4_results.json |
| Tool call JSON validity | 100% | Phase 1 + Step 0.9 |
| Mechanical proof completion | 10/10 | tool_loop_results.json |
| Mechanical proof avg tool calls | 2.8 | tool_loop_results.json |
| Mechanical proof save/break | 5/2 = +3 net | tool_loop_results.json |
| Image dimensions | 504 x 336 pixels | Phase 1 |
| Waypoint grid size | 63 x 63 | visual_tools.py |
| Zoom upscale factor | 4x (LANCZOS) | visual_tools.py |
| FAISS index size | 8,613 vectors x 512 dims | visual_tools.py |
| Image injection verified | 9/10 zoom-aware (Task 8) | phase1_tasks6to10_results.json |
| Waypoint comprehension | 10/10 (Task 9) | phase1_tasks6to10_results.json |
