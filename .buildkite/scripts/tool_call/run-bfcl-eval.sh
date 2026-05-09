#!/bin/bash
# Run BFCL (Berkeley Function Call Leaderboard) tool-calling correctness
# evaluation against a local vLLM server.
#
# Usage:
#   # Run with defaults (gpt-oss-20b, multi_turn)
#   bash .buildkite/scripts/tool_call/run-bfcl-eval.sh
#
#   # Run with gpt-oss-120b and multiple test categories
#   BFCL_MODEL="openai/gpt-oss-120b" BFCL_TP_SIZE=4 \
#     BFCL_TEST_CATEGORY="live_simple, multiple, parallel_multiple" \
#     bash .buildkite/scripts/tool_call/run-bfcl-eval.sh
#
#   # Chain both API types (use BFCL_OUTPUT_DIR to avoid overwriting results)
#   BFCL_OUTPUT_DIR=./bfcl-chat-completions BFCL_API_TYPE=chat_completions \
#     bash .buildkite/scripts/tool_call/run-bfcl-eval.sh && \
#   BFCL_OUTPUT_DIR=./bfcl-responses BFCL_API_TYPE=responses \
#     bash .buildkite/scripts/tool_call/run-bfcl-eval.sh
#
# Environment variables (all optional, with defaults):
#   BFCL_MODEL          - HF model name (default: openai/gpt-oss-20b)
#   BFCL_API_TYPE       - API type: "chat_completions" or "responses" (default: chat_completions)
#   BFCL_OUTPUT_DIR     - Directory for BFCL results (default: current working directory)
#   BFCL_TEST_CATEGORY  - BFCL test categories (default: multi_turn)
#   BFCL_TOOL_CALL_PARSER - Tool call parser name (default: openai)
#   BFCL_NUM_THREADS    - Threads for BFCL generate (default: 8)
#   BFCL_TP_SIZE        - Tensor parallel size (default: 1)
#   BFCL_MAX_MODEL_LEN  - Max model length (default: 4096)
#   BFCL_PORT           - Server port (default: 8000)
#   BFCL_REASONING_PARSER - Reasoning parser name (default: disabled)
#   BFCL_TEMPERATURE    - Temperature (default: 0.0)
#   BFCL_EXTRA_ARGS     - Additional vLLM server args

set -euo pipefail

# ---- Configuration ----
MODEL="${BFCL_MODEL:-openai/gpt-oss-20b}"
API_TYPE="${BFCL_API_TYPE:-chat_completions}"
OUTPUT_DIR="${BFCL_OUTPUT_DIR:-}"
TEST_CATEGORY="${BFCL_TEST_CATEGORY:-multi_turn}"
TOOL_CALL_PARSER="${BFCL_TOOL_CALL_PARSER:-openai}"
NUM_THREADS="${BFCL_NUM_THREADS:-8}"
TP_SIZE="${BFCL_TP_SIZE:-1}"
MAX_MODEL_LEN="${BFCL_MAX_MODEL_LEN:-4096}"
PORT="${BFCL_PORT:-8000}"
REASONING_PARSER="${BFCL_REASONING_PARSER:-}"
TEMPERATURE="${BFCL_TEMPERATURE:-0.0}"
EXTRA_ARGS="${BFCL_EXTRA_ARGS:-}"

# Set up output directory
if [ -n "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
fi

echo "============================================"
echo "BFCL Tool Call Correctness Evaluation"
echo "============================================"
echo "Model:          $MODEL"
echo "Tool parser:    $TOOL_CALL_PARSER"
echo "API type:       $API_TYPE"
echo "Output dir:     ${OUTPUT_DIR:-<cwd>}"
echo "Test category:  $TEST_CATEGORY"
echo "TP size:        $TP_SIZE"
echo "Max model len:  $MAX_MODEL_LEN"
echo "Port:           $PORT"
echo "Num threads:    $NUM_THREADS"
echo "============================================"

# ---- Install bfcl-eval if missing ----
if ! python3 -c "import bfcl_eval" 2>/dev/null; then
    echo "Installing bfcl-eval..."
    pip install "bfcl-eval>=2025.10.20.1,<2026"
fi

# ---- Cleanup handler ----
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (pid=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Remove BFCL lock files (created by filelock for thread-safe writes)
    rm -rf .file_locks/
    if [ -n "${OUTPUT_DIR:-}" ]; then
        rm -rf "$OUTPUT_DIR/.file_locks/"
    fi
}
trap cleanup EXIT

# ---- Start vLLM server ----
echo "Starting vLLM server..."

SERVE_ARGS=(
    "$MODEL"
    --port "$PORT"
    --enable-auto-tool-choice
    --tool-call-parser "$TOOL_CALL_PARSER"
    --tensor-parallel-size "$TP_SIZE"
    --max-model-len "$MAX_MODEL_LEN"
    --enforce-eager
    --no-enable-prefix-caching
)

# Append reasoning parser if specified
if [ -n "$REASONING_PARSER" ]; then
    SERVE_ARGS+=(--reasoning-parser "$REASONING_PARSER")
fi

# Append any extra args
if [ -n "$EXTRA_ARGS" ]; then
    read -ra EXTRA_ARGS_ARRAY <<< "$EXTRA_ARGS"
    SERVE_ARGS+=("${EXTRA_ARGS_ARRAY[@]}")
fi

echo "Command: vllm serve ${SERVE_ARGS[*]}"
vllm serve "${SERVE_ARGS[@]}" &
SERVER_PID=$!

# ---- Wait for server to be ready ----
echo "Waiting for vLLM server to start (timeout: 600s)..."
SECONDS_WAITED=0
until curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; do
    if [ $SECONDS_WAITED -ge 600 ]; then
        echo ""
        echo "ERROR: vLLM server failed to start within 600s"
        exit 1
    fi
    if (( SECONDS_WAITED % 30 == 0 && SECONDS_WAITED > 0 )); then
        echo "  Still waiting... (${SECONDS_WAITED}s elapsed)"
    fi
    sleep 2
    SECONDS_WAITED=$((SECONDS_WAITED + 2))
done
echo "vLLM server is ready. (started in ${SECONDS_WAITED}s)"

# ---- Run BFCL evaluation ----
# bfcl-eval has no CLI entry point; generate() and evaluate() are Typer
# functions that must be called from Python. The MODEL_CONFIG_MAPPING must
# be patched in-process so BFCL knows to use the OpenAI-compatible handler
# against our local vLLM server.
bfcl_exit_code=0
python3 - "$MODEL" "$TEST_CATEGORY" "$NUM_THREADS" "$PORT" "$API_TYPE" "$TEMPERATURE" "$OUTPUT_DIR" << 'PYEOF' || bfcl_exit_code=$?
import os
import sys

model = sys.argv[1]
test_category = sys.argv[2]
num_threads = int(sys.argv[3])
port = sys.argv[4]
api_type = sys.argv[5]
temperature = float(sys.argv[6])
output_dir = sys.argv[7] if len(sys.argv) > 7 and sys.argv[7] else os.getcwd()

os.environ["OPENAI_BASE_URL"] = f"http://localhost:{port}/v1"
os.environ["OPENAI_API_KEY"] = "dummy"
os.environ["BFCL_PROJECT_ROOT"] = output_dir

import bfcl_eval.constants.model_config as bfcl_model_config
from bfcl_eval.constants.model_config import ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)
from bfcl_eval.model_handler.api_inference.openai_response import (
    OpenAIResponsesHandler,
)

if api_type == "responses":
    handler = OpenAIResponsesHandler
else:
    handler = OpenAICompletionsHandler

bfcl_model_config.MODEL_CONFIG_MAPPING[model] = ModelConfig(
    model_name=model,
    display_name=f"{model} (FC) (vLLM)",
    url=f"https://huggingface.co/{model}",
    org="",
    license="apache-2.0",
    model_handler=handler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=True,
)

from bfcl_eval.__main__ import evaluate, generate
import inspect
import typer


def _get_default_kwargs(function):
    kwargs = {}
    for k, v in inspect.signature(function).parameters.items():
        if v.default is not inspect.Parameter.empty:
            default = v.default
            if isinstance(default, typer.models.OptionInfo):
                default = default.default
            kwargs[k] = default
    return kwargs


# ---- generate ----
print(f"=== BFCL generate: model={model} test_category={test_category} ===")
gen_kwargs = _get_default_kwargs(generate)
gen_kwargs["model"] = [model]
gen_kwargs["test_category"] = [c.strip() for c in test_category.split(",")]
gen_kwargs["skip_server_setup"] = True
gen_kwargs["num_threads"] = num_threads
gen_kwargs["temperature"] = temperature
generate(**gen_kwargs)

# ---- evaluate ----
print(f"=== BFCL evaluate: model={model} test_category={test_category} ===")
eval_kwargs = _get_default_kwargs(evaluate)
eval_kwargs["model"] = [model]
eval_kwargs["test_category"] = [c.strip() for c in test_category.split(",")]
evaluate(**eval_kwargs)

print("=== BFCL evaluation completed successfully ===")
PYEOF

# ---- Upload results to buildkite ----
if command -v buildkite-agent &>/dev/null; then
    if [ $bfcl_exit_code -eq 0 ]; then
        STYLE="success"
        STATUS="PASSED"
    else
        STYLE="error"
        STATUS="FAILED"
    fi

    buildkite-agent annotate --style "$STYLE" --context "bfcl-results" <<EOF
### BFCL Tool Call Correctness - ${STATUS}
- **Model:** \`${MODEL}\`
- **Parser:** \`${TOOL_CALL_PARSER}\`
- **API type:** \`${API_TYPE}\`
- **Test category:** \`${TEST_CATEGORY}\`
EOF

    # BFCL writes results to $BFCL_PROJECT_ROOT/result/ and scores to
    # $BFCL_PROJECT_ROOT/score/
    RESULTS_ROOT="${OUTPUT_DIR:-.}"
    if [ -d "$RESULTS_ROOT/result" ]; then
        buildkite-agent artifact upload "$RESULTS_ROOT/result/**/*"
    fi
    if [ -d "$RESULTS_ROOT/score" ]; then
        buildkite-agent artifact upload "$RESULTS_ROOT/score/**/*"
    fi
fi

exit $bfcl_exit_code
