#!/usr/bin/env bash
set -euo pipefail

# Try loading DeepSeek-V3.2 with successive transformers versions
# starting at 5.0.0 to find the minimum compatible version.

MODEL="nvidia/DeepSeek-V3.2-NVFP4"
LOG_DIR="results/transformers_compat"
mkdir -p "$LOG_DIR"

# ── Versions to try (ascending order) ────────────────────────────
# Add / remove entries as new releases come out.
VERSIONS=(
  "5.0.0"
  "5.0.1"
  "5.0.2"
  "5.1.0"
  "5.1.1"
  "5.1.2"
  "5.2.0"
  "5.2.1"
  "5.2.2"
  "5.3.0"
  "5.3.1"
  "5.3.2"
  "5.4.0"
  "5.4.1"
  "5.4.2"
  "5.5.0"
)

# ── Helpers ───────────────────────────────────────────────────────
PYTHON="${PYTHON:-$(command -v python3)}"

try_load() {
  # Attempt to load the model config + tokenizer (no weights / no GPU).
  # Returns 0 on success, non-zero on failure.
  "$PYTHON" -c "
import sys, traceback
try:
    from transformers import AutoConfig, AutoTokenizer
    print(f'  Loading config for ${MODEL} …')
    cfg = AutoConfig.from_pretrained('${MODEL}', trust_remote_code=True)
    print(f'  Config OK – model_type={getattr(cfg, \"model_type\", \"?\")}')

    print(f'  Loading tokenizer for ${MODEL} …')
    tok = AutoTokenizer.from_pretrained('${MODEL}', trust_remote_code=True)
    print(f'  Tokenizer OK – vocab_size={tok.vocab_size}')

    # Quick sanity: try to instantiate the model class (meta device, no weights)
    import torch
    from transformers import AutoModelForCausalLM
    print(f'  Instantiating model on meta device …')
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    param_count = sum(p.numel() for p in model.parameters())
    print(f'  Model OK – {param_count/1e9:.1f}B params (meta)')
    sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(1)
"
}

# ── Main loop ─────────────────────────────────────────────────────
echo "============================================================"
echo "Transformers version sweep for: $MODEL"
echo "============================================================"
echo ""

SUMMARY=()

for VER in "${VERSIONS[@]}"; do
  LOG_FILE="${LOG_DIR}/transformers_${VER}.log"
  echo "------------------------------------------------------------"
  echo ">>> Trying transformers==${VER}"
  echo "------------------------------------------------------------"

  # Install the target version (suppress pip noise)
  if ! uv pip install "transformers==${VER}" > "$LOG_FILE" 2>&1; then
    echo "  SKIP – transformers==${VER} not found on PyPI or install failed."
    SUMMARY+=("${VER}  SKIP (install failed)")
    continue
  fi

  # Confirm installed version
  INSTALLED=$("$PYTHON" -c "import transformers; print(transformers.__version__)")
  echo "  Installed: transformers==${INSTALLED}"

  # Attempt the load
  if try_load >> "$LOG_FILE" 2>&1; then
    echo "  ✅  SUCCESS with transformers==${VER}"
    SUMMARY+=("${VER}  SUCCESS")
  else
    echo "  ❌  FAILED  with transformers==${VER}"
    SUMMARY+=("${VER}  FAILED")
  fi

  echo "  Log: $LOG_FILE"
  echo ""
done

# ── Summary ───────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
printf '%s\n' "${SUMMARY[@]}"
echo "============================================================"
