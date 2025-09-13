#!/bin/bash
# GTS-based disaggregated prefill/decode test based on v1 pattern

rm -rf prefill_results.json

# The directory of current script
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Create timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "🧪 Running GTS Disaggregated Test (v1 pattern)"
echo "📁 Logs will be saved to: $LOG_DIR"
echo "=" x 50

# Set environment for vLLM
export PYTHONPATH=/serenity/scratch/anirudha/vllm:$PYTHONPATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_PLATFORM="cuda"

echo "Step 1: Running prefill worker (kv_producer)..."
CUDA_VISIBLE_DEVICES=0 python3 "$SCRIPT_DIR/prefill_gts.py" 2>&1 | tee "$LOG_DIR/prefill_gts.log"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Step 2: Running decode worker (kv_consumer)..."
    CUDA_VISIBLE_DEVICES=1 python3 "$SCRIPT_DIR/decode_gts.py" 2>&1 | tee "$LOG_DIR/decode_gts.log"
else
    echo "❌ Prefill failed, skipping decode"
    exit 1
fi

echo "🎉 GTS disaggregated test completed!"
echo "📋 Check logs in: $LOG_DIR"
echo "   - $LOG_DIR/prefill_gts.log"
echo "   - $LOG_DIR/decode_gts.log"