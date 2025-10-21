#!/bin/bash
# Script to compare self_specs vs self_spec_ngram performance

set -e

# Default parameters
NUM_PROMPTS=10
NUM_SPEC_TOKENS=8
MAX_TOKENS=64
PROMPT_LOOKUP_MAX=3
SINK_SIZE=32
RECENT_RATIO=0.05

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-prompts)
            NUM_PROMPTS="$2"
            shift 2
            ;;
        --num-spec-tokens)
            NUM_SPEC_TOKENS="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --prompt-lookup-max)
            PROMPT_LOOKUP_MAX="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num-prompts N] [--num-spec-tokens N] [--max-tokens N] [--prompt-lookup-max N]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Self-Speculative Decoding Method Comparison"
echo "============================================================"
echo "Configuration:"
echo "  Num prompts: $NUM_PROMPTS"
echo "  Num speculative tokens (threshold): $NUM_SPEC_TOKENS"
echo "  Max output tokens: $MAX_TOKENS"
echo "  N-gram window (max): $PROMPT_LOOKUP_MAX"
echo "  Sink size: $SINK_SIZE"
echo "  Recent ratio: $RECENT_RATIO"
echo "============================================================"
echo ""

# Run baseline self_specs
echo "▶ Running BASELINE: self_specs"
echo "============================================================"
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts $NUM_PROMPTS \
    --enable_sspec \
    --sspec_method self_specs \
    --num_speculative_tokens $NUM_SPEC_TOKENS \
    --sink_size $SINK_SIZE \
    --recent_ratio $RECENT_RATIO \
    --verbose \
    2>&1 | tee /tmp/self_specs_output.txt

BASELINE_TIME=$(grep "Generation time:" /tmp/self_specs_output.txt | awk '{print $3}')
echo ""
echo "Baseline (self_specs) time: ${BASELINE_TIME}s"
echo ""

# Run self_spec_ngram
echo "▶ Running WITH N-GRAM: self_spec_ngram"
echo "============================================================"
python offline_inference_self_spec.py \
    --dataset_name debug \
    --num_prompts $NUM_PROMPTS \
    --enable_sspec \
    --sspec_method self_spec_ngram \
    --num_speculative_tokens $NUM_SPEC_TOKENS \
    --prompt_lookup_max $PROMPT_LOOKUP_MAX \
    --prompt_lookup_min 1 \
    --sink_size $SINK_SIZE \
    --recent_ratio $RECENT_RATIO \
    --verbose \
    2>&1 | tee /tmp/self_spec_ngram_output.txt

NGRAM_TIME=$(grep "Generation time:" /tmp/self_spec_ngram_output.txt | awk '{print $3}')
echo ""
echo "N-gram assisted (self_spec_ngram) time: ${NGRAM_TIME}s"
echo ""

# Calculate speedup
echo "============================================================"
echo "COMPARISON RESULTS"
echo "============================================================"
echo "Baseline (self_specs):      ${BASELINE_TIME}s"
echo "With n-gram (self_spec_ngram): ${NGRAM_TIME}s"

if command -v bc &> /dev/null; then
    SPEEDUP=$(echo "scale=2; $BASELINE_TIME / $NGRAM_TIME" | bc)
    echo "Speedup: ${SPEEDUP}x"
else
    echo "Speedup: (install 'bc' to calculate)"
fi
echo "============================================================"

# Extract and compare acceptance metrics
echo ""
echo "Extracting acceptance metrics..."
echo ""

echo "Baseline (self_specs) metrics:"
grep -A 5 "Mean acceptance length:" /tmp/self_specs_output.txt || echo "No metrics found"

echo ""
echo "N-gram assisted (self_spec_ngram) metrics:"
grep -A 5 "Mean acceptance length:" /tmp/self_spec_ngram_output.txt || echo "No metrics found"

echo ""
echo "Full outputs saved to:"
echo "  Baseline: /tmp/self_specs_output.txt"
echo "  N-gram:   /tmp/self_spec_ngram_output.txt"
