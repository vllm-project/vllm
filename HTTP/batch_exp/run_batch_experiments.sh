#!/bin/bash

# Batch Inference Experiment Launcher Script
# This script provides easy-to-use commands for running different types of
# batch inference experiments with vLLM.

set -e

# Default values
MODEL="meta-llama/Llama-2-7b-chat-hf"
BATCH_SIZES="1,4,8,16"
SEQ_LENS="512,1024,2048"
OUTPUT_DIR="experiment_results"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "vLLM Batch Inference Experiment Launcher"
echo "=========================================="

# Function to run experiment
run_experiment() {
    local name=$1
    local tp=$2
    local pp=$3
    local dp=$4
    local extra_args=$5
    
    echo "Running experiment: $name"
    echo "Config: TP=$tp, PP=$pp, DP=$dp"
    echo "Extra args: $extra_args"
    echo "----------------------------------------"
    
    python batch_inference_experiment.py \
        --model "$MODEL" \
        --tensor-parallel-size "$tp" \
        --pipeline-parallel-size "$pp" \
        --data-parallel-size "$dp" \
        --batch-sizes "$BATCH_SIZES" \
        --seq-lens "$SEQ_LENS" \
        --output-file "$OUTPUT_DIR/${name}_results.json" \
        $extra_args
    
    echo "Experiment $name completed!"
    echo ""
}

# Function to run torchrun experiment
run_torchrun_experiment() {
    local name=$1
    local tp=$2
    local pp=$3
    local dp=$4
    local extra_args=$5
    
    local total_gpus=$((tp * pp * dp))
    
    echo "Running torchrun experiment: $name"
    echo "Config: TP=$tp, PP=$pp, DP=$dp (Total GPUs: $total_gpus)"
    echo "Extra args: $extra_args"
    echo "----------------------------------------"
    
    torchrun --nproc-per-node="$total_gpus" batch_inference_experiment.py \
        --model "$MODEL" \
        --tensor-parallel-size "$tp" \
        --pipeline-parallel-size "$pp" \
        --data-parallel-size "$dp" \
        --distributed-executor-backend "external_launcher" \
        --batch-sizes "$BATCH_SIZES" \
        --seq-lens "$SEQ_LENS" \
        --output-file "$OUTPUT_DIR/${name}_results.json" \
        $extra_args
    
    echo "Torchrun experiment $name completed!"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch-sizes)
            BATCH_SIZES="$2"
            shift 2
            ;;
        --seq-lens)
            SEQ_LENS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS] EXPERIMENT_TYPE"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model to use (default: $MODEL)"
            echo "  --batch-sizes SIZES    Comma-separated batch sizes (default: $BATCH_SIZES)"
            echo "  --seq-lens LENS        Comma-separated sequence lengths (default: $SEQ_LENS)"
            echo "  --output-dir DIR       Output directory (default: $OUTPUT_DIR)"
            echo "  --help, -h             Show this help message"
            echo ""
            echo "Experiment Types:"
            echo "  tensor-parallel        Run tensor parallel experiments (TP=1,2,4,8)"
            echo "  pipeline-parallel      Run pipeline parallel experiments (PP=1,2,4)"
            echo "  mixed-parallel         Run mixed TP+PP experiments"
            echo "  data-parallel          Run data parallel experiments (DP=1,2,4)"
            echo "  scaling-study          Run comprehensive scaling study"
            echo "  custom TP=x PP=y DP=z  Run custom parallel configuration"
            echo ""
            echo "Examples:"
            echo "  $0 tensor-parallel"
            echo "  $0 --model llama-2-13b mixed-parallel"
            echo "  $0 --batch-sizes 1,2,4 custom TP=2 PP=2 DP=1"
            exit 0
            ;;
        *)
            EXPERIMENT_TYPE="$1"
            shift
            ;;
    esac
done

if [[ -z "$EXPERIMENT_TYPE" ]]; then
    echo "Error: No experiment type specified"
    echo "Use --help for usage information"
    exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Batch sizes: $BATCH_SIZES"
echo "  Sequence lengths: $SEQ_LENS"
echo "  Output directory: $OUTPUT_DIR"
echo ""

case "$EXPERIMENT_TYPE" in
    "tensor-parallel")
        echo "Running Tensor Parallel Experiments"
        echo "=================================="
        run_experiment "tp_1" 1 1 1
        run_experiment "tp_2" 2 1 1
        run_experiment "tp_4" 4 1 1
        run_experiment "tp_8" 8 1 1
        ;;
    
    "pipeline-parallel")
        echo "Running Pipeline Parallel Experiments"
        echo "===================================="
        run_experiment "pp_1" 1 1 1
        run_experiment "pp_2" 1 2 1
        run_experiment "pp_4" 1 4 1
        ;;
    
    "mixed-parallel")
        echo "Running Mixed Tensor + Pipeline Parallel Experiments"
        echo "==================================================="
        run_experiment "tp2_pp2" 2 2 1
        run_experiment "tp4_pp2" 4 2 1
        run_experiment "tp2_pp4" 2 4 1
        run_experiment "tp4_pp4" 4 4 1
        ;;
    
    "data-parallel")
        echo "Running Data Parallel Experiments"
        echo "================================="
        run_experiment "dp_1" 1 1 1
        run_experiment "dp_2" 1 1 2
        run_experiment "dp_4" 1 1 4
        ;;
    
    "scaling-study")
        echo "Running Comprehensive Scaling Study"
        echo "==================================="
        # Single GPU baseline
        run_experiment "baseline" 1 1 1
        
        # Tensor parallel scaling
        run_experiment "tp_2" 2 1 1
        run_experiment "tp_4" 4 1 1
        run_experiment "tp_8" 8 1 1
        
        # Pipeline parallel scaling
        run_experiment "pp_2" 1 2 1
        run_experiment "pp_4" 1 4 1
        
        # Mixed scaling
        run_experiment "tp2_pp2" 2 2 1
        run_experiment "tp4_pp2" 4 2 1
        
        # Data parallel scaling
        run_experiment "dp_2" 1 1 2
        run_experiment "dp_4" 1 1 4
        ;;
    
    "torchrun-tensor-parallel")
        echo "Running Torchrun Tensor Parallel Experiments"
        echo "============================================"
        run_torchrun_experiment "torchrun_tp_2" 2 1 1
        run_torchrun_experiment "torchrun_tp_4" 4 1 1
        run_torchrun_experiment "torchrun_tp_8" 8 1 1
        ;;
    
    "torchrun-mixed-parallel")
        echo "Running Torchrun Mixed Parallel Experiments"
        echo "==========================================="
        run_torchrun_experiment "torchrun_tp2_pp2" 2 2 1
        run_torchrun_experiment "torchrun_tp4_pp2" 4 2 1
        ;;
    
    custom)
        # Parse custom configuration
        TP=1
        PP=1
        DP=1
        
        shift  # Remove "custom" from arguments
        while [[ $# -gt 0 ]]; do
            case $1 in
                TP=*)
                    TP="${1#TP=}"
                    shift
                    ;;
                PP=*)
                    PP="${1#PP=}"
                    shift
                    ;;
                DP=*)
                    DP="${1#DP=}"
                    shift
                    ;;
                *)
                    echo "Unknown custom parameter: $1"
                    exit 1
                    ;;
            esac
        done
        
        echo "Running Custom Parallel Configuration"
        echo "===================================="
        run_experiment "custom_tp${TP}_pp${PP}_dp${DP}" "$TP" "$PP" "$DP"
        ;;
    
    *)
        echo "Error: Unknown experiment type '$EXPERIMENT_TYPE'"
        echo "Use --help for available experiment types"
        exit 1
        ;;
esac

echo "All experiments completed!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To analyze results, you can use:"
echo "  python -c \"import json; data=json.load(open('$OUTPUT_DIR/*_results.json')); print('Results:', len(data), 'experiments')\"" 