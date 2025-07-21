# Batch Inference Experiment Scripts for vLLM

This directory contains comprehensive scripts for experimenting with batch inference using different parallel configurations (tensor parallel, pipeline parallel, data parallel) in vLLM.

## Files Overview

- **`batch_inference_experiment.py`** - Main experiment script with configurable parallel settings
- **`run_batch_experiments.sh`** - Convenient launcher script for different experiment types
- **`analyze_results.py`** - Results analysis and visualization script
- **`README_batch_inference.md`** - This documentation file

## Quick Start

### 1. Basic Tensor Parallel Experiment

```bash
# Run tensor parallel experiments with TP=1,2,4,8
./run_batch_experiments.sh tensor-parallel

# Custom model and batch sizes
./run_batch_experiments.sh --model "meta-llama/Llama-2-13b-chat-hf" \
                          --batch-sizes "1,2,4,8" \
                          tensor-parallel
```

### 2. Tensor + Pipeline Parallel Experiment

```bash
# Run mixed TP+PP experiments
./run_batch_experiments.sh mixed-parallel

# Using torchrun for better distributed execution
./run_batch_experiments.sh torchrun-mixed-parallel
```

### 3. Data Parallel Experiment

```bash
# Run data parallel experiments
./run_batch_experiments.sh data-parallel
```

### 4. Comprehensive Scaling Study

```bash
# Run all types of experiments for comprehensive analysis
./run_batch_experiments.sh scaling-study
```

## Detailed Usage

### Main Experiment Script

The `batch_inference_experiment.py` script provides fine-grained control over all parameters:

```bash
python batch_inference_experiment.py \
    --model "meta-llama/Llama-2-7b-chat-hf" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --data-parallel-size 1 \
    --batch-sizes "1,4,8,16" \
    --seq-lens "512,1024,2048" \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.8 \
    --output-file "my_experiment_results.json"
```

#### Key Parameters

**Parallel Configuration:**
- `--tensor-parallel-size` - Number of GPUs for tensor parallelism
- `--pipeline-parallel-size` - Number of GPUs for pipeline parallelism  
- `--data-parallel-size` - Number of GPUs for data parallelism
- `--distributed-executor-backend` - Backend for distributed execution ("ray", "mp", "external_launcher")

**Batch Configuration:**
- `--batch-sizes` - Comma-separated list of batch sizes to test
- `--seq-lens` - Comma-separated list of sequence lengths to test

**Model Configuration:**
- `--model` - Model name or path
- `--max-model-len` - Maximum model length
- `--dtype` - Model data type ("auto", "float16", "bfloat16", etc.)

**Performance Configuration:**
- `--max-num-seqs` - Maximum number of sequences
- `--max-num-batched-tokens` - Maximum number of batched tokens
- `--gpu-memory-utilization` - GPU memory utilization (0.0-1.0)

**Sampling Configuration:**
- `--temperature` - Sampling temperature
- `--top-p` - Top-p sampling parameter
- `--max-tokens` - Maximum tokens to generate

### Launcher Script

The `run_batch_experiments.sh` script provides predefined experiment configurations:

#### Available Experiment Types

1. **`tensor-parallel`** - Tests TP=1,2,4,8
2. **`pipeline-parallel`** - Tests PP=1,2,4  
3. **`mixed-parallel`** - Tests TP+PP combinations (TP2_PP2, TP4_PP2, etc.)
4. **`data-parallel`** - Tests DP=1,2,4
5. **`scaling-study`** - Comprehensive study with all configurations
6. **`torchrun-tensor-parallel`** - Tensor parallel using torchrun
7. **`torchrun-mixed-parallel`** - Mixed parallel using torchrun
8. **`custom TP=x PP=y DP=z`** - Custom parallel configuration

#### Launcher Script Options

```bash
./run_batch_experiments.sh [OPTIONS] EXPERIMENT_TYPE

Options:
  --model MODEL          Model to use (default: meta-llama/Llama-2-7b-chat-hf)
  --batch-sizes SIZES    Comma-separated batch sizes (default: 1,4,8,16)
  --seq-lens LENS        Comma-separated sequence lengths (default: 512,1024,2048)
  --output-dir DIR       Output directory (default: experiment_results)
  --help, -h             Show help message
```

#### Examples

```bash
# Basic tensor parallel with custom model
./run_batch_experiments.sh --model "meta-llama/Llama-2-13b-chat-hf" tensor-parallel

# Custom batch sizes and sequence lengths
./run_batch_experiments.sh --batch-sizes "1,2,4,8,16" --seq-lens "256,512,1024" mixed-parallel

# Custom parallel configuration
./run_batch_experiments.sh custom TP=4 PP=2 DP=1

# Multi-node experiment (run on each node)
./run_batch_experiments.sh --model "meta-llama/Llama-2-70b-chat-hf" scaling-study
```

### Results Analysis

After running experiments, analyze the results:

```bash
# Analyze all results in a directory
python analyze_results.py --input experiment_results

# Analyze a single result file
python analyze_results.py --input experiment_results/tp_2_results.json

# Generate analysis without plots
python analyze_results.py --input experiment_results --no-plots

# Custom output directory
python analyze_results.py --input experiment_results --output-dir my_analysis
```

The analysis script generates:
- **Summary statistics** - Best configurations, performance metrics
- **Visualizations** - Throughput vs batch size, latency analysis, scaling efficiency
- **Detailed CSV report** - All results in tabular format

## Parallel Configuration Guide

### Tensor Parallelism (TP)
- **Purpose**: Splits model layers across multiple GPUs
- **Use case**: Large models that don't fit on single GPU
- **Configuration**: `--tensor-parallel-size N`
- **Example**: `TP=2` splits attention and MLP layers across 2 GPUs

### Pipeline Parallelism (PP)  
- **Purpose**: Splits model stages across multiple GPUs
- **Use case**: Very large models with many layers
- **Configuration**: `--pipeline-parallel-size N`
- **Example**: `PP=2` splits model into 2 stages, each on different GPU

### Data Parallelism (DP)
- **Purpose**: Processes different batches on different GPUs
- **Use case**: High throughput with multiple model replicas
- **Configuration**: `--data-parallel-size N`
- **Example**: `DP=2` runs 2 model replicas, each processing different batches

### Combined Configurations

**TP + PP (Recommended for large models):**
```bash
# 4 GPUs: 2 for tensor parallel, 2 for pipeline parallel
python batch_inference_experiment.py \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 2 \
    --batch-sizes "1,2,4,8"
```

**TP + DP (High throughput):**
```bash
# 4 GPUs: 2 for tensor parallel, 2 for data parallel
python batch_inference_experiment.py \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --batch-sizes "1,4,8,16"
```

## Performance Optimization Tips

### 1. Memory Management
- Start with `--gpu-memory-utilization 0.8` and adjust based on OOM errors
- Use `--max-num-batched-tokens` to control memory usage
- Monitor memory usage in results

### 2. Batch Size Selection
- Start with small batch sizes (1, 2, 4) and scale up
- Larger batch sizes generally improve throughput but may increase latency
- Find the sweet spot for your use case

### 3. Sequence Length Impact
- Longer sequences require more memory and computation
- Test realistic sequence lengths for your application
- Consider using chunked prefill for very long sequences

### 4. Parallel Configuration Selection
- **Small models (<7B)**: Start with tensor parallelism
- **Medium models (7B-30B)**: Use TP + PP combination
- **Large models (>30B)**: Use TP + PP + DP combination

### 5. Distributed Backend Selection
- **`ray`**: Good for multi-node, complex topologies
- **`mp`**: Good for single-node, simple topologies  
- **`external_launcher`**: Good for torchrun integration

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce `--gpu-memory-utilization`
   - Reduce `--max-num-batched-tokens`
   - Use smaller batch sizes
   - Increase tensor/pipeline parallel size

2. **Slow Performance**
   - Check GPU utilization
   - Try different batch sizes
   - Experiment with different parallel configurations
   - Use `--enforce-eager` for debugging

3. **Distributed Setup Issues**
   - Ensure all nodes can communicate
   - Check firewall settings
   - Use `--distributed-executor-backend external_launcher` with torchrun

4. **Model Loading Issues**
   - Use `--trust-remote-code` for custom models
   - Check model path and access permissions
   - Verify model format compatibility

### Debugging Tips

```bash
# Enable eager mode for debugging
python batch_inference_experiment.py --enforce-eager --num-runs 1

# Test with minimal configuration
python batch_inference_experiment.py \
    --tensor-parallel-size 1 \
    --batch-sizes "1" \
    --seq-lens "512" \
    --num-runs 1

# Check GPU memory usage
nvidia-smi
```

## Example Workflows

### Workflow 1: Model Performance Characterization

```bash
# 1. Run baseline single-GPU experiment
./run_batch_experiments.sh --model "my-model" tensor-parallel

# 2. Analyze results
python analyze_results.py --input experiment_results

# 3. Run scaling study if needed
./run_batch_experiments.sh --model "my-model" scaling-study
```

### Workflow 2: Production Configuration Optimization

```bash
# 1. Test different parallel configurations
./run_batch_experiments.sh --batch-sizes "1,4,8,16,32" mixed-parallel

# 2. Focus on best configurations
./run_batch_experiments.sh custom TP=4 PP=2 DP=1

# 3. Fine-tune batch sizes for production
python batch_inference_experiment.py \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 2 \
    --batch-sizes "8,16,32,64" \
    --seq-lens "1024,2048"
```

### Workflow 3: Multi-Node Deployment

```bash
# Node 0
./run_batch_experiments.sh --model "large-model" \
    --master-addr "192.168.1.100" \
    --master-port 29500 \
    scaling-study

# Node 1 (run simultaneously)
./run_batch_experiments.sh --model "large-model" \
    --master-addr "192.168.1.100" \
    --master-port 29500 \
    scaling-study
```

## Output Files

### Experiment Results
- **JSON files**: Raw experiment results with detailed metrics
- **CSV report**: Tabular format for further analysis
- **Visualizations**: PNG plots showing performance characteristics

### Key Metrics Measured
- **Throughput**: Tokens per second, requests per second
- **Latency**: End-to-end response time
- **Memory Usage**: GPU memory consumption
- **Efficiency**: Throughput per GPU
- **Scaling**: Performance vs number of GPUs

## Contributing

To extend these scripts:

1. **Add new experiment types** to `run_batch_experiments.sh`
2. **Add new metrics** to `batch_inference_experiment.py`
3. **Add new visualizations** to `analyze_results.py`
4. **Update documentation** in this README

## Dependencies

Required Python packages:
```
vllm
torch
pandas
matplotlib
seaborn
```

Install with:
```bash
pip install vllm torch pandas matplotlib seaborn
``` 