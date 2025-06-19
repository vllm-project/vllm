# vLLM Profiling and Custom Schedule Planning System

This repository provides a comprehensive system for profiling vLLM performance and generating custom schedule plans based on workload characteristics. The system considers the key factors that affect vLLM performance:

- **C**: Number of input tokens/prompt
- **M**: Number of KV Cache blocks for precomputed tokens
- **B**: Batch size
- **block_size**: KV Cache block size

## Overview

The system consists of several components:

1. **Profiling Workload Generator** (`profiling_workload_generator.py`) - Generates custom workloads and runs profiling
2. **Advanced Profiling Analyzer** (`advanced_profiling_analyzer.py`) - Provides detailed performance analysis
3. **Custom Schedule Planner** (`custom_schedule_planner.py`) - Generates optimized schedule plans
4. **Complete Workflow Example** (`example_profiling_workflow.py`) - Demonstrates the full workflow

## Installation

1. Install vLLM and its dependencies:
```bash
pip install vllm
```

2. Install additional dependencies:
```bash
pip install matplotlib numpy psutil
```

3. Clone or download the profiling scripts to your working directory.

## Quick Start

### Basic Profiling

Run a quick profiling session with default parameters:

```bash
python profiling_workload_generator.py --C 512 --M 32 --B 32 --block-size 16 --profiling-type mixed
```

### Advanced Profiling with Analysis

Run comprehensive profiling with detailed analysis:

```bash
python advanced_profiling_analyzer.py --C 1024 --M 64 --B 64 --block-size 16 --save-plots
```

### Complete Workflow

Run the complete workflow with multiple configurations:

```bash
python example_profiling_workflow.py --quick-test --target-optimization balanced
```

## Detailed Usage

### 1. Profiling Workload Generator

The `profiling_workload_generator.py` script provides basic profiling capabilities:

```bash
python profiling_workload_generator.py \
    --C 1024 \                    # Number of input tokens
    --M 64 \                      # Number of KV cache blocks
    --B 64 \                      # Batch size
    --block-size 16 \             # KV cache block size
    --profiling-type mixed \      # prefill/decode/mixed
    --num-warmup 3 \              # Number of warmup runs
    --num-measurements 10 \       # Number of measurement runs
    --model microsoft/DialoGPT-medium
```

**Key Parameters:**
- `--C`: Number of input tokens per prompt (affects prefill performance)
- `--M`: Number of KV cache blocks (affects memory usage and cache efficiency)
- `--B`: Batch size (affects throughput and latency trade-offs)
- `--block-size`: KV cache block size (affects memory efficiency)
- `--profiling-type`: Type of profiling to run
  - `prefill`: Only prefill stage profiling
  - `decode`: Only decode stage profiling
  - `mixed`: Both prefill and decode profiling

### 2. Advanced Profiling Analyzer

The `advanced_profiling_analyzer.py` script provides detailed performance analysis:

```bash
python advanced_profiling_analyzer.py \
    --C 1024 \
    --M 64 \
    --B 64 \
    --block-size 16 \
    --save-plots \
    --analysis-file detailed_analysis.json
```

**Features:**
- Detailed bottleneck analysis
- Resource utilization tracking
- Performance visualization plots
- Optimization recommendations
- Custom schedule plan generation

### 3. Custom Schedule Planner

Generate custom schedule plans based on profiling results:

```bash
python custom_schedule_planner.py \
    --profiling-results profiling_results.json \
    --output-file custom_plans.json \
    --generate-parameter-sweep
```

**Features:**
- Workload characteristic analysis
- Bottleneck identification
- Multiple optimization strategies
- Parameter sweep generation

### 4. Complete Workflow

Run the complete profiling and analysis workflow:

```bash
python example_profiling_workflow.py \
    --output-dir profiling_results \
    --model microsoft/DialoGPT-medium \
    --target-optimization balanced \
    --quick-test
```

**Workflow Steps:**
1. Generate workload configurations
2. Run profiling for each configuration
3. Analyze performance characteristics
4. Generate custom schedule plans
5. Save results and generate reports

## Understanding the Results

### Performance Metrics

The system tracks several key performance metrics:

- **Prefill Latency**: Time to process initial prompt tokens
- **Decode Latency**: Time to generate each new token
- **Throughput**: Tokens processed per second
- **Memory Usage**: GPU and CPU memory utilization
- **Batch Efficiency**: How well the scheduler utilizes batches
- **Cache Hit Rate**: KV cache efficiency

### Bottleneck Analysis

The system identifies four main types of bottlenecks:

1. **Compute-bound**: High latency relative to token count
2. **Memory-bound**: High memory usage relative to KV cache size
3. **Scheduling-bound**: Low batch efficiency
4. **Cache-bound**: Low cache hit rate

### Custom Schedule Plans

The system generates several types of optimized schedule plans:

1. **Throughput Optimized**: Maximize tokens per second
2. **Latency Optimized**: Minimize response time
3. **Memory Optimized**: Reduce memory usage
4. **Cache Optimized**: Improve cache efficiency
5. **Balanced**: Balance throughput and latency

## Configuration Examples

### Low Latency Configuration

For applications requiring fast response times:

```python
config = ProfilingConfig(
    C=256,      # Short prompts
    M=16,       # Fewer cache blocks
    B=16,       # Smaller batches
    block_size=16,
    max_num_batched_tokens=4096,
    max_num_seqs=32,
    enable_chunked_prefill=True,
    scheduling_policy="priority",
)
```

### High Throughput Configuration

For applications requiring maximum throughput:

```python
config = ProfilingConfig(
    C=2048,     # Long prompts
    M=128,      # More cache blocks
    B=128,      # Larger batches
    block_size=16,
    max_num_batched_tokens=262144,
    max_num_seqs=256,
    enable_chunked_prefill=True,
    scheduling_policy="fcfs",
)
```

### Memory Efficient Configuration

For memory-constrained environments:

```python
config = ProfilingConfig(
    C=512,      # Medium prompts
    M=32,       # Moderate cache blocks
    B=32,       # Moderate batches
    block_size=8,  # Smaller block size
    max_num_batched_tokens=16384,
    max_num_seqs=64,
    enable_chunked_prefill=False,  # Disable to save memory
    gpu_memory_utilization=0.7,
    swap_space=8.0,
)
```

## Integration with Your Profiling Code

To integrate this system with your existing profiling code:

1. **Import the modules**:
```python
from profiling_workload_generator import ProfilingConfig, ProfilingRunner
from custom_schedule_planner import CustomSchedulePlanner
```

2. **Create a configuration**:
```python
config = ProfilingConfig(
    C=your_input_tokens,
    M=your_kv_cache_blocks,
    B=your_batch_size,
    block_size=your_block_size,
)
```

3. **Run profiling**:
```python
engine = create_profiling_engine(config, model_name)
runner = ProfilingRunner(engine, config)
result = runner.run_mixed_profiling()
```

4. **Generate schedule plans**:
```python
planner = CustomSchedulePlanner(result)
plans = planner.generate_optimized_schedule_plans()
```

## Output Files

The system generates several output files:

- `*_results.json`: Raw profiling results
- `performance_analysis.json`: Detailed performance analysis
- `custom_schedule_plans.json`: Generated schedule plans
- `vllm_performance_analysis.png`: Performance visualization plots

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**:
   - Reduce batch size (`--B`)
   - Reduce max_num_batched_tokens
   - Enable CPU offloading with swap_space

2. **Slow Profiling**:
   - Use `--quick-test` for faster testing
   - Reduce num_measurements
   - Use a smaller model

3. **Import Errors**:
   - Ensure vLLM is properly installed
   - Check Python path and dependencies

### Performance Tips

1. **For accurate results**:
   - Run multiple warmup iterations
   - Use sufficient measurement runs
   - Ensure consistent system load

2. **For faster profiling**:
   - Use smaller models for testing
   - Reduce batch sizes
   - Use quick-test mode

3. **For better analysis**:
   - Profile multiple configurations
   - Use parameter sweeps
   - Compare different optimization strategies

## Advanced Usage

### Custom Workload Generation

You can create custom workload configurations:

```python
configs = [
    {'C': 512, 'M': 32, 'B': 32, 'block_size': 16, 'profiling_type': 'mixed'},
    {'C': 1024, 'M': 64, 'B': 64, 'block_size': 16, 'profiling_type': 'mixed'},
    {'C': 2048, 'M': 128, 'B': 128, 'block_size': 16, 'profiling_type': 'mixed'},
]

workflow = CompleteProfilingWorkflow("results")
results = workflow.run_profiling_suite(configs, "your-model-name")
```

### Parameter Sweep Analysis

Generate comprehensive parameter sweeps:

```python
planner = CustomSchedulePlanner(profiling_result)
sweep_plans = planner.generate_parameter_sweep_plans(
    C_range=[256, 512, 1024, 2048],
    M_range=[16, 32, 64, 128],
    B_range=[16, 32, 64, 128],
    block_size_range=[8, 16, 32]
)
```

## Contributing

To extend the system:

1. **Add new profiling metrics** in `ProfilingRunner`
2. **Implement new optimization strategies** in `CustomSchedulePlanner`
3. **Create custom workload generators** by extending `ProfilingWorkloadGenerator`
4. **Add new visualization plots** in `AdvancedProfilingAnalyzer`

## License

This project is provided as-is for educational and research purposes. Please ensure compliance with vLLM's license and your specific use case requirements.
