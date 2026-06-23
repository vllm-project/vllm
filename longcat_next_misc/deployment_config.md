# LongCat-Next Deployment Configuration

## Optimal Configuration for 8xH20 (46GB each)

### Model Architecture

- **Model**: LongCat-Next (MoE architecture)
- **Total Layers**: 14 (configured in `config.json`)
- **Routed Experts**: 256
- **Hidden Size**: 3584
- **Multimodal**: Visual + Audio encoders

### Tensor Parallelism Setup

```javascript
{
  tensor_parallel_size: 8,
  data_parallel_size: 1,
  pipeline_parallel_size: 1,
  enable_expert_parallel: true,
  enable_ep_weight_filter: true,
}
```

### Memory Configuration

```javascript
{
  dtype: "bfloat16",
  max_model_len: 8192,
  gpu_memory_utilization: 0.90,
  max_num_seqs: 1,
  max_num_batched_tokens: 512,
}
```

### Critical Flags

```javascript
{
  disable_custom_all_reduce: true,  // Required for machines with P2P issues
  enforce_eager: true,              // Skip CUDA graph capture to reduce memory
  mm_encoder_tp_mode: "data",       // Multimodal encoder data parallel mode
  trust_remote_code: true,          // Required for custom model code
}
```

### Environment Variables

```bash
export NCCL_P2P_DISABLE=1           # Disable NCCL P2P for problematic hardware
export VLLM_NCCL_P2P_CGA_SIZE=0     # Disable P2P CGA
export CUDA_LAUNCH_BLOCKING=1       # Make CUDA errors visible (debugging)
export NCCL_DEBUG=INFO              # Verbose NCcl logging
```

### Expert Parallel Weight Filter

- **EP Size**: 8 (one per GPU)
- **Experts per GPU**: 32 out of 256 total
- **Memory Savings**: ~87.5% reduction in expert weight loading per GPU
- **Filter Logic**: Only loads expert weights belonging to local rank

### Weight Loading Pipeline

1. Checkpoint contains layers 0-13 (14 layers total)
2. Weight mapper converts `model.layers.*` → `language_model.layers.*`
3. EP weight filter skips non-local expert weights BEFORE reading from disk
4. Each GPU loads only its 32 local experts
5. Visual and audio towers loaded separately

### Initialization Sequence

1. **NCCL Initialization**: Distributed environment setup
2. **Weight Loading**: Model weights loaded with EP filter
3. **Communication Buffer**: `prepare_communication_buffer_for_model()` (disabled with `--disable-custom-all-reduce`)
4. **Profile Run**: Forward pass with dummy inputs to determine KV cache memory
5. **KV Cache Allocation**: Based on profile results
6. **CUDA Graph Capture**: (skipped with `--enforce-eager`)
7. **Server Ready**: Listening on port 52099

### Known Issues & Workarounds

#### Issue 1: Silent Server Death

- **Symptom**: Server dies after weight loading with no error message
- **Root Cause**: CustomAllreduce IPC buffer allocation fails when P2P unavailable
- **Solution**: `--disable-custom-all-reduce`

#### Issue 2: DMA Faults (Machine-Specific)

- **Symptom**: Kernel logs show `DMAR: [DMA Write NO_PASID]` errors
- **Root Cause**: IOMMU/DMA remapping issues on specific hardware
- **Solution**: Use machines with proper P2P support (e.g., H20 instead of L20)

#### Issue 3: Weight Mismatch

- **Symptom**: `ValueError: Following weights were not initialized from checkpoint`
- **Root Cause**: Checkpoint has 14 layers but config specified 32 or 56
- **Solution**: Set `num_hidden_layers=14` in `config.json` to match checkpoint

#### Issue 4: OOM During Loading

- **Symptom**: `torch.OutOfMemoryError` during weight loading
- **Root Cause**: All expert weights loaded onto single GPU without EP filter
- **Solution**: Enable `--enable-expert-parallel --enable-ep-weight-filter`

### Startup Script

```bash
# File: vllm_pm2_start.cjs
python -m vllm.entrypoints.openai.api_server \
  --model /data1/meituan-longcat/LongCat-Next \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --pipeline-parallel-size 1 \
  --dtype bfloat16 \
  --trust-remote-code \
  --port 52099 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --mm-encoder-tp-mode data \
  --disable-custom-all-reduce \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --enforce-eager
```

### Memory Breakdown (Per GPU)

- **Model Weights**: ~18GB (after EP filter)
- **KV Cache**: ~20GB (with 0.90 utilization)
- **Activations**: ~4GB (with max_num_seqs=1)
- **Total**: ~42GB out of 46GB

### Performance Notes

- EP weight filter reduces loading time by ~87.5%
- Each GPU processes 1/8 of attention/embedding (TP=8)
- Each GPU processes 32/256 experts (EP=8)
- Multimodal encoders use data parallel mode

### Troubleshooting Commands

```bash
# Check GPU memory
nvidia-smi

# Check for DMA faults
dmesg -T | grep -i "dmar\|dma"

# Check for OOM killer
dmesg -T | grep -i "oom\|out of memory"

# Monitor vLLM logs
pm2 logs vllm

# Check process status
pm2 status
```

### Configuration Files

- **Model Config**: `/data1/meituan-longcat/LongCat-Next/config.json`
    - `num_hidden_layers`: 14
    - `num_experts`: 256
    - `encoder_layers`: 32 (audio encoder)
  
- **Startup Script**: `/root/learning/vllm/vllm_pm2_start.cjs`
- **Model Code**: `/root/learning/vllm/vllm/model_executor/models/longcat_next.py`

### Version Information

- **vLLM**: Latest main branch
- **Python**: 3.12
- **CUDA**: 12.x
- **PyTorch**: 2.x with CUDA support

### References

- vLLM Documentation: <https://docs.vllm.ai/>
- Expert Parallel PR: vllm-project/vllm#XXXXX
- MoE Architecture: LongCat-Next technical report
