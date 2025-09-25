# Pick & Ban Routing for MoE Models

This directory contains examples and utilities for implementing Pick & Ban routing algorithms in vLLM MoE models.

## Files

- `patch_qwen2_moe.py` - Runtime patching system for Qwen2MoE models
- `test_pick_ban_with_import_patch.py` - Integration test for Pick & Ban routing

## Usage

### Prerequisites

1. Install vLLM with MoE support
2. Ensure you have a Qwen2MoE model available
3. Sufficient GPU memory (recommended: 2x 24GB GPUs)

### Running the Test

```bash
# From the vLLM root directory
cd examples/others/moe_examples
python test_pick_ban_with_import_patch.py
```

### Expected Output

You should see debug messages indicating that the Pick & Ban routing is working:

```bash
🔧 Importing patch module...
✅ Patch module imported!
🚀🚀🚀 PATCHED Qwen2MoeSparseMoeBlock.__init__ called!
🚀🚀🚀 PICK & BAN ROUTING FUNCTION CALLED!
🔑 Using key experts for layer 0: {8, 23, 15}
✅✅✅ PICK & BAN ROUTING COMPLETED! ✅✅✅
```

## Configuration

### Key Expert Configuration

Edit `patch_qwen2_moe.py` to modify the key experts:

```python
key_experts_per_layer = {
    i: {8, 15, 23} for i in range(24)  # Modify these expert IDs
}
```

### Threshold Parameters

```python
custom_routing_function = create_pick_and_ban_routing_function(
    key_experts_per_layer=key_experts_per_layer,
    lambda_threshold=0.7,  # Adjust for performance/efficiency trade-off
    tau_threshold=0.9      # Adjust for pruning aggressiveness
)
```

## Memory Optimization

The test script includes several memory optimization settings:

- `gpu_memory_utilization=0.8` - GPU memory usage
- `max_model_len=64` - Sequence length limit
- `kv_cache_dtype="fp8"` - FP8 KV cache compression
- `cpu_offload_gb=4` - CPU weight offloading
- `swap_space=4` - System swap space usage

Adjust these parameters based on your hardware configuration.

## Troubleshooting

### Memory Issues

- Reduce `gpu_memory_utilization`
- Decrease `max_model_len`
- Enable more aggressive CPU offloading

### Import Issues

- Ensure you're running from the correct directory
- Check that vLLM is properly installed
- Verify the model path is correct

### Routing Not Applied

- Check that patch is imported before vLLM
- Look for debug messages in the output
- Verify the model is a Qwen2MoE model

## Technical Details

The implementation works by:

1. **Runtime Patching**: Replaces the original `Qwen2MoeSparseMoeBlock` with a patched version
2. **Custom Routing**: Uses Pick & Ban algorithms instead of standard top-k routing
3. **Memory Optimization**: Avoids double allocation by carefully managing component creation
4. **Multi-Process Support**: Ensures patches apply to all worker processes

## References

- [Pick & Ban: Selective Knowledge Distillation for Mixture of Experts](https://arxiv.org/pdf/2509.06346)
- [vLLM Documentation](https://docs.vllm.ai/)
