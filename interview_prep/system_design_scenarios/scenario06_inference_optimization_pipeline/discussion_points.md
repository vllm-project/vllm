# Scenario 06: Discussion Points - Inference Optimization

## Key Topics

### 1. Quantization Strategies
**Question:** "Compare INT8, INT4, and AWQ quantization."
- INT8: 2x speedup, minimal accuracy loss
- INT4: 4x speedup, <1% accuracy loss with AWQ
- AWQ: Activation-aware, better than naive INT4

### 2. Kernel Optimization
**Question:** "What kernel optimizations can you apply?"
- Flash Attention: 30% faster, 50% less memory
- Fused kernels: GEMM + bias + activation
- TensorRT: 2-3x speedup vs PyTorch

### 3. A/B Testing Optimizations
**Question:** "How do you safely roll out optimizations?"
- Baseline vs optimized comparison
- Gradual traffic shift
- Quality verification
- Automatic rollback

## Red/Green Flags
**Red:** Only mentions one optimization type, no testing strategy
**Green:** Multi-stage pipeline, accurate speedup estimates, comprehensive testing
