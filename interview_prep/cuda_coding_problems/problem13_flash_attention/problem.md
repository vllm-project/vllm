# Problem 13: Flash Attention (Simplified)

**Difficulty:** Hard
**Estimated Time:** 60-75 minutes
**Tags:** Advanced Algorithms, Tiling, Online Algorithms, Transformers

## Problem Statement

Implement a simplified version of Flash Attention that reduces memory reads/writes by computing attention in blocks using online softmax. This is a critical optimization for transformer models.

## Background

Standard attention:
```
S = Q @ K^T / sqrt(d)  # (seq_len × seq_len)
P = softmax(S)         # Memory bottleneck!
O = P @ V              # (seq_len × d)
```

Flash Attention avoids materializing full (seq_len × seq_len) attention matrix.

## Requirements

- Compute attention in tiles without materializing full attention matrix
- Use online softmax (incremental max and sum)
- Fuse operations to reduce memory traffic
- Handle sequence lengths >> shared memory capacity
- Correct numerical stability

## Algorithm Outline

1. Divide Q, K, V into blocks
2. For each Q block:
   - For each K, V block:
     - Compute partial attention scores
     - Update running max, sum, output (online softmax)
3. Final normalization

## Success Criteria

- ✅ Correct attention output
- ✅ Tiled/blocked computation
- ✅ Online softmax implementation
- ✅ Reduced memory footprint vs. standard
- ✅ Handles large sequences
- ✅ Numerically stable
