# Scenario 03: Evaluation Rubric - KV Cache Management

## Scoring (Total: 100 points)

| Category | Points |
|----------|--------|
| Memory Management Design | 30 |
| Prefix Caching & Sharing | 25 |
| Eviction Policy | 20 |
| Implementation Details | 15 |
| Performance Analysis | 10 |

## Detailed Criteria

### Memory Management (30 pts)

**Outstanding (25-30):**
- Proposes paged memory with fixed-size blocks
- Accurate memory calculations including layers
- Explains fragmentation reduction
- Discusses block size trade-offs

**Example:**
```
Memory per block = num_layers × block_size × num_heads × head_dim × 2 bytes
For 80 layers, 16 tokens, 64 heads, 128 dim:
= 80 × 16 × 64 × 128 × 2 = 20.97 MB
```

**Weak (0-15):**
- Only considers contiguous allocation
- No memory calculations
- Doesn't understand fragmentation

### Prefix Caching (25 pts)

**Outstanding (20-25):**
- Hash-based prefix detection
- Reference counting for sharing
- Copy-on-write implementation
- Eviction of cached prefixes

**Strong (15-19):**
- Basic prefix caching idea
- Understands sharing benefits
- Some implementation details

### Eviction Policy (20 pts)

**Outstanding (17-20):**
- Multiple policies considered (LRU, priority, hybrid)
- Detailed scoring function
- Trade-off analysis
- Fairness considerations

**Acceptable (10-15):**
- Basic LRU or similar
- Simple implementation
- Limited trade-off discussion

## Level Expectations

**L6 (85-100):** Complete paged design, prefix caching, sophisticated eviction, accurate calculations
**L5 (70-84):** Solid paged design, basic prefix caching, reasonable eviction
**L4 (55-69):** Understands problem, basic block allocation, simple eviction

## Key Discriminators

1. **Paged vs Contiguous:** Must propose paged memory (critical)
2. **Memory Calculation:** Should calculate block memory accurately
3. **Copy-on-Write:** Understanding of safe sharing
4. **Reference Counting:** Essential for correctness
