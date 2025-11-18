# Grading Rubric: LayerNorm Kernel

**Total: 100 points**

## Correctness (45 pts)
- [ ] **25 pts:** Correct normalization (mean≈0, std≈1)
- [ ] **10 pts:** Correct gamma/beta application
- [ ] **10 pts:** Numerically stable

## Implementation (40 pts)
- [ ] **20 pts:** Efficient mean/variance computation
- [ ] **15 pts:** Fused operations (2-pass or online)
- [ ] **5 pts:** Proper synchronization

## Performance (10 pts)
- [ ] **10 pts:** Efficient memory access

## Code Quality (5 pts)
- [ ] **5 pts:** Clean code

## Bonus (+10)
- [ ] **+10 pts:** Single-pass Welford's algorithm
