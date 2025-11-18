# Grading Rubric: Element-wise Operations

**Total: 100 points**

## Correctness (40 pts)
- [ ] **30 pts:** Correct computation for all inputs
- [ ] **10 pts:** Handles arbitrary array sizes

## Performance (30 pts)
- [ ] **20 pts:** Single fused kernel (not separate kernels)
- [ ] **10 pts:** >80% memory bandwidth utilization

## Implementation (20 pts)
- [ ] **10 pts:** Grid-stride loop
- [ ] **10 pts:** Coalesced memory access

## Code Quality (10 pts)
- [ ] **10 pts:** Clean code, error checking

## Bonus (+10 pts)
- [ ] **+10 pts:** Vectorized float4 version
