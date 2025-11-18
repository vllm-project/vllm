# Grading Rubric: Attention Scores

**Total: 100 points**

## Correctness (50 pts)
- [ ] **30 pts:** Correct Q*K^T computation
- [ ] **20 pts:** Proper scaling by 1/sqrt(d_k)

## Implementation (35 pts)
- [ ] **25 pts:** Tiled matrix multiplication with shared memory
- [ ] **10 pts:** Handles arbitrary dimensions

## Performance (10 pts)
- [ ] **10 pts:** Coalesced access, efficient tiling

## Code Quality (5 pts)
- [ ] **5 pts:** Clean code

## Bonus (+10)
- [ ] **+10 pts:** Fused with softmax or masking
