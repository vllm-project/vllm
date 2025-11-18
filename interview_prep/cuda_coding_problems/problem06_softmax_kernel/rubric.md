# Grading Rubric: Softmax Kernel

**Total: 100 points**

## Correctness (45 pts)
- [ ] **25 pts:** Correct softmax computation
- [ ] **20 pts:** Numerically stable (subtracts max)

## Implementation (35 pts)
- [ ] **15 pts:** Three-pass approach (max, exp+sum, normalize)
- [ ] **10 pts:** Shared memory/warp reductions
- [ ] **10 pts:** Handles arbitrary dimensions

## Performance (15 pts)
- [ ] **15 pts:** Efficient memory access, minimized passes

## Code Quality (5 pts)
- [ ] **5 pts:** Clean code

## Bonus (+10)
- [ ] **+5 pts:** Single-pass online softmax
- [ ] **+5 pts:** Handles very large feature dimensions (multi-block per row)
