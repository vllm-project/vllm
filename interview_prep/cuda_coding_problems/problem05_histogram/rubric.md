# Grading Rubric: Histogram

**Total: 100 points**

## Correctness (50 pts)
- [ ] **40 pts:** Correct histogram counts
- [ ] **10 pts:** Handles edge cases (empty bins, large inputs)

## Atomic Operations (25 pts)
- [ ] **15 pts:** Correct use of atomicAdd
- [ ] **10 pts:** No race conditions

## Optimization (20 pts)
- [ ] **20 pts:** Shared memory privatization

## Code Quality (5 pts)
- [ ] **5 pts:** Clean code

## Bonus (+10)
- [ ] **+10 pts:** Warp-level aggregation before atomics
