# Grading Rubric: Matrix Transpose

**Total Points: 100**

## Correctness (40 points)
- [ ] **20 pts:** Correct transpose for all test cases
- [ ] **10 pts:** Handles non-square matrices
- [ ] **10 pts:** Handles non-tile-aligned dimensions

## Optimization (35 points)
- [ ] **15 pts:** Uses shared memory with tiling
- [ ] **10 pts:** Coalesced memory access (reads and writes)
- [ ] **10 pts:** Padding to avoid bank conflicts (+1 in tile dimension)

## Implementation Quality (15 points)
- [ ] **8 pts:** Proper synchronization
- [ ] **7 pts:** Correct boundary handling

## Code Quality (10 points)
- [ ] **5 pts:** Clean, readable code with comments
- [ ] **5 pts:** Error checking and memory management

## Scoring
- **90-100:** Fully optimized with padding, coalescing, correct
- **75-89:** Uses shared memory, mostly correct, minor issues
- **60-74:** Basic shared memory, has bugs or missing optimizations
- **<60:** Naive implementation or major correctness issues
