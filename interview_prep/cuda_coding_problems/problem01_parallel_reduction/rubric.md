# Grading Rubric: Parallel Reduction

**Total Points: 100**

## 1. Correctness (40 points)

### Basic Functionality (20 points)
- [ ] **15 pts:** Produces correct sum for all test cases
  - Small arrays (< 100 elements)
  - Large arrays (> 1M elements)
  - Edge cases (empty, single element)
- [ ] **5 pts:** Handles non-power-of-2 array sizes correctly

### Numerical Stability (10 points)
- [ ] **5 pts:** Acceptable floating-point error accumulation
- [ ] **5 pts:** Handles negative numbers and mixed values

### Error Handling (10 points)
- [ ] **5 pts:** Proper CUDA error checking
- [ ] **5 pts:** Handles edge cases gracefully (null pointers, n=0)

## 2. Implementation Quality (30 points)

### Shared Memory Usage (15 points)
- [ ] **10 pts:** Uses shared memory for reduction
- [ ] **5 pts:** Correct synchronization with `__syncthreads()`

### Memory Access Patterns (10 points)
- [ ] **5 pts:** Sequential addressing to avoid bank conflicts
- [ ] **5 pts:** Coalesced global memory access

### Algorithm Efficiency (5 points)
- [ ] **3 pts:** Proper reduction tree structure
- [ ] **2 pts:** Minimizes warp divergence

## 3. Optimizations (20 points)

### Advanced Techniques (15 points)
- [ ] **5 pts:** Multiple elements per thread
- [ ] **5 pts:** Warp-level primitives (`__shfl_down_sync`)
- [ ] **5 pts:** Unrolled last warp

### Performance (5 points)
- [ ] **3 pts:** Within 2x of optimal bandwidth
- [ ] **2 pts:** Appropriate block size selection

## 4. Code Quality (10 points)

### Style and Documentation (5 points)
- [ ] **3 pts:** Clear comments explaining algorithm
- [ ] **2 pts:** Good variable names and code structure

### Completeness (5 points)
- [ ] **3 pts:** Complete solution with host wrapper
- [ ] **2 pts:** Proper memory management (no leaks)

---

## Scoring Breakdown

### Excellent (90-100 points)
- Fully correct implementation
- Uses shared memory effectively
- Implements warp-level optimizations
- Excellent performance (near-optimal)
- Clean, well-documented code

### Good (75-89 points)
- Correct for all test cases
- Uses shared memory
- Some optimizations applied
- Good performance
- Readable code with comments

### Satisfactory (60-74 points)
- Mostly correct (minor bugs)
- Uses shared memory
- Basic optimization
- Acceptable performance
- Adequate documentation

### Needs Improvement (< 60 points)
- Incorrect results or major bugs
- Missing shared memory usage
- Poor performance
- Minimal or no optimization
- Poor code quality

---

## Common Deductions

- **-10 pts:** Race conditions or synchronization bugs
- **-10 pts:** No shared memory usage
- **-5 pts:** Bank conflicts not addressed
- **-5 pts:** No error checking
- **-5 pts:** Memory leaks
- **-5 pts:** Incorrect handling of non-power-of-2 sizes
- **-3 pts:** No comments or documentation
- **-3 pts:** Hard-coded block sizes with no configuration

---

## Bonus Points (up to +10)

- [ ] **+5 pts:** Implements template version for multiple data types
- [ ] **+5 pts:** Compares performance with cuBLAS/CUB and discusses
- [ ] **+3 pts:** Implements multiple reduction strategies and benchmarks them

---

## Interview Performance Indicators

### Red Flags 🚩
- Doesn't understand need for `__syncthreads()`
- Creates race conditions
- Naive implementation without shared memory
- Can't explain why optimizations matter
- No consideration for different array sizes

### Green Flags ✅
- Immediately thinks about shared memory
- Discusses bank conflicts and coalescing
- Mentions warp-level primitives
- Considers numerical stability
- Tests with various sizes and edge cases
- Discusses trade-offs between different approaches

### Excellent Candidates 🌟
- Implements warp shuffle without prompting
- Explains multiple reduction strategies
- Discusses performance implications of block size
- Compares to library implementations
- Handles edge cases proactively
- Clean, production-quality code
