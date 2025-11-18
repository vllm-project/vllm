# Grading Rubric: Quantized INT8 GEMM

**Total: 100 points**

## Correctness (40 pts)
- [ ] **20 pts:** Correct quantization
- [ ] **20 pts:** Correct GEMM result after dequantization

## INT8 Computation (35 pts)
- [ ] **20 pts:** Uses INT8 arithmetic
- [ ] **15 pts:** INT32 accumulation to avoid overflow

## Optimization (15 pts)
- [ ] **10 pts:** Tiled implementation
- [ ] **5 pts:** Proper scaling

## Advanced (10 pts)
- [ ] **10 pts:** Uses dp4a or tensor cores

## Bonus (+10)
- [ ] **+10 pts:** Per-channel quantization or asymmetric quantization
