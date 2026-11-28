# compile test folder structure

- `compile/test_*.py` : various unit tests meant for testing particular code path/features. Future tests are most likely added here. New test files added here will be included in CI automatically
- `compile/fullgraph/` : full model tests, including all tests previously in compile/piecewise. These tests do not target particular features. New test files added here will be included in CI automatically
- `compile/passes/` and `compile/correctness_e2e/` : pass-specific and end-to-end correctness tests (including multi-GPU tests).
