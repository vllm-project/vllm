# compile test folder structure

- `compile/test_*.py` : various unit tests meant for testing particular code path/features. Future tests are most likely added here. New test files added here will be included in CI automatically
- `compile/fullgraph/` : full model tests, including all tests previously in compile/piecewise. These tests do not target particular features. New test files added here will be included in CI automatically
- `compile/distributed/` : tests that require multiple GPUs. New test files added here will **NOT** be included in CI automatically as these tests generally need to be manually configured to run in runners with particular number/type of GPUs.
