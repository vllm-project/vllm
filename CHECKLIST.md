# Checklist: What to Check In

## ✅ Files to COMMIT (Production Code & Tests)

### Main Test Suite
- [ ] `tests/models/test_generic_models.py` - Comprehensive pytest test suite
  - Flash attention with backward pass
  - Training-compatible model implementation
  - vLLM integration wrapper
  - 5 core tests + 2 skipped integration tests

### Examples
- [ ] `examples/generic_model_parallelism.py` - Parallelism demonstration
  - Tensor parallel linear layers
  - User model builder pattern
  - 3 runnable examples

### Documentation
- [ ] `tests/models/README_GENERIC_MODELS.md` - Test documentation
- [ ] `GENERIC_MODEL_TEST_README.md` - Comprehensive guide
- [ ] `IMPLEMENTATION_SUMMARY.md` - Implementation overview
- [ ] `RFC_IMPLEMENTATION_STATUS.md` - Current status and validation
- [ ] `RFC.md` - Original RFC proposal (if not already committed)

### Template
- [ ] `run_generic_tests.sh.template` - Template for users to customize

## ❌ Files to KEEP LOCAL (Device-Specific)

### Test Runners (contain hardcoded paths)
- `run_all_tests.sh` - Meta-specific LD_PRELOAD paths
- `run_generic_model_tests.py` - Local runner
- `run_generic_tests.sh` - User's customized version (from template)

### Integration Tests (local testing)
- `test_vllm_integration.py` - Local integration tests
- `test_vllm_advanced.py` - Local advanced tests

### Old Files (replaced by proper test structure)
- `tests/test_generic_model_support.py` - Replaced by `tests/models/test_generic_models.py`

## .gitignore Additions (Recommended)

Add these patterns to `.gitignore`:

```gitignore
# Generic model test runners (device-specific)
run_generic_tests.sh
run_all_tests.sh
test_vllm_integration.py
test_vllm_advanced.py
run_generic_model_tests.py
```

## Quick Test Before Committing

```bash
# Standalone test (should work without any setup)
python tests/models/test_generic_models.py

# Expected output:
# Testing Flash Attention Forward...
# ✓ Flash Attention Forward works
# ...
# All basic tests passed!
```

## Summary

**Production files to commit:** 7 files
- 1 test suite (pytest compatible)
- 1 example file
- 4 documentation files
- 1 template

**Local files to keep:** 5+ files
- Device-specific test runners
- Integration test scripts
- Deprecated/old test files

All production files are clean, well-documented, and pytest-compatible!
