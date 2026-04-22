# Pytest Marks Implementation Summary

## Progress Overview

This document tracks the implementation of pytest marks for intelligent CI test selection in vLLM.

## Completed Steps

### 1. Mark Definitions (pyproject.toml)
- Added 40+ pytest marks covering all functional areas
- Organized into categories:
  - Core functional areas (attention, kernels, quantization, etc.)
  - Distributed & parallelism (tensor_parallel, pipeline_parallel, etc.)
  - Model types (models_language, models_multimodal, etc.)
  - API & serving (entrypoints, openai_api, etc.)
  - Advanced features (spec_decode, kv_cache, tool_calling, etc.)
  - V1 engine (v1, v1_core, v1_distributed)
  - Platform-specific (cuda, rocm, tpu)
  - Test categories (correctness, benchmark, e2e)

### 2. Documentation
- Created comprehensive marking guide: `.buildkite/PYTEST_MARKING_GUIDE.md`
- Includes:
  - Quick start examples
  - Full mark reference table
  - Marking guidelines and best practices
  - Directory-to-mark mapping suggestions
  - Local development usage examples
  - Migration strategy

### 3. Automation Tools
- Created marking automation script: `.buildkite/scripts/mark_tests.py`
- Features:
  - Automatically suggests marks based on directory location
  - Analyzes file content for additional marks
  - Supports dry-run mode to preview changes
  - Can process individual files or entire directories
  - Skips files that already have marks

## Next Steps

### Phase 1: Mark Priority Test Areas

Priority areas (by test count and CI impact):
1. V1 tests (140 files) - Start here
2. Kernels tests (105 files) - High impact
3. Entrypoints tests (105 files) - API surface
4. Model tests (101 files) - Correctness critical
5. Distributed tests (32 files) - Complex dependencies

### Phase 2: Update CI Pipeline Generator

Modify the pipeline generator to use pytest marks for test selection:

Location: `https://github.com/vllm-project/ci-infra/buildkite/pipeline_generator`

Changes needed:
1. Add mark-based test selection to `buildkite_step.py`
2. Update `global_config.py` to support mark mapping
3. Modify test area configs to specify marks instead of file paths

### Phase 3: Gradual Rollout

1. Mark all tests in priority areas
2. Update a single test area config to use marks
3. Monitor CI for one week
4. Expand to all test areas
5. Deprecate file-based dependencies

### Phase 4: Validation & Monitoring

1. Create mark coverage validation script
2. Add pre-commit hook to check new tests are marked
3. Generate mark coverage reports
4. Monitor CI time savings

## Test Directory Analysis

Total: 676 test files

Top directories by test count:
- tests/v1/ - 140 files
- tests/entrypoints/ - 105 files (67 OpenAI API)
- tests/kernels/ - 105 files (30 quant, 29 MoE, 23 attn)
- tests/models/ - 101 files (44 multimodal, 37 language)
- tests/distributed/ - 32 files
- tests/lora/ - 26 files
- tests/compile/ - 25 files
- tests/quantization/ - 19 files

## Directory to Mark Mapping

### V1 Tests
- `tests/v1/attention/` → `v1`, `attention`
- `tests/v1/distributed/` → `v1`, `v1_distributed`, `distributed_comm`
- `tests/v1/core/` → `v1`, `v1_core`
- `tests/v1/engine/` → `v1`, `engine`
- `tests/v1/spec_decode/` → `v1`, `spec_decode`

### Kernels Tests
- `tests/kernels/attention/` → `kernels`, `attention`
- `tests/kernels/moe/` → `kernels`, `models_moe`, `expert_parallel`
- `tests/kernels/quantization/` → `kernels`, `quantization`

### Model Tests
- `tests/models/language/` → `models_language`, `correctness`
- `tests/models/multimodal/` → `models_multimodal`, `multimodal`, `correctness`
- `tests/models/quantization/` → `models_language`, `quantization`, `correctness`

### Entrypoints Tests
- `tests/entrypoints/openai/` → `entrypoints`, `openai_api`
- `tests/entrypoints/pooling/` → `entrypoints`, `pooling`
- `tests/entrypoints/llm/` → `entrypoints`, `offline_inference`

### Distributed Tests
- `tests/distributed/` → `distributed_comm`
- Tests with TP → add `tensor_parallel`
- Tests with PP → add `pipeline_parallel`
- Tests with EP → add `expert_parallel`

## Usage Examples

### Running Tests Locally

```bash
# Run all attention tests
pytest -m attention

# Run V1 attention tests only
pytest -m "v1 and attention"

# Run kernel tests excluding slow ones
pytest -m "kernels and not slow_test"

# Run distributed tests that require 2 GPUs
pytest -m "distributed_comm and distributed"

# Run all quantization tests
pytest -m quantization
```

### Automated Marking

```bash
# Dry run - preview what would be marked
python .buildkite/scripts/mark_tests.py tests/v1/attention/

# Actually apply marks
python .buildkite/scripts/mark_tests.py tests/v1/attention/ --apply

# Mark entire kernels directory
python .buildkite/scripts/mark_tests.py tests/kernels/ --apply
```

## CI Integration Plan

### Current State
- Test selection based on `source_file_dependencies` in YAML configs
- Simple substring matching: `if "vllm/attention/" in changed_files → run attention tests`
- Located in test area configs (`.buildkite/test_areas/*.yaml`)

### Target State
- Test selection based on pytest marks
- Intelligent mapping: `if "vllm/attention/" changed → pytest -m "attention or kernels"`
- Configurable mark mappings in `ci_config.yaml`

### Migration Path

1. Add mark mapping to `ci_config.yaml`:
```yaml
file_to_mark_mapping:
  vllm/attention/: ["attention", "kernels"]
  vllm/distributed/: ["distributed_comm", "tensor_parallel", "pipeline_parallel"]
  vllm/kernels/: ["kernels", "attention", "quantization"]
  # ... etc
```

2. Update pipeline generator logic:
```python
# In buildkite_step.py
def get_marks_for_changes(changed_files, mapping):
    marks = set()
    for changed_file in changed_files:
        for pattern, file_marks in mapping.items():
            if pattern in changed_file:
                marks.update(file_marks)
    return marks

# Then run: pytest -m "mark1 or mark2 or mark3"
```

3. Gradual rollout:
   - Week 1: Run both old and new selection in parallel, compare
   - Week 2-3: Use new selection for non-critical test areas
   - Week 4+: Full rollout, deprecate file-based dependencies

## Benefits

### For Developers
- Clear test categorization
- Easy to run relevant tests locally
- Faster local development iteration

### For CI
- More precise test selection
- Reduced over-triggering
- Better handling of cross-cutting changes
- Clearer test organization

### For Maintainability
- Self-documenting test organization
- Standard pytest feature
- No custom infrastructure
- Easy to extend

## Files Modified

1. `pyproject.toml` - Added mark definitions
2. `.buildkite/PYTEST_MARKING_GUIDE.md` - Created
3. `.buildkite/scripts/mark_tests.py` - Created
4. `.buildkite/PYTEST_MARKS_IMPLEMENTATION_SUMMARY.md` - This file

## Files to Create

1. `.buildkite/scripts/validate_test_marks.py` - Mark coverage validation
2. Pre-commit hook for marking enforcement
3. CI configuration updates in `ci-infra` repo

## Timeline Estimate

- Week 1: Mark priority test areas (V1, kernels, entrypoints)
- Week 2: Mark remaining test areas
- Week 3: Update pipeline generator, test in parallel mode
- Week 4: Full rollout and monitoring

## Success Metrics

1. Mark coverage: Target 95%+ of test files marked
2. CI time reduction: Target 15-30% reduction in test time
3. False negatives: Target <1% (tests that should run but don't)
4. Developer adoption: Marks used in local development

## Open Questions

1. Should we enforce marking via pre-commit hook?
2. How to handle tests that don't fit existing marks?
3. Should we generate mark suggestions in code review?
4. How often to audit and update mark mappings?

## Resources

- Pytest marks documentation: https://docs.pytest.org/en/stable/mark.html
- vLLM marking guide: `.buildkite/PYTEST_MARKING_GUIDE.md`
- Marking script: `.buildkite/scripts/mark_tests.py`
