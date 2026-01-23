# vLLM Repository Guide for Claude

## Overview
vLLM is a fast and easy-to-use library for LLM inference and serving, originally developed at UC Berkeley's Sky Computing Lab. It features PagedAttention for efficient KV cache management, continuous batching, and support for various hardware accelerators.

**Documentation**: https://docs.vllm.ai
**Contributing Guide**: https://docs.vllm.ai/en/latest/contributing
**Repository**: https://github.com/vllm-project/vllm

## Repository Structure

### Core Directories

```
vllm/
├── v1/                          # V1 API (new architecture)
│   ├── core/                    # Core scheduling and KV cache management
│   │   ├── sched/              # Scheduler implementation
│   │   │   ├── scheduler.py    # Main scheduler class
│   │   │   ├── async_scheduler.py  # Async scheduling variant
│   │   │   └── output.py       # SchedulerOutput dataclass
│   │   └── kv_cache_manager.py # KV cache management
│   ├── engine/                  # Engine coordination
│   ├── worker/                  # Worker processes (GPU/CPU)
│   ├── attention/               # Attention implementations
│   ├── executor/                # Distributed execution
│   └── sample/                  # Sampling logic
├── engine/                      # Legacy engine (v0 API)
├── attention/                   # Attention backends
├── model_executor/              # Model execution layer
├── distributed/                 # Distributed inference (KV transfer, etc.)
├── config/                      # Configuration classes
├── entrypoints/                 # API servers (OpenAI-compatible, etc.)
└── utils/                       # Utility functions

tests/
├── v1/                          # V1 API tests
│   ├── core/                   # Scheduler, KV cache tests
│   ├── worker/                 # Worker tests
│   └── attention/              # Attention tests
├── distributed/                 # Distributed system tests
└── models/                      # Model-specific tests

requirements/                    # Dependency specifications
├── common.txt                  # Common dependencies
├── dev.txt                     # Development dependencies
├── test.txt                    # Test dependencies
├── cuda.txt                    # CUDA-specific dependencies
└── cpu.txt                     # CPU-specific dependencies
```

### Key Files

- **pyproject.toml**: Project metadata, dependencies, and tool configurations
- **setup.py**: Installation script
- **.pre-commit-config.yaml**: Pre-commit hooks (formatting, linting)
- **CONTRIBUTING.md**: Contribution guidelines (points to docs)

## Development Setup

### Virtual Environment
The repo uses a virtual environment in `.venv/`:
```bash
# Already activated if you see (.venv) in prompt
source .venv/bin/activate  # To activate manually

# Install dependencies
pip install -r requirements/common.txt
pip install -r requirements/dev.txt

# Install vLLM in editable mode
pip install -e .
```

### Running Tests

```bash
# Run specific test file
pytest tests/v1/core/test_scheduler.py -v

# Run specific test function
pytest tests/v1/core/test_scheduler.py::test_scheduler_step_counter -v

# Run all tests in a directory
pytest tests/v1/core/ -v

# Run with short traceback
pytest tests/v1/core/test_scheduler.py --tb=short

# Common markers
pytest -m cpu_test          # CPU-only tests
pytest -m slow_test         # Slow tests (usually skipped)
pytest -m core_model        # Core model tests
```

### Test File Locations
- **Scheduler tests**: `tests/v1/core/test_scheduler.py`
- **Async scheduler tests**: `tests/v1/core/test_async_scheduler.py`
- **KV cache tests**: `tests/v1/core/test_prefix_caching.py`, `tests/v1/core/test_kv_cache_utils.py`
- **Worker tests**: `tests/v1/worker/test_gpu_model_runner.py`
- **Integration tests**: `tests/v1/core/test_scheduler_e2e.py`

## Code Conventions

### Commit Messages
Follow the pattern used in the repository:
```
[Category] Brief description (#PR_NUMBER)

Examples:
[Feature] Add monotonically increasing step counter to scheduler
[Bugfix] Fix getting vision features in Transformer Multimodal backend (#32933)
[Misc] Postpone torch_profiler deprecation (#32867)
[CPU][Feat] Update PyTorch to v2.10 for CPU Backend (#32869)
```

**Categories**: `Feature`, `Bugfix`, `Misc`, `Refactor`, `Docs`, `CI/Build`, `Frontend`, `Backend`

### Coding Style
- **Type hints**: Use throughout (Python 3.12+ syntax with `|` for unions)
- **Docstrings**: Google style for classes and public methods
- **Line length**: ~88 characters (Black formatter)
- **Imports**: Organized by standard library, third-party, local
- **Comments**: Explain "why" not "what" for complex logic

### Dataclass Conventions
When adding fields to dataclasses:
- Fields with defaults must come after fields without defaults
- Use `= None` for optional fields
- Use `= 0`, `= False`, etc. for backward compatibility
- Document all fields with inline comments

Example:
```python
@dataclass
class SchedulerOutput:
    # Required fields first
    scheduled_new_reqs: list[NewRequestData]
    num_scheduled_tokens: dict[str, int]

    # Optional fields with defaults last
    preempted_req_ids: set[str] | None = None
    scheduler_step: int = 0  # Default for backward compatibility
```

## V1 Architecture (Current Focus)

### Scheduler (`vllm/v1/core/sched/`)
The scheduler is responsible for:
- Managing request queues (waiting, running)
- Allocating KV cache blocks
- Producing `SchedulerOutput` for workers
- Handling preemption and prefix caching

**Key Classes**:
- `Scheduler`: Main synchronous scheduler
- `AsyncScheduler`: Async variant with output placeholders
- `SchedulerOutput`: Output dataclass sent to workers
- `Request`: Request state tracking

**Important Methods**:
- `schedule()`: Main scheduling loop (increments scheduler_step_counter)
- `add_request()`: Add new request to waiting queue
- `finish_requests()`: Mark requests as completed
- `reset_prefix_cache()`: Reset prefix cache (doesn't reset step counter)

### Scheduler Step Counter
- **Location**: `scheduler.scheduler_step_counter` (instance variable)
- **Initialization**: `0` (first call produces `1`)
- **Increment**: At start of `schedule()` method
- **Output**: Passed to `SchedulerOutput.scheduler_step`
- **Purpose**: Track scheduler invocations for tracing and correlation
- **Properties**: Monotonic, never reset, increments even on empty schedules

### KV Cache Management
- **Manager**: `KVCacheManager` handles block allocation and caching
- **Prefix caching**: Enabled via `enable_prefix_caching` config
- **KV Connector**: Handles distributed KV cache transfer
- **Events**: Optional KV cache event publishing for tracing

## Common Testing Patterns

### Creating Test Fixtures
```python
from tests.v1.core.utils import create_scheduler, create_requests

# Create scheduler with default config
scheduler = create_scheduler()

# Create scheduler with options
scheduler = create_scheduler(
    enable_prefix_caching=True,
    max_num_batched_tokens=2048,
    model="llava-hf/llava-1.5-7b-hf"
)

# Create test requests
requests = create_requests(num_requests=10)
```

### Testing Scheduler Behavior
```python
# Add requests
for request in requests:
    scheduler.add_request(request)

# Run scheduling
output = scheduler.schedule()

# Assertions
assert len(output.scheduled_new_reqs) == expected_count
assert output.scheduler_step == expected_step
assert output.total_num_scheduled_tokens == expected_tokens
```

### Testing AsyncScheduler
AsyncScheduler has additional complexity with output placeholders:
```python
from tests.v1.core.utils import create_async_scheduler

scheduler = create_async_scheduler()
# ... similar patterns as Scheduler
```

## Important Design Patterns

### Immutable Output Objects
`SchedulerOutput` and similar dataclasses are created once and not modified. This ensures thread safety and clear ownership.

### Backward Compatibility
When adding new fields:
1. Add default values for optional fields
2. Place new fields at end of dataclass
3. Update `make_empty()` if needed (may not be required with defaults)
4. Test that existing code works without specifying new fields

### Monotonic Counters
When adding counters or IDs:
- Initialize to 0, increment at start (first value is 1)
- Never reset unless explicitly required
- Document lifetime and scope clearly
- Consider thread safety (scheduler is single-threaded)

## File References (Line Numbers)

### Key Scheduler Files
- `vllm/v1/core/sched/scheduler.py`
  - Line ~100-110: Instance variable initialization
  - Line ~258: `schedule()` method starts
  - Line ~770-800: `SchedulerOutput` construction
  - Line ~1536: `reset_prefix_cache()` method

- `vllm/v1/core/sched/output.py`
  - Line ~183-240: `SchedulerOutput` dataclass definition
  - Line ~242: `make_empty()` classmethod

- `vllm/v1/core/sched/async_scheduler.py`
  - Line ~12: Class definition (inherits from `Scheduler`)
  - Does not override `schedule()` - inherits from parent

### Test Files
- `tests/v1/core/test_scheduler.py`
  - Line ~43-63: Basic scheduler tests
  - Line ~76-123: Scheduler step counter test
  - Line ~130-200: Multimodal and partial request tests

## Common Tasks

### Adding a Field to SchedulerOutput
1. Add field at end of dataclass with default value
2. Update scheduler to pass value when constructing SchedulerOutput
3. Add test verifying the field is set correctly
4. Verify backward compatibility (existing code still works)

### Adding a Scheduler Feature
1. Add instance variables in `Scheduler.__init__()`
2. Implement logic in relevant methods (`schedule()`, etc.)
3. Update `SchedulerOutput` if needed
4. Add comprehensive unit tests
5. Check if `AsyncScheduler` needs updates (usually inherits correctly)
6. Run full test suite to ensure no regressions

### Debugging Scheduler Issues
1. Check `tests/v1/core/test_scheduler.py` for similar test patterns
2. Use `create_scheduler()` and `create_requests()` utilities
3. Add assertions on `SchedulerOutput` fields
4. Check if issue exists in both `Scheduler` and `AsyncScheduler`
5. Look at `SchedulerOutput` to see what data is available

## Useful Commands

### Git Operations
```bash
# Check status
git status

# Create feature branch
git checkout -b feature-name

# Stage changes
git add file1 file2

# Commit with proper format
git commit -m "[Category] Description"

# Push and create PR
git push -u origin feature-name
gh pr create --title "[Category] Title" --body "Description"
```

### Testing Shortcuts
```bash
# Quick syntax check
python -m py_compile file.py

# Run tests matching pattern
pytest -k "test_scheduler" -v

# Run tests in parallel (if pytest-xdist installed)
pytest -n auto tests/v1/core/

# Show print statements
pytest tests/v1/core/test_scheduler.py -v -s
```

### Code Quality
```bash
# Format code (if Black is configured)
black vllm/ tests/

# Run type checker (if mypy is configured)
mypy vllm/

# Run linter
ruff check vllm/
```

## Tips for Claude

### Before Making Changes
1. Read existing code in the area you're modifying
2. Check test files for patterns and conventions
3. Look at recent commits for style guidance
4. Verify the change aligns with vLLM's architecture

### When Adding Features
1. Start with the data structures (`SchedulerOutput`, etc.)
2. Add instance variables and initialization
3. Implement core logic
4. Add to output/return values
5. Write comprehensive tests
6. Test backward compatibility

### When Writing Tests
1. Use utility functions from `tests/v1/core/utils.py`
2. Test both positive cases and edge cases
3. Verify behavior with empty inputs
4. Check that cleanup/reset operations work correctly
5. Ensure tests are deterministic and don't depend on timing

### When Debugging
1. Check if `AsyncScheduler` is involved (different behavior)
2. Look at `SchedulerOutput` to see what's being produced
3. Add temporary assertions to narrow down issues
4. Check if prefix caching or KV connector is affecting behavior
5. Review recent changes in related files

## Current State (As of This PR)

### Recent Changes
- Added `scheduler_step` counter to track scheduler invocations
- Counter is monotonically increasing, never resets
- Included in `SchedulerOutput` with backward-compatible default
- All 149 tests pass with no regressions

### Next Steps (Potential)
- Use `scheduler_step` for trace stream implementation
- Add KV cache transfer stream correlated by step
- Implement request tracing using step numbers
- Add performance profiling by scheduler step

## Resources

- **Main docs**: https://docs.vllm.ai
- **Architecture docs**: https://docs.vllm.ai/en/latest/design/
- **Contributing**: https://docs.vllm.ai/en/latest/contributing
- **API reference**: https://docs.vllm.ai/en/latest/api/
- **Blog**: https://blog.vllm.ai/
- **Paper**: https://arxiv.org/abs/2309.06180

## Notes

- The repository is in active development with frequent updates
- V1 API is the new architecture; V0 is legacy but still supported
- Tests are critical - always run full test suite before submitting PR
- Use descriptive variable names and add comments for complex logic
- When in doubt, check existing patterns in the codebase
