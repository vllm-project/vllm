# vLLM Codebase Structure

## Root Directory Layout

```
vllm/
├── vllm/                    # Main Python package
├── csrc/                    # CUDA/C++ kernel source code
├── tests/                   # Test suite
├── docs/                    # Documentation source (MkDocs)
├── examples/                # Example scripts and usage
├── benchmarks/              # Benchmarking tools and scripts
├── tools/                   # Development and utility tools
├── requirements/            # Dependency specifications
├── cmake/                   # CMake build configuration
├── docker/                  # Docker configurations
├── .github/                 # GitHub Actions CI/CD
├── .buildkite/             # Buildkite CI configuration
└── pyproject.toml          # Project configuration
```

## Main Package Structure (`vllm/`)

### Core Components

- **`engine/`** - Core inference engine
  - LLMEngine and AsyncLLMEngine implementations
  - Request scheduling and execution logic

- **`v1/`** - V1 architecture (major upgrade)
  - 1.7x speedup with architectural improvements
  - Zero-overhead prefix caching
  - Enhanced multimodal support

- **`model_executor/`** - Model execution logic
  - Model loading and inference
  - Parallel execution strategies
  - Model-specific implementations

- **`attention/`** - Attention mechanisms
  - PagedAttention implementation
  - Various attention backends

- **`entrypoints/`** - Entry points for different interfaces
  - CLI (`cli/`)
  - OpenAI-compatible API server
  - Python API (`LLM` class)

### Supporting Components

- **`config/`** - Configuration management
  - Model configuration
  - Engine configuration
  - Parallel configuration

- **`distributed/`** - Distributed inference support
  - Tensor parallelism
  - Pipeline parallelism
  - Communication primitives

- **`multimodal/`** - Multi-modal model support
  - Vision-language models
  - Input processing for different modalities

- **`lora/`** - LoRA (Low-Rank Adaptation) support
  - Multi-LoRA inference
  - LoRA model loading and management

- **`transformers_utils/`** - Hugging Face transformers utilities
  - Model compatibility
  - Tokenizer utilities

### Utilities and Tools

- **`utils/`** - General utilities
- **`logging_utils/`** - Logging utilities
- **`triton_utils/`** - Triton kernel utilities
- **`compilation/`** - Torch compilation support
- **`profiler/`** - Performance profiling tools
- **`usage/`** - Usage tracking
- **`platforms/`** - Platform-specific code
- **`device_allocator/`** - Memory allocation

### Other Components

- **`inputs/`** - Input processing and validation
- **`reasoning/`** - Reasoning capabilities
- **`plugins/`** - Plugin system
- **`third_party/`** - Third-party dependencies (excluded from linting)
- **`vllm_flash_attn/`** - FlashAttention integration
- **`assets/`** - Static assets

### Top-level Files

- `__init__.py` - Package initialization (uses lazy imports)
- `logger.py` - Logging configuration
- `envs.py` - Environment variable handling
- `sampling_params.py` - Sampling parameter definitions
- `outputs.py` - Output data structures
- `sequence.py` - Sequence management
- `tasks.py` - Task definitions
- `version.py` - Version information

## CUDA/C++ Source (`csrc/`)

Contains custom CUDA kernels and C++ extensions:
- Quantization kernels
- Attention kernels
- Activation kernels
- Position encoding kernels
- Other optimized operations

## Tests Structure (`tests/`)

Organized by component and feature:

- **`v1/`** - V1 architecture tests
- **`models/`** - Model-specific tests
- **`kernels/`** - Kernel tests with torch.library.opcheck()
- **`basic_correctness/`** - Correctness tests
- **`distributed/`** - Distributed inference tests
- **`entrypoints/`** - API and entrypoint tests
- **`multimodal/`** - Multi-modal tests
- **`lora/`** - LoRA tests
- **`quantization/`** - Quantization tests
- **`samplers/`** - Sampling algorithm tests
- **`compile/`** - Compilation tests
- **`vllm_test_utils/`** - Test utilities

Test files follow naming convention: `test_<feature>.py`

## Documentation (`docs/`)

- **`contributing/`** - Contribution guidelines
  - Model addition guide
  - Kernel development guide
  - CI/Build documentation
- **`getting_started/`** - Installation and quickstart
- **`models/`** - Supported models documentation
- API reference (auto-generated from docstrings)

## Configuration Files

- **`.pre-commit-config.yaml`** - Pre-commit hook configuration
- **`pyproject.toml`** - Python project metadata, ruff/mypy/pytest config
- **`setup.py`** - Build configuration (setuptools)
- **`CMakeLists.txt`** - CMake build configuration
- **`mkdocs.yaml`** - Documentation site configuration

## Build Artifacts (Not in Git)

- `build/` - CMake build directory
- `.venv/` - Virtual environment
- `*.egg-info/` - Package metadata
- `.mypy_cache/`, `.ruff_cache/` - Tool caches
- `.serena/` - Serena tool data
