# Guidelines, Patterns, and Best Practices

## Design Patterns and Architectural Principles

### 1. Configuration Classes with msgspec.Struct
vLLM uses `msgspec.Struct` for configuration classes instead of dataclasses:

```python
class SamplingParams(
    msgspec.Struct,
    omit_defaults=True,
    dict=True,
):
    """Sampling parameters with validation."""
    temperature: float = 1.0
    top_p: float = 1.0
    
    def __post_init__(self) -> None:
        self._verify_args()
```

**Benefits:**
- Faster serialization/deserialization
- Better performance than dataclasses
- Support for validation in `__post_init__`

### 2. Lazy Imports
The root `__init__.py` uses lazy imports to improve startup time:

```python
# Don't import everything at package level
# Use lazy loading for heavy dependencies
```

### 3. Logger Usage
Always use the vLLM logger, not print statements:

```python
from vllm.logger import logger

logger.info("Starting inference")
logger.warning("This is a warning")
logger.debug("Debug information")
```

Special logger methods for one-time messages:
```python
logger.info_once("This will only print once")
logger.warning_once("One-time warning")
```

### 4. Environment Variables
Use `vllm.envs` for environment variable handling:

```python
import vllm.envs as envs

if envs.VLLM_CONFIGURE_LOGGING:
    # Configure logging
```

### 5. Type Safety
Use type hints extensively and ensure mypy compliance:

```python
from typing import Literal, cast, Annotated

def process_tokens(
    tokens: list[int],
    max_tokens: int | None = None,
) -> list[int]:
    ...
```

## Code Organization Patterns

### 1. Validation Pattern
Separate validation into dedicated methods:

```python
class SamplingParams:
    def __post_init__(self) -> None:
        self._verify_args()
        self._verify_greedy_sampling()
    
    def _verify_args(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}.")
```

### 2. Properties for Derived Values
Use `@property` or `@cached_property` for derived values:

```python
from functools import cached_property

class SamplingParams:
    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        return SamplingType.RANDOM
```

### 3. Factory Methods
Use static methods for alternative constructors:

```python
class SamplingParams:
    @staticmethod
    def from_optional(...) -> "SamplingParams":
        # Handle None values and create instance
        return SamplingParams(...)
```

## Import Organization

Imports should be organized in three sections:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Module docstring."""

# Standard library
import os
import sys
from typing import Any

# Third-party
import torch
import msgspec

# Local/vLLM
import vllm.envs as envs
from vllm.logger import logger
```

## Special Import Rules

### 1. Use regex instead of re
```python
import regex as re  # Enforced by pre-commit

# NOT: import re
```

### 2. Lazy Triton Imports
Don't import triton at module level:

```python
# Good: lazy import
def use_triton():
    import triton
    ...

# Bad: top-level import
import triton  # Pre-commit will catch this
```

### 3. Avoid pickle/cloudpickle
New uses of pickle/cloudpickle are discouraged and checked by pre-commit.

## Testing Patterns

### 1. Fixture Usage
Use pytest fixtures from `conftest.py`:

```python
def test_model_inference(vllm_runner):
    with vllm_runner("facebook/opt-125m") as llm:
        outputs = llm.generate(...)
```

### 2. Test Markers
Use appropriate markers:

```python
@pytest.mark.slow_test
def test_large_model():
    ...

@pytest.mark.core_model
def test_critical_model():
    ...

@pytest.mark.skip_v1
def test_v0_only():
    ...
```

### 3. Kernel Tests
For custom ops, use `torch.library.opcheck()`:

```python
from torch.library import opcheck

def test_custom_op():
    opcheck(torch.ops.vllm.my_op, ...)
```

## Documentation Patterns

### 1. Module Docstrings
Every module should have a docstring:

```python
"""Logging configuration for vLLM.

This module provides centralized logging setup and utilities for the vLLM
project.
"""
```

### 2. Class and Method Docstrings
Use descriptive docstrings with details:

```python
class SamplingParams:
    """Sampling parameters for text generation.
    
    Overall, we follow the sampling parameters from the OpenAI text completion
    API. In addition, we support beam search.
    """
    
    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        model_eos_token_id: int | None = None,
    ) -> None:
        """Update if there are non-default values from generation_config"""
```

### 3. Inline Attribute Docstrings
Use inline docstrings for class attributes:

```python
class SamplingParams:
    n: int = 1
    """Number of outputs to return for the given prompt request."""
    
    temperature: float = 1.0
    """Controls the randomness of the sampling."""
```

## Performance Considerations

### 1. Avoid Unnecessary Allocations
Reuse tensors when possible, avoid creating temporary objects in hot paths.

### 2. Use Appropriate Data Structures
- `msgspec.Struct` for serializable config objects
- `set` for fast membership testing
- `dict` for key-value lookups

### 3. Lazy Loading
Defer expensive imports and initialization:

```python
# Load heavy dependency only when needed
def use_heavy_library():
    import heavy_library
    return heavy_library.do_something()
```

## Error Handling

### 1. Specific Exceptions
Raise specific exceptions with helpful messages:

```python
if self.n < 1:
    raise ValueError(f"n must be at least 1, got {self.n}.")
```

### 2. Validation Early
Validate inputs early in `__post_init__` or at function entry.

### 3. Logging for Warnings
Use logger for non-fatal issues:

```python
if 0 < self.temperature < _MAX_TEMP:
    logger.warning(
        "temperature %s is less than %s, clamping to %s",
        self.temperature, _MAX_TEMP, _MAX_TEMP
    )
```

## Version Compatibility

### 1. Python Version Support
- Maintain compatibility with Python 3.10-3.13
- Use modern syntax where possible (PEP 604 union types)

### 2. Deprecation Warnings
Use proper deprecation warnings:

```python
import warnings

if guided_decoding is not None:
    warnings.warn(
        "guided_decoding is deprecated. Use structured_outputs instead.",
        DeprecationWarning,
        stacklevel=2,
    )
```

## Special Considerations

### 1. Platform-Specific Code
Handle different platforms appropriately:

```python
import sys

if sys.platform.startswith("darwin"):
    # macOS-specific code
    VLLM_TARGET_DEVICE = "cpu"
```

### 2. GPU vs CPU
Use appropriate device checks and fallbacks.

### 3. Backward Compatibility
Maintain backward compatibility when possible, deprecate with warnings.
