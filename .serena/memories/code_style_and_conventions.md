# Code Style and Conventions

## Style Guides
- **Python**: [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- **C++/CUDA**: [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

## Python Code Conventions

### Type Hints
- Use modern Python type hint syntax (PEP 604): `int | None` instead of `Optional[int]`
- Use `list[str]` instead of `List[str]`
- Use `dict[str, Any]` instead of `Dict[str, Any]`
- Type hints are checked with mypy (configured in pyproject.toml)
- Use `from typing import Literal, cast, Annotated` for advanced types

### Docstrings
- Use triple-quoted docstrings with detailed descriptions
- Class docstrings should describe the purpose of the class
- Attribute docstrings should be inline using triple-quotes after the attribute declaration
- Method docstrings should describe parameters and return values when needed

Example:
```python
class SamplingParams:
    """Sampling parameters for text generation.
    
    Overall, we follow the sampling parameters from the OpenAI text completion
    API. In addition, we support beam search.
    """
    
    n: int = 1
    """Number of outputs to return for the given prompt request."""
```

### File Headers
All Python files must have SPDX license headers:
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

### Naming Conventions
- Use snake_case for functions, variables, and modules
- Use PascalCase for classes
- Use UPPER_CASE for constants
- Private/internal attributes/methods prefixed with underscore (_)

### Code Organization
- Imports: standard library, third-party, local (separated by blank lines)
- Use `import regex as re` instead of plain `import re` (enforced by pre-commit)
- Avoid direct `import triton` (use lazy imports when needed)
- Prevent new pickle/cloudpickle imports (checked by pre-commit)

### Other Conventions
- Use f-strings for string formatting
- Use `logger` from `vllm.logger` for logging
- Follow dataclass/msgspec.Struct patterns for configuration classes
- Validation in `__post_init__` methods

## Linting and Formatting

### Ruff Configuration
The project uses ruff for linting and formatting with the following rules:
- pycodestyle (E)
- Pyflakes (F)
- pyupgrade (UP)
- flake8-bugbear (B)
- flake8-simplify (SIM)
- isort (I)
- flake8-logging-format (G)

Ignored rules:
- F405, F403: star imports
- E731: lambda expression assignment
- B905: zip without `strict=`
- B007: Loop control variable not used
- UP032: f-string format

### Code Quality Requirements
- All code must pass pre-commit checks before committing
- Well-documented code with comments explaining complex logic
- Sufficient tests (unit and integration)
- Documentation updates for user-facing changes
