# Contributing to CacheFlow

Thank you for your interest in contributing to CacheFlow!

### Install CacheFlow for development

```bash
pip install -r requirements.txt
pip install -e .  # This may take several minutes.
```

### Test CacheFlow

```bash
pip install -r requirements-dev.txt

# Static type checking
mypy
# Unit tests
pytest tests
```

**Note:** Currently, the repository does not pass the mypy tests.

### Style Guide

In general, we adhere to [Google Python style guide](https://google.github.io/styleguide/pyguide.html) and [Google C++ style guide](https://google.github.io/styleguide/cppguide.html).
