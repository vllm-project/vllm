
# Guide: Updating PyTorch Nightly Dependencies for vLLM

This document provides guidance on managing PyTorch nightly dependencies in the vLLM project.

## Overview

The vLLM project uses an automated system to manage PyTorch nightly dependencies:

- **Pre-commit hook**: Automatically modifies `requirements/nightly_torch_test.txt` using the script `tools/generate_nightly_torch_test.py`
- **Dependency validation**: A test ensures that dependencies don't override PyTorch nightly versions

## Conflict Resolution

When `requirements/nightly_torch_test.txt` causes compatibility issues, follow these steps:

### Option 1: Whitelist the Dependency
1. Add the problematic dependency to the `white_list` in `tools/generate_nightly_torch_test.py`
2. This allows the dependency to coexist with PyTorch nightly

### Option 2: Manual Dependency Management
1. Add the dependency to `requirements/nightly_torch_test_manual.txt`
2. This file contains manually managed dependencies that bypass the automated system

### Option 3: Docker Build Requirements
If a dependency needs to be built from source and is used across multiple tests:

1. Contact the PyTorch dev infra team for assistance
2. Alternatively, reach out to the vLLM team to add the necessary build steps to the Docker configuration

## Files Involved

- `requirements/nightly_torch_test.txt` - Auto-generated nightly test requirements
- `requirements/nightly_torch_test_manual.txt` - Manually managed dependencies
- `tools/generate_nightly_torch_test.py` - Script that generates nightly requirements
- Docker configurations - For dependencies requiring source builds
