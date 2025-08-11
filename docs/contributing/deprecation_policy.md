# Deprecation Policy

This document outlines the official policy and process for deprecating features
in the vLLM project.

## Overview

vLLM uses a structured "deprecation pipeline" to guide the lifecycle of
deprecated features. This policy ensures that users are given clear and
sufficient notice when a feature is deprecated and that deprecations proceed in
a consistent and predictable manner.

We aim to strike a balance between continued innovation and respecting users’
reliance on existing functionality. Deprecations are tied to our **minor (Y)
releases** following semantic versioning (X.Y.Z), where:

- **X** is a major version (rare)
- **Y** is a minor version (used for significant changes, including deprecations/removals)
- **Z** is a patch version (used for fixes and safer enhancements)

Features that fall under this policy include (at a minimum) the following:

- CLI flags
- Environment variables
- Configuration files
- APIs in the OpenAI-compatible API server
- Public Python APIs for the `vllm` library

## Deprecation Pipeline

The deprecation process consists of several clearly defined stages that span
multiple Y releases:

### 1. Deprecated (Still On By Default)

- **Action**: Feature is marked as deprecated.
- **Timeline**: A removal version is explicitly stated in the deprecation
warning (e.g., "This will be removed in v0.10.0").
- **Communication**: Deprecation is noted in the following, as applicable:
    - Help strings
    - Log output
    - API responses
    - `/metrics` output (for metrics features)
    - User-facing documentation
    - Release notes
    - GitHub Issue (RFC) for feedback
    - Documentation and use of the `@typing_extensions.deprecated` decorator for Python APIs

### 2.Deprecated (Off By Default)

- **Action**: Feature is disabled by default, but can still be re-enabled via a
CLI flag or environment variable. Feature throws an error when used without
re-enabling.
- **Purpose**: Allows users who missed earlier warnings a temporary escape hatch
while signaling imminent removal. Ensures any remaining usage is clearly
surfaced and blocks silent breakage before full removal.

### 3. Removed

- **Action**: Feature is completely removed from the codebase.
- **Note**: Only features that have passed through the previous deprecation
stages will be removed.

## Example Timeline

Assume a feature is deprecated in `v0.9.0`.

| Release       | Status                                                                                          |
|---------------|-------------------------------------------------------------------------------------------------|
| `v0.9.0`      | Feature is deprecated with clear removal version listed.                                        |
| `v0.10.0`     | Feature is now off by default, throws an error when used, and can be re-enabled for legacy use. |
| `v0.11.0`     | Feature is removed.                                                                             |

## Important Guidelines

- **No Removals in Patch Releases**: Removing deprecated features in patch
(`.Z`) releases is disallowed to avoid surprising users.
- **Grace Period for Existing Deprecations**: Any feature deprecated **before
this policy** will have its grace period start **now**, not retroactively.
- **Documentation is Critical**: Ensure every stage of the pipeline is
documented clearly for users.

## Final Notes

This policy is a living document and may evolve as the needs of the project and
its users change. Community feedback is welcome and encouraged as we refine the
process.
