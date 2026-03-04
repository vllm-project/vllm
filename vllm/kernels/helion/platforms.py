# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Platform tier constants for Helion kernel versioning policy."""

# Platforms that a new kernel version must have configs for before it can
# coexist alongside an older version.  CI enforces this via
# test_newest_version_has_core_coverage.
CORE_PLATFORMS: frozenset[str] = frozenset({"nvidia_h100", "nvidia_h200"})
