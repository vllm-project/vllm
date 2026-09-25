# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Share pooling/'s autouse GPU-memory cleanup with the core pooling tests."""

from ...pooling.conftest import release_gpu_memory_between_tests  # noqa: F401
