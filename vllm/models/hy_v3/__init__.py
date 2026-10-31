# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HY V3 model entry point.

Unlike HY V4 (`vllm.models.hy_v4`), HY V3 has no custom kernels and no
hardware-specific code, so it needs only a `common/` directory.
"""

from .common.model import HYV3ForCausalLM
from .common.mtp import HYV3MTP

__all__ = ["HYV3ForCausalLM", "HYV3MTP"]
