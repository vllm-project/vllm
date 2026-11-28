# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional process-local Torch thread cap during default weight loading."""

import os
from functools import wraps

import torch


def limit_load_threads(fn):
    """Restore the original intra-op thread count even if weight loading fails."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        limit = int(os.environ.get("VLLM_LOAD_TORCH_THREADS", "0"))
        old = torch.get_num_threads()
        if limit <= 0 or old <= limit:
            return fn(*args, **kwargs)
        torch.set_num_threads(limit)
        try:
            return fn(*args, **kwargs)
        finally:
            torch.set_num_threads(old)

    return wrapped
