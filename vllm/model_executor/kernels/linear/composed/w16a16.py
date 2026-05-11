# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import vllm.model_executor.kernels.linear.aiter.w16a16 as aiter_w16a16
import vllm.model_executor.kernels.linear.base.w16a16 as w16a16
import vllm.model_executor.kernels.linear.vllm_c.w16a16 as vllm_c_w16a16


class Kernel(w16a16.Composite):
    """ROCm w16a16 dispatch chain.
    WvSplitKrc → aiter_skinny → WvSplitK → LLMM1 → aiter_tgemm → F.linear
    """

    _scheme_tag = "rocm_w16a16"
    _chain = [
        vllm_c_w16a16.WvSplitKrcKernel,
        aiter_w16a16.SkinnyKernel,
        vllm_c_w16a16.WvSplitKKernel,
        vllm_c_w16a16.LLMM1Kernel,
        aiter_w16a16.Kernel,
    ]
