# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model geometry constants shared by the Turing (SM75) kernels.

DeepSeek-V4-Flash MLA heads are 512-wide: a 448-dim nope latent plus a 64-dim
RoPE tail. The base class sizes buffers from the model config, but the Triton
kernels here take HEAD_DIM/ROPE_DIM as constexpr and must stay in sync, so they
are centralized in one place.
"""

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2
