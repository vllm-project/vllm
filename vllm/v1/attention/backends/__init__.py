# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# in priority/performance order (when available)
_POSSIBLE_BACKEND_LIST: list[str] = [
    "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend",
    "vllm.v1.attention.backends.triton_attn.TritonSplitPrefillDecodeAttentionBackend",
    "vllm.v1.attention.backends.triton_attn.TritonUnifiedAttentionBackend",
]


def choose_attention_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    block_size: int,
) -> str:
    from vllm.attention.selector import is_attn_backend_supported

    for backend in _POSSIBLE_BACKEND_LIST:

        backend_obj_name = backend.rsplit(".", 1)[1].removesuffix("Backend")

        if backend_obj_name in envs.VLLM_DISABLED_BACKENDS:
            message = f"{backend_obj_name} has been disabled. Remove \
                      {backend_obj_name} from VLLM_DISABLED_BACKENDS \
                      to re-enable it."

            logger.warning(message)
            continue

        if is_attn_backend_supported(backend,
                                     head_size,
                                     dtype,
                                     kv_cache_dtype,
                                     block_size,
                                     allow_import_error=False):
            return backend

    raise ValueError("Couldn't find any supported backend.")
