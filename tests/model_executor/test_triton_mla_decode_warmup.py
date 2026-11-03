# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the Triton MLA decode warmup."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.warmup import triton_mla_decode_warmup
from vllm.v1.attention.backends.mla.triton_mla import TritonMLAImpl


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("kv_cache_dtype", "expected_dtype"),
    [("auto", torch.bfloat16), ("fp8", torch.float8_e4m3fn)],
)
def test_decode_warmup_uses_runtime_kv_cache_dtype(
    monkeypatch: pytest.MonkeyPatch, kv_cache_dtype: str, expected_dtype: torch.dtype
) -> None:
    impl = object.__new__(TritonMLAImpl)
    impl.num_heads = 16
    impl.dcp_world_size = 1
    impl.kv_lora_rank = 512
    impl.scale = 1.0
    layer: Any = object.__new__(MLAAttention)
    layer.impl = impl
    layer.layer_name = "layer.0"
    layer.kv_cache_dtype = kv_cache_dtype
    layer._k_scale = torch.ones(1)
    # Like the runtime allocation: fp8 caches are stored as uint8.
    layer.kv_cache = torch.zeros(
        (2, 16, 576), dtype=torch.uint8 if kv_cache_dtype == "fp8" else torch.bfloat16
    )
    cache_dtypes: list[torch.dtype] = []

    def decode_attention_fwd(
        q: torch.Tensor, cache: torch.Tensor, *args: Any, **kwargs: Any
    ) -> None:
        cache_dtypes.append(cache.dtype)

    monkeypatch.setattr(
        triton_mla_decode_warmup, "decode_attention_fwd", decode_attention_fwd
    )
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "target_and_draft_models",
        lambda worker: [SimpleNamespace(modules=lambda: [layer])],
    )
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "kv_cache_group_ids",
        lambda worker: {"layer.0": 0},
    )
    monkeypatch.setattr(
        triton_mla_decode_warmup,
        "block_table",
        lambda worker, gid, num_reqs: torch.zeros((1, 8), dtype=torch.int32),
    )
    worker: Any = SimpleNamespace(
        model_runner=SimpleNamespace(
            is_pooling_model=False,
            device="cpu",
            model_config=SimpleNamespace(dtype=torch.bfloat16),
        )
    )

    triton_mla_decode_warmup.triton_mla_decode_warmup(worker)

    # MLAAttention.forward_impl views a quantized cache as fp8 before decode,
    # so the warmup must compile that specialization (one call per seq_lens
    # dtype).
    assert cache_dtypes == [expected_dtype, expected_dtype]
