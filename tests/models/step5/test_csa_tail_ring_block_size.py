# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.step5.common.csa_tail_ring import Step5CSATailRing
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.kv_cache_interface import FullAttentionSpec

_HEAD_SIZE = 192
_FP8 = torch.uint8
_DEFAULT_BLOCK_SIZE = 16


class _MultipleOf16(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "MULTIPLE_OF_16"

    @staticmethod
    def get_impl_cls():
        raise NotImplementedError

    @staticmethod
    def get_builder_cls():
        raise NotImplementedError

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [MultipleOf(16)]


class _MainLayer(AttentionLayerBase):
    def __init__(
        self, num_kv_heads: int, dtype: torch.dtype, page_size_padded: int | None
    ) -> None:
        self.num_kv_heads = num_kv_heads
        self.dtype = dtype
        self.page_size_padded = page_size_padded

    def get_attn_backend(self) -> type[AttentionBackend]:
        return _MultipleOf16

    def get_kv_cache_spec(self, vllm_config):
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=_HEAD_SIZE,
            dtype=self.dtype,
            page_size_padded=self.page_size_padded,
        )


@pytest.mark.parametrize(
    ("num_kv_heads", "dtype", "speculative_tokens", "page_size_padded", "expected"),
    [
        pytest.param(1, _FP8, 0, None, 32, id="fp8-tp8"),
        pytest.param(2, _FP8, 0, None, 16, id="fp8-tp2"),
        pytest.param(4, _FP8, 0, None, 16, id="fp8-tp1"),
        pytest.param(1, torch.bfloat16, 0, None, 16, id="bf16-tp8"),
        pytest.param(1, _FP8, 6, None, 64, id="fp8-tp8-mtp6"),
        pytest.param(1, _FP8, 0, 13312, 32, id="fp8-tp8-padded"),
    ],
)
def test_block_size_fits_tail_ring(
    num_kv_heads, dtype, speculative_tokens, page_size_padded, expected
):
    context: dict = {}
    ring = Step5CSATailRing(
        prefix="layers.0.csa_tail_ring",
        main_layer_name="layers.0.attn",
        num_speculative_tokens=speculative_tokens,
        static_forward_context=context,
    )
    context["layers.0.attn"] = _MainLayer(num_kv_heads, dtype, page_size_padded)
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=_DEFAULT_BLOCK_SIZE,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
        ),
        compilation_config=SimpleNamespace(static_forward_context=context),
        model_config=SimpleNamespace(is_hybrid=False),
    )
    with set_current_vllm_config(config):
        Platform.update_block_size_for_backend(config)
    ring.get_kv_cache_spec(config)
    assert ring.capacity >= ring.required_rows
    assert config.cache_config.block_size == expected
