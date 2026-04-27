# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused_qk_norm_rope helion kernel

Run `pytest tests/kernels/helion/test_fused_qk_norm_rope.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.benchmarks.lib.utils import default_vllm_config
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.fused_qk_norm_rope import (
    _pick_cache,
    baseline,
    fused_qk_norm_rope,
    pick_config,
)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


@default_vllm_config()
def _generate_fake_input(
    num_tokens: int, num_q_heads: int, num_kv_heads: int
) -> tuple[Any, ...]:
    with FakeTensorMode():
        head_dim = 128
        eps = 1e-6
        is_neox = True
        rotary_ratio = 1.0
        device = "cuda"
        dtype = torch.bfloat16
        total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
        qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        q_weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(head_dim,),
            dtype=qkv.dtype,
            device=device,
        )
        k_weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(head_dim,),
            dtype=qkv.dtype,
            device=device,
        )
        rotary_dim = int(head_dim * rotary_ratio)
        rope = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=4096,
            base=10000.0,
            is_neox_style=is_neox,
            dtype=dtype,
        ).to(device)
        args = (
            qkv,
            num_q_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        )
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestFusedQkNormRopeConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"q_heads": 2048, "kv_heads": 64, "num_tokens": 16}),
            CaseKey({"q_heads": 4096, "kv_heads": 128, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"q_heads": 4096, "kv_heads": 128, "num_tokens": 16}
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"q_heads": 2048, "kv_heads": 64, "num_tokens": 16}),
            CaseKey({"q_heads": 2048, "kv_heads": 64, "num_tokens": 32}),
            CaseKey({"q_heads": 2048, "kv_heads": 128, "num_tokens": 16}),
            CaseKey({"q_heads": 2048, "kv_heads": 128, "num_tokens": 32}),
            CaseKey({"q_heads": 4096, "kv_heads": 64, "num_tokens": 16}),
            CaseKey({"q_heads": 4096, "kv_heads": 64, "num_tokens": 32}),
            CaseKey({"q_heads": 4096, "kv_heads": 128, "num_tokens": 16}),
            CaseKey({"q_heads": 4096, "kv_heads": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"q_heads": 2048, "kv_heads": 64, "num_tokens": 32}
        )

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"q_heads": 2048, "kv_heads": 64, "num_tokens": 16}),
            CaseKey({"q_heads": 2048, "kv_heads": 64, "num_tokens": 32}),
            CaseKey({"q_heads": 2048, "kv_heads": 128, "num_tokens": 16}),
            CaseKey({"q_heads": 2048, "kv_heads": 128, "num_tokens": 32}),
            CaseKey({"q_heads": 4096, "kv_heads": 64, "num_tokens": 16}),
            CaseKey({"q_heads": 4096, "kv_heads": 64, "num_tokens": 32}),
            CaseKey({"q_heads": 4096, "kv_heads": 128, "num_tokens": 16}),
            CaseKey({"q_heads": 4096, "kv_heads": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"q_heads": 4096, "kv_heads": 128, "num_tokens": 32}
        )


class TestFusedQkNormRopeCorrectness:
    @pytest.mark.parametrize(
        "num_heads, num_kv_heads, head_dim", [(16, 4, 128), (64, 8, 128)]
    )
    @pytest.mark.parametrize("num_tokens", [1, 7, 1024, 1025])
    @pytest.mark.parametrize("is_neox", [False, True])
    @pytest.mark.parametrize("rotary_ratio", [1.0, 0.5, 0.25])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @default_vllm_config()
    def test_fused_qk_norm_rope(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        num_tokens: int,
        is_neox: bool,
        rotary_ratio: float,
        dtype: torch.dtype,
    ):
        skip_if_platform_unsupported("fused_qk_norm_rope")

        torch.manual_seed(42)
        eps = 1e-6
        device = "cuda"
        total_dim = (num_heads + 2 * num_kv_heads) * head_dim
        ref_qkv = torch.empty(
            num_tokens, total_dim, dtype=dtype, device=device
        ).uniform_(-0.1, 0.1)
        ops_qkv = ref_qkv.clone()
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        q_weight = torch.empty(head_dim, dtype=dtype, device=device).uniform_(0.8, 1.2)
        k_weight = torch.empty(head_dim, dtype=dtype, device=device).uniform_(0.8, 1.2)
        rotary_dim = int(head_dim * rotary_ratio)
        rope = RotaryEmbedding(
            head_size=head_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=40960,
            base=10000.0,
            is_neox_style=is_neox,
            dtype=dtype,
        ).to(device)

        baseline(
            ref_qkv,
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        )

        fused_qk_norm_rope(
            ops_qkv,
            num_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            rope.cos_sin_cache,
            is_neox,
            positions.view(-1),
        )

        if dtype == torch.bfloat16:
            atol = 5e-2
            rtol = 5e-2
        else:
            atol = 1e-2
            rtol = 1e-2

        torch.testing.assert_close(
            ref_qkv,
            ops_qkv,
            atol=atol,
            rtol=rtol,
        )


class TestFusedQkNormRopeIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "fused_qk_norm_rope" in registered_kernels

        kernel_wrapper = registered_kernels["fused_qk_norm_rope"]
        assert kernel_wrapper.op_name == "fused_qk_norm_rope"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["qkv"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("fused_qk_norm_rope")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["fused_qk_norm_rope"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
