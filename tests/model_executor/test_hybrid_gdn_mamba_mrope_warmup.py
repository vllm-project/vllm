# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.layers.rotary_embedding.mrope import (
    MropeKernel,
    MropeWarmupConfig,
)
from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
    PackedGdnDecodeKernel,
    PackedGdnDecodeWarmupConfig,
)


def test_mrope_warmup_keys_cover_integer_specializations() -> None:
    config = MropeWarmupConfig(
        q_dtype=torch.bfloat16,
        k_dtype=torch.bfloat16,
        cos_dtype=torch.float32,
        sin_dtype=torch.float32,
        n_qh=28,
        n_kh=4,
        head_size=128,
        rotary_dim=64,
        mrope_section=(8, 12, 12),
        is_interleaved=False,
        is_neox_style=False,
    )

    keys = MropeKernel().get_warmup_keys([config, config])

    assert len(keys) == 2
    assert {key.num_tokens_divisible_by_16 for key in keys} == {False, True}
    assert all(key.pad_n_qh == 32 for key in keys)
    assert all(key.pad_n_kh == 4 for key in keys)
    assert all(key.pad_rd == 64 for key in keys)
    assert all(not key.is_neox_style for key in keys)
    expected_num_warps = 1 if current_platform.is_rocm() else 4
    assert all(key.num_warps == expected_num_warps for key in keys)


def test_mrope_warmup_config_tracks_runtime_layout(monkeypatch) -> None:
    from vllm.model_executor.layers.rotary_embedding import mrope
    from vllm.model_executor.warmup import hybrid_gdn_mamba_mrope_warmup as warmup

    class FakeMRotaryEmbedding:
        mrope_section = [8, 12, 12]
        cos_sin_cache = torch.empty(1, dtype=torch.float32)
        rotary_dim = 64
        mrope_interleaved = True
        is_neox_style = False

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rotary_emb = FakeMRotaryEmbedding()
            self.num_heads = 28
            self.num_kv_heads = 4
            self.head_dim = 128

    monkeypatch.setattr(mrope, "MRotaryEmbedding", FakeMRotaryEmbedding)

    configs = warmup._mrope_configs(Attention(), torch.bfloat16)

    assert configs == [
        MropeWarmupConfig(
            q_dtype=torch.bfloat16,
            k_dtype=torch.bfloat16,
            cos_dtype=torch.float32,
            sin_dtype=torch.float32,
            n_qh=28,
            n_kh=4,
            head_size=128,
            rotary_dim=64,
            mrope_section=(8, 12, 12),
            is_interleaved=True,
            is_neox_style=False,
        )
    ]


def test_packed_gdn_warmup_key_matches_kernel_meta_parameters() -> None:
    config = PackedGdnDecodeWarmupConfig(
        mixed_qkv_dtype=torch.bfloat16,
        a_dtype=torch.bfloat16,
        b_dtype=torch.bfloat16,
        A_log_dtype=torch.float32,
        dt_bias_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        state_dtype=torch.float32,
        scale=128**-0.5,
        stride_mixed_qkv_tok=8192,
        stride_a_tok=32,
        stride_b_tok=32,
        stride_init_state_token=524288,
        stride_final_state_token=524288,
        stride_indices_seq=1,
        H=16,
        HV=32,
        K=128,
        V=128,
        use_qk_l2norm_in_kernel=True,
    )

    keys = PackedGdnDecodeKernel().get_warmup_keys([config, config])

    assert len(keys) == 1
    assert keys[0].BK == 128
    assert keys[0].BV == 32
    assert keys[0].USE_QK_L2NORM_IN_KERNEL is True


def test_shared_warmup_uses_kernel_owned_compile_only_paths(monkeypatch) -> None:
    from vllm.model_executor.warmup import hybrid_gdn_mamba_mrope_warmup as warmup

    packed_configs = [object()]
    mrope_configs = [object()]
    calls: list[tuple[str, list[object]]] = []

    monkeypatch.setattr(warmup, "_packed_gdn_configs", lambda *args: packed_configs)
    monkeypatch.setattr(warmup, "_mrope_configs", lambda *args: mrope_configs)
    monkeypatch.setattr(
        "vllm.third_party.flash_linear_attention.ops.fused_recurrent"
        "._PACKED_GDN_DECODE_KERNEL.warmup",
        lambda configs: calls.append(("packed", configs)),
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.rotary_embedding.mrope._MROPE_KERNEL.warmup",
        lambda configs: calls.append(("mrope", configs)),
    )

    model = torch.nn.Module()
    worker = SimpleNamespace(
        get_model=lambda: model,
        model_runner=SimpleNamespace(is_pooling_model=False),
        vllm_config=SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16)),
    )

    warmup.hybrid_gdn_mamba_mrope_warmup(worker)

    assert calls == [("packed", packed_configs), ("mrope", mrope_configs)]
