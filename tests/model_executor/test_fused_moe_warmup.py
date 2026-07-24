# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.model_executor.warmup.fused_moe_warmup as fused_moe_warmup
from vllm.model_executor.layers.fused_moe.fused_moe import (
    Wna16TritonKernel,
    Wna16TritonWarmupConfig,
    fused_moe_kernel_gptq_awq,
)
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    trace_triton_kernel_specialization_args,
)
from vllm.triton_utils import tl


def _warmup_config(**overrides) -> Wna16TritonWarmupConfig:
    values = dict(
        a_dtype=torch.bfloat16,
        b_dtype=torch.uint8,
        c_dtype=torch.bfloat16,
        b_scale_dtype=torch.bfloat16,
        b_zp_dtype=None,
        topk_weights_dtype=None,
        N=256,
        K=128,
        stride_am=128,
        stride_ak=1,
        stride_be=16384,
        stride_bk=1,
        stride_bn=64,
        stride_cm=256,
        stride_cn=1,
        stride_bse=512,
        stride_bsk=1,
        stride_bsn=2,
        stride_bze=0,
        stride_bzk=0,
        stride_bzn=0,
        block_k_diviable=True,
        group_size=64,
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=32,
        BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,
        SPLIT_K=1,
        MUL_ROUTED_WEIGHT=False,
        top_k=2,
        compute_type=tl.bfloat16,
        has_zp=False,
        use_int4_w4a16=True,
        use_int8_w8a16=False,
        num_warps=4,
        num_stages=3,
    )
    values.update(overrides)
    return Wna16TritonWarmupConfig(**values)


def test_fused_moe_gptq_awq_kernel_does_not_specialize_token_counts() -> None:
    specialization_args = set(
        trace_triton_kernel_specialization_args(fused_moe_kernel_gptq_awq)
    )
    compile_key_fields = set(Wna16TritonKernel.CompileKey.__dataclass_fields__)

    assert "EM" not in specialization_args
    assert "num_valid_tokens" not in specialization_args
    assert specialization_args <= compile_key_fields
    assert set(fused_moe_kernel_gptq_awq.do_not_specialize) >= {
        "EM",
        "num_valid_tokens",
    }


def test_wna16_compile_keys_dedupe_and_cover_static_meta() -> None:
    kernel = Wna16TritonKernel()
    config = _warmup_config()

    keys = kernel.get_warmup_keys([config, config, _warmup_config(BLOCK_SIZE_M=32)])

    assert len(keys) == 2
    assert {key.BLOCK_SIZE_M for key in keys} == {16, 32}
    assert all(key.N == 256 and key.K == 128 for key in keys)


def test_layer_warmup_configs_cover_all_runtime_config_buckets(monkeypatch) -> None:
    monkeypatch.setattr(
        fused_moe_warmup,
        "try_get_optimal_moe_config",
        lambda *args, **kwargs: {
            "BLOCK_SIZE_M": 16 if args[4] <= 20 else 32,
            "GROUP_SIZE_M": 1,
            "SPLIT_K": 1,
        },
    )

    w1 = torch.empty((4, 256, 64), dtype=torch.uint8)
    w2 = torch.empty((4, 128, 64), dtype=torch.uint8)
    layer = SimpleNamespace(
        moe_config=SimpleNamespace(in_dtype=torch.bfloat16),
        w13_weight=w1,
        w2_weight=w2,
        top_k=2,
        apply_router_weight_on_input=False,
    )
    quant_config = SimpleNamespace(
        use_int4_w4a16=True,
        use_int8_w8a16=False,
        w1_zp=None,
        w2_zp=None,
        config_name=lambda dtype: "int4_w4a16",
    )
    experts = SimpleNamespace(
        block_shape=[0, 64],
        quant_config=quant_config,
        w1_scale=torch.empty((4, 256, 2), dtype=torch.bfloat16),
        w2_scale=torch.empty((4, 128, 2), dtype=torch.bfloat16),
    )

    configs = fused_moe_warmup._layer_warmup_configs(layer, experts, 21)
    keys = Wna16TritonKernel().get_warmup_keys(configs)

    assert len(configs) == 42
    assert {key.BLOCK_SIZE_M for key in keys} == {16, 32}
    assert {key.MUL_ROUTED_WEIGHT for key in keys} == {False, True}
    assert {key.top_k for key in keys} == {1, 2}
    assert {key.topk_weights_dtype for key in keys} == {None, torch.float32}
