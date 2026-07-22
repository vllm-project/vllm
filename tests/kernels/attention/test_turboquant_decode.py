# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    trace_triton_kernel_specialization_args,
)


def test_turboquant_kernels_do_not_specialize_runtime_strides() -> None:
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _tq_decode_stage1,
        _tq_full_dequant_kv,
    )

    assert set(_tq_decode_stage1.do_not_specialize) == {
        "stride_qb",
        "stride_qh",
        "stride_bt_b",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_mid_b",
        "stride_mid_h",
        "stride_mid_s",
    }
    assert set(_tq_full_dequant_kv.do_not_specialize) == {
        "stride_bt_b",
        "stride_cache_block",
        "stride_cache_pos",
        "stride_cache_head",
        "stride_ko_b",
        "stride_ko_h",
        "stride_ko_s",
        "stride_vo_b",
        "stride_vo_h",
        "stride_vo_s",
    }


def test_turboquant_kernels_own_their_compile_keys() -> None:
    from vllm.v1.attention.ops.triton_decode_attention import (
        _DECODE_STAGE2_KERNEL,
        _fwd_kernel_stage2,
    )
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _TQ_DECODE_STAGE1_KERNEL,
        _TQ_FULL_DEQUANT_KERNEL,
        _tq_decode_stage1,
        _tq_full_dequant_kv,
    )

    kernels: tuple[tuple[VllmJitKernel[Any], Any, set[str]], ...] = (
        (_TQ_DECODE_STAGE1_KERNEL, _tq_decode_stage1, {"BLOCK_KV"}),
        (_TQ_FULL_DEQUANT_KERNEL, _tq_full_dequant_kv, set()),
        (_DECODE_STAGE2_KERNEL, _fwd_kernel_stage2, set()),
    )
    for owner, triton_kernel, fixed_specializations in kernels:
        assert isinstance(owner, VllmJitKernel)
        compile_key_fields = owner.CompileKey.__dataclass_fields__.keys()
        specialization_args = set(
            trace_triton_kernel_specialization_args(triton_kernel)
        )
        assert specialization_args <= set(compile_key_fields) | fixed_specializations


def test_turboquant_shared_warmup_contract(monkeypatch) -> None:
    from vllm.model_executor.warmup.turboquant_triton_warmup import (
        turboquant_triton_warmup,
    )
    from vllm.v1.attention.ops.triton_decode_attention import (
        _DECODE_STAGE2_KERNEL,
    )
    from vllm.v1.attention.ops.triton_turboquant_decode import (
        _TQ_DECODE_STAGE1_KERNEL,
        _TQ_FULL_DEQUANT_KERNEL,
    )

    calls = []
    owners = (
        _TQ_DECODE_STAGE1_KERNEL,
        _TQ_FULL_DEQUANT_KERNEL,
        _DECODE_STAGE2_KERNEL,
    )
    for owner in owners:
        monkeypatch.setattr(
            owner, "warmup", lambda config, owner=owner: calls.append(owner)
        )

    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="turboquant_4bit_nc")
    )
    worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            is_pooling_model=False,
            vllm_config=vllm_config,
        )
    )
    turboquant_triton_warmup(worker)

    assert calls == list(owners)
