# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate JIT dispatch against pre-contract behavior."""

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

import vllm.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils_module
import vllm.model_executor.layers.fused_moe.moe_fused_mul_sum as mul_sum_module
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    DeepGemmEPGatherKernel,
    DeepGemmEPScatterCopyKernel,
    DeepGemmEPScatterStartKernel,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    TrtLlmLoraFinalizeKernel,
    TrtLlmLoraUnpermuteActivationKernel,
)
from vllm.model_executor.layers.fused_moe.fused_moe import ComputeIdentityKernel
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    MoeFusedMulSumKernel,
)
from vllm.model_executor.layers.fused_moe.router.base_router import (
    EplbMapAndRecordKernel,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import DSV4TopKKernel
from vllm.model_executor.layers.fused_moe.utils import CountExpertNumTokensKernel
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
    _PREPARE_MEGAMOE_INPUTS_KERNEL,
)


def test_deepseek_v4_mega_moe_prepare_inputs_warmup_keys() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                hidden_size=256,
                num_experts_per_tok=6,
            )
        ),
        kernel_config=SimpleNamespace(moe_backend="auto"),
    )

    assert _PREPARE_MEGAMOE_INPUTS_KERNEL.get_warmup_keys(vllm_config) == [
        _PREPARE_MEGAMOE_INPUTS_KERNEL.CompileKey(
            hidden_size=256,
            top_k=6,
            block_topk=8,
            has_padding=False,
            has_shared_x_sf=False,
            shared_block_m=1,
        ),
        _PREPARE_MEGAMOE_INPUTS_KERNEL.CompileKey(
            hidden_size=256,
            top_k=6,
            block_topk=8,
            has_padding=True,
            has_shared_x_sf=False,
            shared_block_m=1,
        ),
    ]


def test_deep_gemm_scatter_start_dispatch_matches_legacy_meta() -> None:
    kernel = DeepGemmEPScatterStartKernel()

    assert kernel.dispatch(num_experts=257, align_m=128) == kernel.CompileKey(
        num_experts=257,
        block_e=128,
        block_expert_num=512,
        align_m=128,
    )


def test_deep_gemm_scatter_start_warmup_covers_dynamic_alignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        deep_gemm_utils_module,
        "get_mk_alignment_for_contiguous_layout",
        lambda: (128, 128),
    )
    monkeypatch.setattr(
        deep_gemm_utils_module,
        "get_theoretical_mk_alignment_for_contiguous_layout",
        lambda *, expected_m, num_groups: 32 if expected_m <= 8 else 48,
    )
    vllm_config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="auto"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                model_type="deepseek_v4",
                n_routed_experts=256,
                num_experts_per_tok=8,
            )
        ),
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            enable_expert_parallel=False,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
    )

    assert DeepGemmEPScatterStartKernel().get_warmup_keys(vllm_config) == [
        DeepGemmEPScatterStartKernel.CompileKey(
            num_experts=256,
            block_e=128,
            block_expert_num=256,
            align_m=32,
        ),
        DeepGemmEPScatterStartKernel.CompileKey(
            num_experts=256,
            block_e=128,
            block_expert_num=256,
            align_m=48,
        ),
    ]


@pytest.mark.parametrize("has_expert_map", [False, True])
def test_count_expert_num_tokens_compile_matches_optional_pointer(
    monkeypatch: pytest.MonkeyPatch,
    has_expert_map: bool,
) -> None:
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    kernel = CountExpertNumTokensKernel()
    monkeypatch.setattr(
        kernel,
        "kernel",
        SimpleNamespace(warmup=lambda *args, **kwargs: calls.append((args, kwargs))),
    )
    compile_key = kernel.CompileKey(
        num_experts=16,
        topk_numel=16,
        has_expert_map=has_expert_map,
        block_size=16,
    )

    kernel.compile(compile_key)

    assert (calls[0][0][4] is not None) is has_expert_map


@pytest.mark.parametrize(
    ("pack_ue8m0", "expected_packed", "expected_packed_pad"),
    [(False, 1, 1), (True, 14, 16)],
)
def test_deep_gemm_scatter_copy_dispatch_matches_legacy_meta(
    pack_ue8m0: bool,
    expected_packed: int,
    expected_packed_pad: int,
) -> None:
    kernel = DeepGemmEPScatterCopyKernel()

    assert kernel.dispatch(
        total_token_num=17,
        hidden_size=7168,
        topk_num=8,
        has_expert_map=True,
        block_size=128,
        pack_ue8m0=pack_ue8m0,
    ) == kernel.CompileKey(
        total_token_num=2,
        topk_num=8,
        has_expert_map=True,
        hidden_size=7168,
        hidden_size_pad=8192,
        scale_hidden_size=56,
        scale_hidden_size_pad=64,
        pack_ue8m0=pack_ue8m0,
        scale_packed_size=expected_packed,
        scale_packed_size_pad=expected_packed_pad,
    )


def test_deep_gemm_gather_dispatch_matches_runtime_layout() -> None:
    kernel = DeepGemmEPGatherKernel()

    assert kernel.dispatch(
        dtype=torch.bfloat16,
        total_token_num=16,
        hidden_size=7168,
        topk_num=8,
        has_expert_map=False,
    ) == kernel.CompileKey(
        dtype=torch.bfloat16,
        total_token_num=16,
        topk_num=8,
        has_expert_map=False,
        block_d=1024,
        hidden_stride=16,
        topk_stride=2,
    )


def test_trtllm_lora_unpermute_dispatch_matches_legacy_meta() -> None:
    kernel = TrtLlmLoraUnpermuteActivationKernel()

    assert kernel.dispatch(
        dtype=torch.bfloat16,
        intermediate_size=18432,
    ) == kernel.CompileKey(
        dtype=torch.bfloat16,
        num_cols=18432,
        block_i=1024,
    )


def test_trtllm_lora_finalize_dispatch_matches_legacy_meta() -> None:
    kernel = TrtLlmLoraFinalizeKernel()

    assert kernel.dispatch(
        dtype=torch.bfloat16,
        hidden_size=7168,
        top_k=8,
    ) == kernel.CompileKey(
        dtype=torch.bfloat16,
        hidden_size=7168,
        top_k=8,
        block_k=512,
    )


def test_compute_identity_dispatch_matches_legacy_meta() -> None:
    kernel = ComputeIdentityKernel()

    assert kernel.dispatch(top_k=8, hidden_dim=7168) == kernel.CompileKey(
        top_k=8,
        hidden_dim=7168,
        block_size=256,
    )


@dataclass
class _FakePlatform:
    capability: int

    def has_device_capability(self, capability: int) -> bool:
        return self.capability >= capability


@pytest.mark.parametrize(
    ("capability", "kwargs", "expected"),
    [
        (
            90,
            dict(
                num_tokens=64,
                top_k=8,
                size=7168,
                element_size=4,
                dtype=torch.float32,
                has_expert_map=False,
            ),
            (torch.float32, False, 8, 7168, 2, 256, 8, 4),
        ),
        (
            80,
            dict(
                num_tokens=64,
                top_k=8,
                size=7168,
                element_size=2,
                dtype=torch.bfloat16,
                has_expert_map=True,
            ),
            (torch.bfloat16, True, 8, 7168, 4, 1024, 16, 2),
        ),
        (
            70,
            dict(
                num_tokens=2048,
                top_k=8,
                size=512,
                element_size=2,
                dtype=torch.bfloat16,
                has_expert_map=False,
            ),
            (torch.bfloat16, False, 8, 512, 8, 512, 8, 2),
        ),
    ],
)
def test_moe_fused_mul_sum_dispatch_matches_legacy_heuristic(
    monkeypatch: pytest.MonkeyPatch,
    capability: int,
    kwargs: dict[str, Any],
    expected: tuple[torch.dtype, bool, int, int, int, int, int, int],
) -> None:
    monkeypatch.setattr(mul_sum_module, "current_platform", _FakePlatform(capability))
    kernel = MoeFusedMulSumKernel()

    assert kernel.dispatch(**kwargs) == kernel.CompileKey(*expected)


def test_globalize_recv_topk_dispatch_matches_legacy_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_name = "vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2"
    if importlib.util.find_spec("deep_ep") is None:
        deep_ep_stub = ModuleType("deep_ep")
        for type_name in ("ElasticBuffer", "EPHandle", "EventOverlap"):
            setattr(deep_ep_stub, type_name, type(type_name, (), {}))
        with monkeypatch.context() as context:
            context.setitem(sys.modules, "deep_ep", deep_ep_stub)
            module = importlib.import_module(module_name)
        sys.modules.pop(module_name, None)
    else:
        module = importlib.import_module(module_name)
    kernel = module.GlobalizeRecvTopkIdxKernel()

    assert kernel.dispatch(
        num_tokens=17,
        topk=8,
        P=4,
        rank_expert_offset=64,
        num_experts=256,
    ) == kernel.CompileKey(
        n_elements=136,
        topk=8,
        p=4,
        rank_expert_offset=64,
        num_experts=256,
        block=1024,
    )


@pytest.mark.parametrize("has_num_unpadded", [False, True])
def test_eplb_map_and_record_dispatch_matches_legacy_meta(
    has_num_unpadded: bool,
) -> None:
    kernel = EplbMapAndRecordKernel()

    assert kernel.dispatch(
        has_num_unpadded=has_num_unpadded,
        num_active_experts=256,
    ) == kernel.CompileKey(
        has_num_unpadded=has_num_unpadded,
        num_active_experts=256,
        block_size=256,
    )


def test_dsv4_topk_dispatch_matches_legacy_meta() -> None:
    kernel = DSV4TopKKernel()

    assert kernel.dispatch(
        num_experts=256,
        indices_dtype=torch.int32,
        routed_scaling_factor=2.5,
        launch_pdl=True,
    ) == kernel.CompileKey(
        num_experts=256,
        block_n=256,
        indices_dtype=torch.int32,
        routed_scaling_factor=2.5,
        launch_pdl=True,
    )
