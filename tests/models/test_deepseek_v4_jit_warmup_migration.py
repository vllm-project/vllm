# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate DSv4 JIT dispatch and warmup keys against pre-contract behavior."""

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_cutedsl

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA DSv4 dispatch tests require CUDA", allow_module_level=True)

import vllm.model_executor.layers.fused_moe.deep_gemm_utils as deep_gemm_utils_module
import vllm.model_executor.layers.fused_moe.moe_fused_mul_sum as mul_sum_module
from vllm.model_executor.kernels.mhc.tilelang_kernels import (
    HcPrenormGemmTileLangKernel,
    MhcFusedTileLangKernel,
    MhcPreBigFuseTileLangKernel,
)
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
from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (
    BF16x3SplitKReduceKernel,
)
from vllm.model_executor.layers.fused_moe.router.dsv4_topk import DSV4TopKKernel
from vllm.model_executor.layers.fused_moe.utils import CountExpertNumTokensKernel
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    FusedKVCompressNormRopeInsertIndexerTritonKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_inv_rope_fp8_quant import (
    FusedInvRopeFP8QuantKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_mtp_input_rmsnorm import (
    FusedMTPInputRMSNormKernel,
    MTPSharedHeadRMSNormKernel,
)
from vllm.models.deepseek_v4.common.ops.fused_qk_rmsnorm import (
    FusedQKVRMSNormKernel,
)
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    SavePartialStatesKernel,
)
from vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe import (
    _PREPARE_MEGAMOE_INPUTS_KERNEL,
)
from vllm.v1.attention.backends.mla.sparse_swa import ComputePrefillMetadataKernel
from vllm.v1.attention.ops.common import CorrectAttnCPOutKernel
from vllm.v1.worker.block_table import ComputeSlotMappingKernel

_HAS_CUTEDSL = has_cutedsl()
requires_cutedsl = pytest.mark.skipif(
    not _HAS_CUTEDSL,
    reason="CuTeDSL is not installed",
)

if _HAS_CUTEDSL:
    from cutlass import BFloat16, Float32

    from vllm.model_executor.kernels.attention.dsa.dcp_indexer_cutedsl import (
        StableTopKFromGatheredCandidatesKernel,
    )
    from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import LLBf16Gemm
    from vllm.model_executor.layers.fused_moe.router.bf16x3_router_gemm_cutedsl import (
        BF16x3RouterGemmKernel,
    )
    from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
        DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL,
        DequantGatherKCacheKernel,
    )
    from vllm.models.deepseek_v4.nvidia.ops.fused_indexer_q_cutedsl import (
        IndexerQFp8Kernel,
        IndexerQMxFp4Kernel,
    )
    from vllm.models.deepseek_v4.nvidia.ops.sparse_attn_compress_cutedsl import (
        SparseAttnCompressC128Block8Kernel,
        SparseAttnCompressNormRopeStoreC4Kernel,
        SparseAttnCompressNormRopeStoreFullC4Kernel,
        SparseAttnNormRopeStoreFullKernel,
        SparseAttnNormRopeStoreKernel,
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
        ),
        _PREPARE_MEGAMOE_INPUTS_KERNEL.CompileKey(
            hidden_size=256,
            top_k=6,
            block_topk=8,
            has_padding=True,
        ),
    ]


def test_deepseek_v4_c128a_topk_metadata_warmup_keys() -> None:
    from vllm.models.deepseek_v4.sparse_mla import (
        _BUILD_C128A_TOPK_METADATA_KERNEL,
    )

    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            max_model_len=65536,
            hf_config=SimpleNamespace(model_type="deepseek_v4"),
        )
    )

    assert _BUILD_C128A_TOPK_METADATA_KERNEL.get_warmup_keys(vllm_config) == [
        _BUILD_C128A_TOPK_METADATA_KERNEL.CompileKey(
            compress_ratio=128,
            max_compressed_tokens=512,
            block_size=2,
            triton_block_size=1024,
        )
    ]


@pytest.mark.parametrize(
    ("kv_cache_block_size", "blocks_per_kv_block", "block_size", "block_size_rep"),
    [
        (256, 1, 256, 16),
        (256, 4, 64, 16),
        (64, 1, 64, 16),
        (8, 1, 8, 2),
        (4, 1, 4, 2),
    ],
)
def test_compute_slot_mapping_warmup_matches_runtime_specializations(
    kv_cache_block_size: int,
    blocks_per_kv_block: int,
    block_size: int,
    block_size_rep: int,
) -> None:
    kernel = ComputeSlotMappingKernel()
    kwargs = dict(
        kv_cache_block_size=kv_cache_block_size,
        blocks_per_kv_block=blocks_per_kv_block,
        total_cp_world_size=1,
        total_cp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_table_stride=32768,
        block_size=block_size,
    )
    expected = kernel.CompileKey(
        kv_cache_block_size=kv_cache_block_size,
        blocks_per_kv_block=blocks_per_kv_block,
        total_cp_world_size=1,
        total_cp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_table_stride=16,
        block_size=block_size_rep,
        pad_id=-1,
        triton_block_size=1024,
    )

    assert kernel.dispatch(**kwargs) == expected
    assert kernel.get_warmup_keys(**kwargs) == [expected]


def test_correct_attn_cp_out_warmup_uses_dsv4_head_dim() -> None:
    hf_config = SimpleNamespace(
        compress_ratios=(1, 4, 128),
        head_dim=512,
        num_attention_heads=128,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=hf_config,
            hf_text_config=hf_config,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            tensor_parallel_size=4,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=1),
    )

    warmup_keys = CorrectAttnCPOutKernel().get_warmup_keys(vllm_config)

    assert len(warmup_keys) == 4
    assert {key.head_dim for key in warmup_keys} == {512}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                num_tokens=64,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
            ),
            (2048, 2, 128, 1024, 4, 1, False, 1),
        ),
        (
            dict(
                num_tokens=1024,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
            ),
            (2048, 2, 128, 512, 12, 1, True, 2),
        ),
        (
            dict(
                num_tokens=64,
                hc_hidden_size=4096,
                hidden_size=2048,
                hc_mult=2,
                n_out=128,
                n_thr=256,
                tile_n=8,
                n_splits=4,
            ),
            (2048, 2, 128, 256, 8, 4, False, 1),
        ),
    ],
)
def test_hc_prenorm_gemm_dispatch_matches_legacy_runtime_config(
    kwargs: dict[str, Any],
    expected: tuple[int, int, int, int, int, int, bool, int],
) -> None:
    kernel = HcPrenormGemmTileLangKernel()

    assert kernel.dispatch(**kwargs) == kernel.CompileKey(*expected)


@pytest.mark.parametrize(
    ("is_broadcast", "use_norm_weight", "expected_use_norm", "expected_eps"),
    [
        (False, False, False, 0.0),
        (False, True, True, 1.0e-5),
        (True, False, True, 2.0e-5),
    ],
)
def test_mhc_pre_big_fuse_dispatch_matches_legacy_runtime_config(
    is_broadcast: bool,
    use_norm_weight: bool,
    expected_use_norm: bool,
    expected_eps: float,
) -> None:
    kernel = MhcPreBigFuseTileLangKernel()

    assert kernel.dispatch(
        hidden_size=4096,
        hc_mult=4,
        n_splits=2,
        is_broadcast=is_broadcast,
        use_norm_weight=use_norm_weight,
        rms_eps=1.0e-6,
        hc_pre_eps=2.0e-6,
        hc_sinkhorn_eps=3.0e-6,
        hc_post_mult_value=0.5,
        sinkhorn_repeat=3,
        norm_eps=1.0e-5,
        broadcast_norm_eps=2.0e-5,
    ) == kernel.CompileKey(
        hidden_size=4096,
        hc_mult=4,
        n_splits=2,
        use_norm_weight=expected_use_norm,
        is_broadcast=is_broadcast,
        rms_eps=1.0e-6,
        hc_pre_eps=2.0e-6,
        hc_sinkhorn_eps=3.0e-6,
        hc_post_mult_value=0.5,
        sinkhorn_repeat=3,
        norm_eps=expected_eps,
    )


@pytest.mark.parametrize(
    ("num_tokens", "hidden_size", "expected_n_splits", "expected_tile_n"),
    [(4, 4096, 8, 2), (4, 8192, 4, 2), (8, 4096, 4, 3)],
)
def test_mhc_fused_dispatch_matches_legacy_runtime_config(
    num_tokens: int,
    hidden_size: int,
    expected_n_splits: int,
    expected_tile_n: int,
) -> None:
    kernel = MhcFusedTileLangKernel()

    assert kernel.dispatch(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        hc_mult=4,
    ) == kernel.CompileKey(
        hidden_size=hidden_size,
        hc_mult=4,
        n_splits=expected_n_splits,
        tile_n=expected_tile_n,
    )


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


@pytest.mark.parametrize(
    ("split_k", "expected_bn", "expected_bm", "expected_bs"),
    [(1, 16, 32, 1), (2, 16, 32, 2), (8, 1, 256, 8), (64, 1, 32, 64)],
)
def test_bf16x3_splitk_reduce_dispatch_matches_legacy_config(
    split_k: int,
    expected_bn: int,
    expected_bm: int,
    expected_bs: int,
) -> None:
    kernel = BF16x3SplitKReduceKernel()

    assert kernel.dispatch(M=256, split_k=split_k, USE_PDL=True) == kernel.CompileKey(
        m=256,
        bn=expected_bn,
        bm=expected_bm,
        bs=expected_bs,
        use_pdl=True,
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


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                use_fp4_cache=False,
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=4,
                cache_block_size=64,
                cache_alignment=1,
                runtime_state_width=1024,
                runtime_quant_block=64,
                runtime_token_stride=576,
                runtime_scale_dim=8,
                runtime_kv_block_stride=9216,
            ),
            (False, 512, 512, 1024, 4, True, 64, 448.0, 64, 576, 8, 64, 16, 9216),
        ),
        (
            dict(
                use_fp4_cache=True,
                head_dim=512,
                rope_head_dim=64,
                compress_ratio=128,
                cache_block_size=64,
                cache_alignment=1,
                runtime_state_width=512,
                runtime_quant_block=32,
                runtime_token_stride=256,
                runtime_scale_dim=16,
                runtime_kv_block_stride=576,
            ),
            (True, 512, 512, 512, 128, False, 64, 448.0, 32, 256, 16, 64, 1, 576),
        ),
    ],
)
def test_fused_compress_quant_dispatch_matches_legacy_runtime_meta(
    kwargs: dict[str, Any],
    expected: tuple[Any, ...],
) -> None:
    kernel = FusedKVCompressNormRopeInsertIndexerTritonKernel()

    assert kernel.dispatch(**kwargs) == kernel.CompileKey(*expected)


def test_fused_mtp_input_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = FusedMTPInputRMSNormKernel()

    assert kernel.dispatch(hidden=7168, hc_mult=4, eps=1.0e-6) == kernel.CompileKey(
        hidden=7168,
        hc_mult=4,
        block_size=8192,
        eps=1.0e-6,
    )


def test_mtp_shared_head_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = MTPSharedHeadRMSNormKernel()

    assert kernel.dispatch(hidden=7168, eps=1.0e-6) == kernel.CompileKey(
        hidden=7168,
        block_size=8192,
        eps=1.0e-6,
    )


def test_fused_qkv_rmsnorm_dispatch_matches_legacy_meta() -> None:
    kernel = FusedQKVRMSNormKernel()

    assert kernel.dispatch(
        q_size=1536,
        kv_size=512,
        q_in_stride=2048,
        q_out_stride=1536,
        kv_in_stride=1024,
        kv_out_stride=512,
        eps=1.0e-6,
    ) == kernel.CompileKey(
        q_size=1536,
        kv_size=512,
        block_size=2048,
        q_in_stride=2048,
        q_out_stride=1536,
        kv_in_stride=1024,
        kv_out_stride=512,
        eps=1.0e-6,
    )


def test_fused_inv_rope_warmup_uses_runtime_stride_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = FusedInvRopeFP8QuantKernel()
    warmup_kwargs: dict[str, Any] = {}

    def capture_warmup(*args: Any, **kwargs: Any) -> None:
        warmup_kwargs.update(kwargs)

    monkeypatch.setattr(kernel.kernel, "warmup", capture_warmup)
    kernel.compile(
        kernel.CompileKey(
            heads_per_group=16,
            fp8_max=448.0,
            quant_group_size=128,
            chunks_per_head=4,
            rope_start=64,
            half_rope=32,
            tma_aligned_scales=True,
            use_gdc=True,
        )
    )

    stride_names = (
        "o_stride_token",
        "o_stride_head",
        "cache_stride_pos",
        "fp8_stride_group",
        "fp8_stride_token",
        "scale_stride_group",
        "scale_stride_k",
    )
    assert {warmup_kwargs[name] for name in stride_names} == {16}


def test_save_partial_states_dispatch_matches_legacy_meta() -> None:
    kernel = SavePartialStatesKernel()

    assert kernel.dispatch(
        head_size=512,
        state_width=1024,
        compress_ratio=4,
        kv_stride=512,
        score_stride=8,
        ape_stride=64,
        state_cache_stride0=32768,
        state_cache_stride1=2048,
        block_size=16,
        launch_pdl=True,
    ) == kernel.CompileKey(
        head_size=512,
        triton_block_size=512,
        state_width=1024,
        compress_ratio=4,
        kv_stride=512,
        score_stride=8,
        ape_stride=64,
        state_cache_stride0=32768,
        state_cache_stride1=2048,
        block_size=16,
        launch_pdl=True,
    )


def test_save_partial_states_warmup_filters_zipped_compression_cases() -> None:
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                head_dim=512,
                compress_ratios=(128,),
            )
        )
    )

    warmup_keys = SavePartialStatesKernel().get_warmup_keys(vllm_config)

    assert len(warmup_keys) == 1
    assert warmup_keys[0].compress_ratio == 128


@pytest.mark.parametrize(
    ("num_prefills", "expected_block_size"),
    [(1, 1), (3, 4), (8, 8), (9, 16)],
)
def test_compute_prefill_metadata_dispatch_matches_legacy_meta(
    num_prefills: int,
    expected_block_size: int,
) -> None:
    kernel = ComputePrefillMetadataKernel()

    assert kernel.dispatch(num_prefills=num_prefills) == kernel.CompileKey(
        block_size=expected_block_size
    )


@requires_cutedsl
def test_stable_topk_dispatch_matches_legacy_compile_args() -> None:
    kernel = StableTopKFromGatheredCandidatesKernel()

    assert kernel.dispatch(topk=512, num_candidates=2048) == kernel.CompileKey(
        topk=512,
        num_candidates=2048,
    )


@requires_cutedsl
@pytest.mark.parametrize(
    ("M", "K", "N", "expected"),
    [
        (4, 7168, 256, ("dotprod", 4, 7168, 128, 0, 0)),
        (6, 7168, 256, ("dotprod", 6, 7168, 128, 0, 0)),
        (7, 7168, 256, ("splitk", 0, 0, 0, 6, 4)),
        (5, 7168, 384, ("splitk", 0, 0, 0, 4, 4)),
        (8, 7168, 384, ("splitk", 0, 0, 0, 5, 4)),
        (16, 1024, 256, ("dotprod", 16, 1024, 128, 0, 0)),
    ],
)
def test_ll_bf16_dispatch_matches_legacy_config(
    M: int,
    K: int,
    N: int,
    expected: tuple[str, int, int, int, int, int],
) -> None:
    kernel = LLBf16Gemm()
    backend, m, k, bs, split_k, num_stages = expected

    assert kernel.dispatch(M=M, K=K, N=N) == kernel.CompileKey(
        backend=backend,
        m=m,
        k=k,
        bs=bs,
        split_k=split_k,
        num_stages=num_stages,
    )


@requires_cutedsl
@pytest.mark.parametrize(
    ("num_tokens", "expected_bn"),
    [(1, 8), (8, 8), (9, 16), (128, 128), (129, 128)],
)
def test_bf16x3_dispatch_matches_legacy_bn(
    num_tokens: int,
    expected_bn: int,
) -> None:
    kernel = BF16x3RouterGemmKernel()

    assert kernel.dispatch(num_tokens=num_tokens, K=6144) == kernel.CompileKey(
        bn=expected_bn,
        k=6144,
    )


@requires_cutedsl
@pytest.mark.parametrize("has_gather_lens", [False, True])
def test_dequant_gather_dispatch_matches_legacy_compile_args(
    has_gather_lens: bool,
) -> None:
    kernel = DEQUANT_GATHER_K_CACHE_CUTEDSL_KERNEL
    assert isinstance(kernel, DequantGatherKCacheKernel)

    assert kernel.dispatch(
        block_size=64,
        has_gather_lens=has_gather_lens,
    ) == kernel.CompileKey(
        block_size=64,
        has_gather_lens=has_gather_lens,
    )


@requires_cutedsl
@pytest.mark.parametrize("kernel_name", ["mx_fp4", "fp8"])
@pytest.mark.parametrize("coarsen", [1, 4])
def test_indexer_q_dispatch_matches_legacy_compile_args(
    kernel_name: str,
    coarsen: int,
) -> None:
    kernel_cls = {
        "mx_fp4": IndexerQMxFp4Kernel,
        "fp8": IndexerQFp8Kernel,
    }[kernel_name]
    kernel = kernel_cls()

    assert kernel.dispatch(
        head_dim=128,
        rope_dim=64,
        num_heads=64,
        cos_sin_dtype=Float32,
        coarsen=coarsen,
    ) == kernel.CompileKey(
        head_dim=128,
        rope_dim=64,
        num_heads=64,
        cos_sin_dtype=Float32,
        coarsen=coarsen,
    )


@requires_cutedsl
def test_sparse_c4_dispatch_matches_legacy_constructor_args() -> None:
    kernel = SparseAttnCompressNormRopeStoreC4Kernel()

    assert kernel.dispatch(
        compress_ratio=4,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
    ) == kernel.CompileKey(
        head_size=512,
        state_width=1024,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        compress_ratio=4,
        overlap=True,
        norm_weight_dtype=Float32,
    )


@requires_cutedsl
@pytest.mark.parametrize("store_full_fp8", [False, True])
def test_sparse_full_c4_dispatch_matches_legacy_constructor_args(
    store_full_fp8: bool,
) -> None:
    kernel = SparseAttnCompressNormRopeStoreFullC4Kernel()

    assert kernel.dispatch(
        compress_ratio=4,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
    ) == kernel.CompileKey(
        head_size=512,
        state_width=1024,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        compress_ratio=4,
        overlap=True,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
    )


@requires_cutedsl
def test_sparse_c128_compress_dispatch_matches_legacy_constructor_args() -> None:
    kernel = SparseAttnCompressC128Block8Kernel()

    assert kernel.dispatch(head_size=512, state_width=512) == kernel.CompileKey(
        head_size=512,
        state_width=512,
    )


@requires_cutedsl
@pytest.mark.parametrize(
    (
        "cache_block_size",
        "runtime_kv_block_stride",
        "kv_cache_block_size",
        "kv_block_stride",
    ),
    [(64, None, 1, 1152), (256, None, 2, 1728), (256, 39168, 2, 39168)],
)
def test_sparse_c128_store_dispatch_matches_legacy_constructor_args(
    cache_block_size: int,
    runtime_kv_block_stride: int | None,
    kv_cache_block_size: int,
    kv_block_stride: int,
) -> None:
    kernel = SparseAttnNormRopeStoreKernel()

    assert kernel.dispatch(
        compress_ratio=128,
        cache_block_size=cache_block_size,
        cache_alignment=576,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
        runtime_kv_block_stride=runtime_kv_block_stride,
    ) == kernel.CompileKey(
        head_size=512,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        token_stride=576,
        scale_dim=8,
        kv_block_stride=kv_block_stride,
        compress_ratio=128,
        norm_weight_dtype=Float32,
        kv_cache_block_size=kv_cache_block_size,
    )


@requires_cutedsl
def test_sparse_c128_store_warmup_uses_bound_packed_cache_stride() -> None:
    kernel = SparseAttnNormRopeStoreKernel()
    packed_stride = 39168
    storage = torch.empty(packed_stride + 1168, dtype=torch.uint8)
    kv_cache = torch.as_strided(
        storage,
        size=(2, 2, 584),
        stride=(packed_stride, 584, 1),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_config=SimpleNamespace(
                head_dim=512,
                qk_rope_head_dim=64,
            ),
        ),
        cache_config=SimpleNamespace(
            block_size=256,
            cache_dtype="fp8_ds_mla",
        ),
        compilation_config=SimpleNamespace(
            static_forward_context={
                "model.layers.0.self_attn": SimpleNamespace(kv_cache=kv_cache)
            }
        ),
    )

    assert kernel.get_warmup_keys(
        vllm_config,
        k_cache_prefix="model.layers.0.self_attn",
        compress_ratio=128,
    ) == [
        kernel.CompileKey(
            head_size=512,
            rope_head_dim=64,
            fp8_max=448.0,
            quant_block=64,
            token_stride=576,
            scale_dim=8,
            kv_block_stride=packed_stride,
            compress_ratio=128,
            norm_weight_dtype=BFloat16,
            kv_cache_block_size=2,
        )
    ]


@requires_cutedsl
@pytest.mark.parametrize("store_full_fp8", [False, True])
def test_sparse_full_c128_store_dispatch_matches_legacy_constructor_args(
    store_full_fp8: bool,
) -> None:
    kernel = SparseAttnNormRopeStoreFullKernel()

    assert kernel.dispatch(
        compress_ratio=128,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
        head_size=512,
        rope_head_dim=64,
    ) == kernel.CompileKey(
        head_size=512,
        rope_head_dim=64,
        fp8_max=448.0,
        quant_block=64,
        compress_ratio=128,
        store_full_fp8=store_full_fp8,
        norm_weight_dtype=Float32,
    )
