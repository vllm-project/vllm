# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the TRITON_MLA_SPARSE backend and SM80+ Triton MQA logits fallback."""

import ast
import inspect
import textwrap
from types import SimpleNamespace

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    DeviceConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.parallel import ParallelConfig
from vllm.config.scheduler import SchedulerConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers import sparse_attn_indexer as sai_mod
from vllm.models.deepseek_v32.common import kernels as dsv32_kernels
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import AttentionCGSupport, AttentionType
from vllm.v1.attention.backends.mla import indexer as indexer_mod
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
    TritonMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.ops import triton_mla_sparse as mla_sparse_ops
from vllm.v1.attention.ops.common import e4m3_uint8_to_f32, f32_to_e4m3_uint8
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_mla_sparse import triton_mla_sparse_fwd
from vllm.v1.attention.selector import AttentionSelectorConfig

pytestmark = pytest.mark.skip_global_cleanup

SM75 = DeviceCapability(major=7, minor=5)
SM80 = DeviceCapability(major=8, minor=0)
SM86 = DeviceCapability(major=8, minor=6)
SM89 = DeviceCapability(major=8, minor=9)
SM90 = DeviceCapability(major=9, minor=0)
SM100 = DeviceCapability(major=10, minor=0)
SM120 = DeviceCapability(major=12, minor=0)


# ---------------------------------------------------------------------------
# Reference math helpers (for GPU kernel parity tests)
# ---------------------------------------------------------------------------


def _ref_fp8_mqa_logits(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    cu_ks: torch.Tensor,
    cu_ke: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    q_f32 = q.to(torch.float32)
    k_f32 = k_quant.to(torch.float32)
    num_rows = q.shape[0]
    out = torch.full(
        (num_rows, max_seqlen_k), float("-inf"), dtype=torch.float32, device=q.device
    )
    for i in range(num_rows):
        ks = int(cu_ks[i].item())
        ke = int(cu_ke[i].item())
        length = max(0, min(ke - ks, max_seqlen_k))
        if length == 0:
            continue
        # [H, D] @ [D, L] -> [H, L]
        dots = torch.matmul(q_f32[i], k_f32[ks : ks + length].T)
        scores = torch.relu(dots) * weights[i].unsqueeze(1)
        out[i, :length] = scores.sum(dim=0) * k_scale[ks : ks + length]
    return out


def _ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n, num_heads, head_dim = q.shape
    block_size = kv_cache.shape[1]
    block_stride = kv_cache.stride(0)
    flat_u8 = kv_cache.untyped_storage()
    storage_u8 = torch.tensor([], dtype=torch.uint8, device=kv_cache.device).set_(
        flat_u8, 0, (len(flat_u8),)
    )
    base_offset = kv_cache.storage_offset()

    q_f32 = q.reshape(batch_size * next_n, num_heads, head_dim).to(torch.float32)
    out = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        dtype=torch.float32,
        device=q.device,
    )
    for b in range(batch_size):
        for n in range(next_n):
            row = b * next_n + n
            if context_lens.ndim == 2:
                seq_len = int(context_lens[b, n].item())
            else:
                seq_len = max(0, int(context_lens[b].item()) - (next_n - 1 - n))
            seq_len = min(max(seq_len, 0), max_model_len)
            for pos in range(seq_len):
                blk = int(block_tables[b, pos // block_size].item())
                off = pos % block_size
                blk_base = base_offset + blk * block_stride
                k_bytes = storage_u8[
                    blk_base + off * head_dim : blk_base + (off + 1) * head_dim
                ]
                k_val = k_bytes.view(torch.float8_e4m3fn).to(torch.float32)
                s_start = blk_base + block_size * head_dim + off * 4
                scale_val = storage_u8[s_start : s_start + 4].view(torch.float32)[0]
                dots = torch.matmul(q_f32[row], k_val)
                out[row, pos] = (torch.relu(dots) * weights[row]).sum() * scale_val
    return out


def _ref_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    kv_lora_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, _ = q.shape
    out = torch.zeros(
        (num_tokens, num_heads, kv_lora_rank), dtype=torch.float32, device=q.device
    )
    lse = torch.full(
        (num_tokens, num_heads), float("-inf"), dtype=torch.float32, device=q.device
    )
    q_f32 = q.to(torch.float32)
    kv_f32 = kv.to(torch.float32)
    for t in range(num_tokens):
        row_idx = indices[t]
        valid = (row_idx >= 0) & (row_idx < kv.shape[0])
        if not valid.any():
            continue
        gathered = kv_f32[row_idx[valid]]  # [K, D]
        logits = torch.matmul(q_f32[t], gathered.T) * sm_scale  # [H, K]
        row_lse = torch.logsumexp(logits, dim=-1)  # [H]
        probs = torch.softmax(logits, dim=-1)
        out[t] = torch.matmul(probs, gathered[:, :kv_lora_rank])
        lse[t] = row_lse
    return out.to(q.dtype), lse


# ---------------------------------------------------------------------------
# Backend registration, validation, and CUDA selection (CPU-runnable)
# ---------------------------------------------------------------------------


def test_triton_mla_sparse_registered_and_metadata_properties():
    assert AttentionBackendEnum.TRITON_MLA_SPARSE.get_class() is TritonMLASparseBackend
    assert TritonMLASparseBackend.get_name() == "TRITON_MLA_SPARSE"
    assert TritonMLASparseBackend.is_mla() is True
    assert TritonMLASparseBackend.is_sparse() is True
    assert TritonMLASparseBackend.supports_batch_invariance() is True
    assert TritonMLASparseBackend.get_supported_head_sizes() == [512, 576]
    assert (
        TritonMLASparseMetadataBuilder._cudagraph_support
        is AttentionCGSupport.UNIFORM_BATCH
    )


@pytest.mark.parametrize(
    "capability,head_size,dtype,kv_cache_dtype,use_sparse,expected_valid",
    [
        (SM80, 576, torch.bfloat16, "auto", True, True),
        (SM80, 576, torch.float16, "float16", True, True),
        (SM80, 512, torch.bfloat16, "bfloat16", True, True),
        (SM86, 576, torch.bfloat16, "auto", True, True),
        (SM89, 576, torch.bfloat16, "auto", True, True),
        (SM90, 576, torch.bfloat16, "auto", True, True),
        (SM100, 576, torch.bfloat16, "auto", True, True),
        (SM120, 576, torch.bfloat16, "auto", True, True),
        (SM75, 576, torch.float16, "auto", True, False),
        (SM80, 576, torch.bfloat16, "fp8", True, False),
        (SM80, 576, torch.bfloat16, "fp8_ds_mla", True, False),
        (SM80, 576, torch.bfloat16, "auto", False, False),
        (SM80, 128, torch.bfloat16, "auto", True, False),
    ],
)
def test_triton_mla_sparse_validate_configuration(
    capability, head_size, dtype, kv_cache_dtype, use_sparse, expected_valid
):
    cfg = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=64,
        use_mla=True,
        has_sink=False,
        use_sparse=use_sparse,
        use_mm_prefix=False,
        attn_type=AttentionType.DECODER,
    )
    reasons = TritonMLASparseBackend.validate_configuration(
        device_capability=capability,
        **cfg._asdict(),
    )
    assert (len(reasons) == 0) is expected_valid, reasons


def test_triton_mla_sparse_rejects_dcp_and_missing_index_topk(monkeypatch):
    def _validate(vllm_config):
        monkeypatch.setattr(
            "vllm.config.get_current_vllm_config_or_none", lambda: vllm_config
        )
        cfg = AttentionSelectorConfig(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="auto",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            attn_type=AttentionType.DECODER,
        )
        return TritonMLASparseBackend.validate_configuration(
            device_capability=SM80,
            **cfg._asdict(),
        )

    dsa_hf = SimpleNamespace(index_topk=2048, kv_lora_rank=512, qk_rope_head_dim=64)
    dcp_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=dsa_hf),
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
    )
    assert any("DCP" in r for r in _validate(dcp_cfg))

    non_dsa_cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(kv_lora_rank=512, qk_rope_head_dim=64)
        ),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert any("index_topk" in r for r in _validate(non_dsa_cfg))

    valid_cfg = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=dsa_hf),
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    assert _validate(valid_cfg) == []


def _make_dsa_vllm_config() -> VllmConfig:
    dsa_hf = SimpleNamespace(
        index_topk=2048,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
    )
    vllm_cfg = VllmConfig(
        parallel_config=ParallelConfig(decode_context_parallel_size=1),
        device_config=DeviceConfig(device="cpu"),
    )
    vllm_cfg.model_config = SimpleNamespace(  # type: ignore[assignment]
        hf_text_config=dsa_hf
    )
    return vllm_cfg


def test_cuda_selector_selects_triton_mla_sparse_on_sm80(monkeypatch):
    """On SM80, DSA sparse MLA must resolve to TRITON_MLA_SPARSE (issue #66960)."""
    try:
        from vllm.platforms.cuda import CudaPlatform, _get_backend_priorities
    except ImportError as e:
        pytest.skip(f"CUDA platform unavailable on this host: {e}")

    monkeypatch.setattr(
        CudaPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: SM80),
    )

    selector_cfg = AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=None,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        attn_type=AttentionType.DECODER,
    )
    # Candidate backends probed during selection (e.g. FLASHINFER_MLA) read
    # get_current_vllm_config(), so a VllmConfig context must be active.
    with set_current_vllm_config(_make_dsa_vllm_config()):
        backend_path = CudaPlatform.get_attn_backend_cls(
            selected_backend=None,
            attn_selector_config=selector_cfg,
            num_heads=64,
        )
    assert backend_path.endswith(".TritonMLASparseBackend")

    # Hopper/Blackwell high-priority sparse backends remain ahead of TRITON_MLA_SPARSE.
    for cap in (SM90, SM100, SM120):
        order = _get_backend_priorities(
            use_mla=True,
            device_capability=cap,
            num_heads=64,
            kv_cache_dtype="auto",
            head_size=576,
        )
        assert order[-1] is AttentionBackendEnum.TRITON_MLA_SPARSE


def test_sm80_candidate_validation_uses_vllm_config_context(monkeypatch):
    """Regression test: MLA candidate validation that calls
    `get_current_vllm_config()` requires `set_current_vllm_config(...)`, and
    under that context only `TRITON_MLA_SPARSE` validates on SM80.
    """
    from vllm.v1.attention.backends.mla import (
        flashinfer_mla_sparse_sm90 as fi_sm90_mod,
    )
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
    from vllm.v1.attention.backends.mla.triton_mla import TritonMLABackend

    # Force FlashInferMLASparseSM90Backend.supports_combination past the
    # FlashInfer version check so it exercises get_current_vllm_config().
    monkeypatch.setattr(fi_sm90_mod, "has_flashinfer_sm90_nope_mla", lambda: True)

    selector_cfg = AttentionSelectorConfig(
        head_size=576,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=None,
        use_mla=True,
        has_sink=False,
        use_sparse=True,
        attn_type=AttentionType.DECODER,
    )

    with pytest.raises(AssertionError, match="Current vLLM config is not set"):
        fi_sm90_mod.FlashInferMLASparseSM90Backend.validate_configuration(
            device_capability=SM80,
            **selector_cfg._asdict(),
        )

    with set_current_vllm_config(_make_dsa_vllm_config()):
        for rejected_cls in (
            TritonMLABackend,
            FlashMLASparseBackend,
            fi_sm90_mod.FlashInferMLASparseSM90Backend,
        ):
            reasons = rejected_cls.validate_configuration(
                device_capability=SM80,
                **selector_cfg._asdict(),
            )
            assert reasons, f"{rejected_cls.__name__} should be rejected on SM80"

        assert (
            TritonMLASparseBackend.validate_configuration(
                device_capability=SM80,
                **selector_cfg._asdict(),
            )
            == []
        )


def test_fused_q_kernel_guards_fp8_dtypes_under_quantize_mqa():
    """Regression test: Triton compiles code after a constexpr early `return`,
    so every `tl.float8e4nv`/`tl.float8e4b8` reference in `_fused_q_kernel`
    must be lexically nested inside the positive branch of `if QUANTIZE_MQA:`
    to avoid SM80 compilation failures when `QUANTIZE_MQA=False`.
    """
    kernel_fn = getattr(
        dsv32_kernels._fused_q_kernel, "fn", dsv32_kernels._fused_q_kernel
    )
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel_fn)))
    fp8_sites: list[tuple[int, str, bool]] = []

    def _walk(node: ast.AST, inside_quantize_mqa: bool) -> None:
        if isinstance(node, ast.If):
            is_q_mqa = isinstance(node.test, ast.Name) and node.test.id == (
                "QUANTIZE_MQA"
            )
            for stmt in node.body:
                _walk(stmt, inside_quantize_mqa or is_q_mqa)
            for stmt in node.orelse:
                _walk(stmt, inside_quantize_mqa)
            return
        if isinstance(node, ast.Attribute) and node.attr in (
            "float8e4nv",
            "float8e4b8",
        ):
            fp8_sites.append((node.lineno, node.attr, inside_quantize_mqa))
        for child in ast.iter_child_nodes(node):
            _walk(child, inside_quantize_mqa)

    _walk(tree, False)
    assert fp8_sites, "Expected FP8 dtype references in _fused_q_kernel"
    unguarded = [(ln, attr) for ln, attr, guarded in fp8_sites if not guarded]
    assert not unguarded, (
        f"Found FP8 dtype references not nested under `if QUANTIZE_MQA:`: {unguarded}"
    )


def test_triton_mla_sparse_kernels_pass_loge2_as_constexpr():
    """Regression test: `@triton.jit` kernels must receive `LOGE2` as a
    `tl.constexpr` parameter rather than reading a Python module global.
    """
    for kernel in (
        mla_sparse_ops._triton_mla_sparse_fwd_kernel,
        mla_sparse_ops._triton_mla_sparse_reduce_kernel,
    ):
        fn = getattr(kernel, "fn", kernel)
        sig = inspect.signature(fn)
        assert "LOGE2" in sig.parameters
        assert sig.parameters["LOGE2"].annotation is tl.constexpr

    for kernel in (
        mla_sparse_ops._triton_mla_sparse_fwd_kernel,
        mla_sparse_ops._triton_mla_sparse_splitk_kernel,
        mla_sparse_ops._triton_mla_sparse_reduce_kernel,
    ):
        fn = getattr(kernel, "fn", kernel)
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        params = {a.arg for a in func_node.args.args + func_node.args.kwonlyargs}
        assigned = set(params)
        body_loaded: set[str] = set()
        for stmt in func_node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name):
                    if isinstance(sub.ctx, ast.Store):
                        assigned.add(sub.id)
                    elif isinstance(sub.ctx, ast.Load):
                        body_loaded.add(sub.id)
        allowed = {"range", "float", "int", "min", "max", "abs", "True", "False", "tl"}
        unbound = sorted(body_loaded - assigned - allowed)
        assert not unbound, f"{fn.__name__} references unbound globals: {unbound}"


# ---------------------------------------------------------------------------
# Indexer SM80 / non-DeepGEMM guard and dispatch tests (CPU-runnable)
# ---------------------------------------------------------------------------


def test_indexer_skips_deepgemm_metadata_when_deepgemm_unsupported(monkeypatch):
    """Even if `deep_gemm` is installed on SM80 (`has_deep_gemm()==True`),
    `update_draft_decode_metadata` must not invoke DeepGEMM when
    `is_deep_gemm_supported()` is False.
    """
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(indexer_mod, "has_deep_gemm", lambda: True)
    monkeypatch.setattr(indexer_mod, "is_deep_gemm_supported", lambda: False)

    def _boom(*args, **kwargs):
        raise AssertionError("get_paged_mqa_logits_metadata must not be called on SM80")

    monkeypatch.setattr(indexer_mod, "get_paged_mqa_logits_metadata", _boom)

    num_reqs = 2
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    decode_meta = indexer_mod.DeepSeekV32IndexerDecodeMetadata(
        block_table=torch.zeros_like(block_table),
        seq_lens=torch.zeros((num_reqs, 1), dtype=torch.int32),
        decode_lens=torch.zeros(num_reqs, dtype=torch.int32),
        requires_padding=False,
        schedule_metadata=torch.zeros((4, 2), dtype=torch.int32),
    )
    meta = indexer_mod.DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([16, 24], dtype=torch.int32),
        max_seq_len=24,
        slot_mapping=torch.zeros(num_reqs, dtype=torch.int64),
        num_decodes=num_reqs,
        num_decode_tokens=num_reqs,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode_meta,
        block_table=block_table,
    )
    builder = object.__new__(indexer_mod.DeepseekV32IndexerMetadataBuilder)
    builder.dcp_world_size = 1
    builder.compress_ratio = 1
    builder.num_sms = 108
    builder.arange_buffer = torch.arange(num_reqs + 1, dtype=torch.int32)
    builder.kv_cache_spec = SimpleNamespace(num_states=64)

    builder.update_draft_decode_metadata(meta)
    torch.testing.assert_close(decode_meta.seq_lens.flatten(), meta.seq_lens)
    torch.testing.assert_close(decode_meta.block_table, block_table)


def test_sparse_attn_indexer_init_allows_sm80_fp8_and_rejects_fp4(monkeypatch):
    """CUDA SM80 without DeepGEMM must allow FP8 indexer init and reject FP4."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: cap <= 80
    )
    monkeypatch.setattr(sai_mod, "is_deep_gemm_supported", lambda: False)

    fake_vllm_config = VllmConfig(
        cache_config=CacheConfig(),
        parallel_config=ParallelConfig(),
        scheduler_config=SchedulerConfig(max_model_len=256, is_encoder_decoder=False),
        attention_config=AttentionConfig(),
        device_config=DeviceConfig(device="cpu"),
    )
    fake_vllm_config.kernel_config.enable_jit_warmup = False

    topk_buf = torch.empty((4, 8), dtype=torch.int32)
    dummy_cache = SimpleNamespace(
        prefix="layer.0.indexer.k_cache", kv_cache=torch.empty(0)
    )

    with set_current_vllm_config(fake_vllm_config):
        # FP8 cache succeeds without DeepGEMM (uses Triton fallback).
        op = sai_mod.SparseAttnIndexer(
            dummy_cache,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=8,
            head_dim=128,
            max_model_len=256,
            max_total_seq_len=256,
            topk_indices_buffer=topk_buf,
            use_fp4_cache=False,
        )
        assert op.use_fp4_cache is False

        # FP4 cache still requires DeepGEMM.
        with pytest.raises(RuntimeError, match="FP4 cache requires DeepGEMM"):
            sai_mod.SparseAttnIndexer(
                dummy_cache,
                quant_block_size=128,
                scale_fmt="ue8m0",
                topk_tokens=8,
                head_dim=128,
                max_model_len=256,
                max_total_seq_len=256,
                topk_indices_buffer=topk_buf,
                use_fp4_cache=True,
            )


def test_sparse_attn_indexer_decode_dispatch_2d_seq_lens_and_max_model_len(
    monkeypatch,
):
    """Regression test for issue #48285:
    1) Triton paged MQA logits must receive 2D `(B, next_n)` `seq_lens`.
    2) Output width must use static `max_model_len`, NOT per-batch `max_seq_len`.
    """
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: cap <= 80
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda fam, device_id=0: fam == 80,
    )
    monkeypatch.setattr(sai_mod, "is_deep_gemm_supported", lambda: False)
    monkeypatch.setattr(sai_mod, "use_fused_mqa_topk", lambda **kwargs: False)

    captured: dict[str, object] = {}

    def _fake_paged_triton(
        q, kv_cache, weights, context_lens, block_tables, max_model_len, indices=None
    ):
        captured["q_shape"] = tuple(q.shape)
        captured["context_lens_shape"] = tuple(context_lens.shape)
        captured["max_model_len"] = max_model_len
        num_rows = q.shape[0] * q.shape[1]
        return (
            torch.arange(max_model_len, dtype=torch.float32)
            .unsqueeze(0)
            .expand(num_rows, -1)
            .contiguous()
        )

    monkeypatch.setattr(sai_mod, "fp8_paged_mqa_logits_triton", _fake_paged_triton)

    batch_size, next_n, num_heads, head_dim = 2, 2, 32, 128
    max_model_len = 256
    per_batch_max_seq_len = 12
    topk_tokens = 4
    prefix = "layer.0.indexer.k_cache"

    seq_lens_2d = torch.tensor([[9, 10], [11, 12]], dtype=torch.int32)
    decode_meta = indexer_mod.DeepSeekV32IndexerDecodeMetadata(
        block_table=torch.zeros((batch_size, 4), dtype=torch.int32),
        seq_lens=seq_lens_2d,
        decode_lens=torch.tensor([next_n, next_n], dtype=torch.int32),
        requires_padding=False,
        schedule_metadata=torch.zeros((1, 2), dtype=torch.int32),
    )
    meta = indexer_mod.DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([10, 12], dtype=torch.int32),
        max_seq_len=per_batch_max_seq_len,
        slot_mapping=torch.arange(batch_size * next_n, dtype=torch.int64),
        num_decodes=batch_size,
        num_decode_tokens=batch_size * next_n,
        num_prefills=0,
        num_prefill_tokens=0,
        decode=decode_meta,
    )

    num_tokens = batch_size * next_n
    hidden_states = torch.zeros((num_tokens, 16), dtype=torch.bfloat16)
    kv_cache = torch.zeros((4, 64, head_dim + 4), dtype=torch.uint8)
    q_quant = torch.zeros((num_tokens, num_heads, head_dim), dtype=torch.float8_e4m3fn)
    weights = torch.ones((num_tokens, num_heads), dtype=torch.float32)
    topk_buf = torch.full((num_tokens, topk_tokens), -1, dtype=torch.int32)

    with set_forward_context(
        {prefix: meta}, VllmConfig(device_config=DeviceConfig(device="cpu"))
    ):
        out = sai_mod.sparse_attn_indexer(
            hidden_states=hidden_states,
            k_cache_prefix=prefix,
            kv_cache=kv_cache,
            q_quant=q_quant,
            q_scale=None,
            k=None,
            weights=weights,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=topk_tokens,
            head_dim=head_dim,
            max_model_len=max_model_len,
            total_seq_lens=64,
            topk_indices_buffer=topk_buf,
            skip_k_cache_insert=True,
            use_pcp=False,
            pcp_shard_decode_requests=False,
            dense_mha_metadata_layer_name="",
            topk_backend="torch",
        )

    assert captured["q_shape"] == (batch_size, next_n, num_heads, head_dim)
    assert captured["context_lens_shape"] == (batch_size, next_n)
    assert captured["max_model_len"] == max_model_len
    # Torch topk selected the highest 4 indices in [0, seq_len) per row.
    assert out[0].tolist() == [8, 7, 6, 5]
    assert out[1].tolist() == [9, 8, 7, 6]
    assert out[2].tolist() == [10, 9, 8, 7]
    assert out[3].tolist() == [11, 10, 9, 8]


def test_sparse_attn_indexer_prefill_dispatch_uses_triton_compressed_logits(
    monkeypatch,
):
    """Non-DeepGEMM CUDA prefill routes to `fp8_mqa_logits_triton` with
    compressed `row_ends = cu_seqlen_ke - cu_seqlen_ks` and `row_starts=None`."""
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(
        current_platform, "has_device_capability", lambda cap, device_id=0: cap <= 80
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability_family",
        lambda fam, device_id=0: fam == 80,
    )
    monkeypatch.setattr(sai_mod, "is_deep_gemm_supported", lambda: False)
    monkeypatch.setattr(sai_mod, "use_fused_mqa_topk", lambda **kwargs: False)

    captured: dict[str, object] = {}

    def _fake_prefill_triton(q, kv, weights, cu_ks, cu_ke, max_seqlen_k):
        captured["q_shape"] = tuple(q.shape)
        captured["k_shape"] = tuple(kv[0].shape)
        captured["k_scale_shape"] = tuple(kv[1].shape)
        captured["max_seqlen_k"] = max_seqlen_k
        return (
            torch.arange(max_seqlen_k, dtype=torch.float32)
            .unsqueeze(0)
            .expand(q.shape[0], -1)
            .contiguous()
        )

    monkeypatch.setattr(sai_mod, "fp8_mqa_logits_triton", _fake_prefill_triton)

    def _fake_top_k_per_row_prefill(
        logits, row_starts, row_ends, indices, num_rows, stride0, stride1, topk
    ):
        captured["row_starts"] = row_starts.tolist()
        captured["row_ends"] = row_ends.tolist()
        for r in range(num_rows):
            s = int(row_starts[r].item())
            e = int(row_ends[r].item())
            valid_len = max(0, e - s)
            k_take = min(valid_len, topk)
            if k_take > 0:
                vals = logits[r, s:e]
                top = vals.topk(k_take).indices.to(torch.int32)
                indices[r, :k_take] = top
            if k_take < topk:
                indices[r, k_take:] = -1

    monkeypatch.setattr(
        "vllm.model_executor.layers.indexer_topk.ops.top_k_per_row_prefill",
        _fake_top_k_per_row_prefill,
    )

    class _FakeWorkspace:
        def get_simultaneous(self, *specs):
            return [torch.zeros(shape, dtype=dtype) for shape, dtype in specs]

    monkeypatch.setattr(sai_mod, "current_workspace_manager", lambda: _FakeWorkspace())

    num_tokens, num_heads, head_dim = 3, 32, 128
    max_seqlen_k = 10
    topk_tokens = 4
    prefix = "layer.0.indexer.k_cache"

    chunk = indexer_mod.DeepseekV32IndexerPrefillChunkMetadata(
        block_table=torch.zeros((1, 2), dtype=torch.int32),
        cu_seqlen_ks=torch.tensor([2, 2, 2], dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([5, 8, 12], dtype=torch.int32),
        cu_seq_lens=torch.tensor([0, 12], dtype=torch.int32),
        token_to_seq=torch.zeros((12,), dtype=torch.int32),
        token_start=0,
        token_end=num_tokens,
        num_reqs=1,
        total_seq_lens=12,
        max_seqlen_k=max_seqlen_k,
        skip_kv_gather=True,
        local_cu_seq_lens=torch.tensor([0, 12], dtype=torch.int32),
        local_total_seq_lens=12,
        max_local_total_seq_lens=12,
    )
    meta = indexer_mod.DeepseekV32IndexerMetadata(
        seq_lens=torch.tensor([12], dtype=torch.int32),
        max_seq_len=12,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        num_decodes=0,
        num_decode_tokens=0,
        num_prefills=1,
        num_prefill_tokens=num_tokens,
        prefill=indexer_mod.DeepseekV32IndexerPrefillMetadata(
            chunks=[chunk], max_prefill_seq_len=12
        ),
    )

    hidden_states = torch.zeros((num_tokens, 16), dtype=torch.bfloat16)
    kv_cache = torch.zeros((2, 64, head_dim + 4), dtype=torch.uint8)
    q_quant = torch.zeros((num_tokens, num_heads, head_dim), dtype=torch.float8_e4m3fn)
    weights = torch.ones((num_tokens, num_heads), dtype=torch.float32)
    topk_buf = torch.full((num_tokens, topk_tokens), -1, dtype=torch.int32)

    with set_forward_context(
        {prefix: meta}, VllmConfig(device_config=DeviceConfig(device="cpu"))
    ):
        out = sai_mod.sparse_attn_indexer(
            hidden_states=hidden_states,
            k_cache_prefix=prefix,
            kv_cache=kv_cache,
            q_quant=q_quant,
            q_scale=None,
            k=None,
            weights=weights,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=topk_tokens,
            head_dim=head_dim,
            max_model_len=256,
            total_seq_lens=64,
            topk_indices_buffer=topk_buf,
            skip_k_cache_insert=True,
            use_pcp=False,
            pcp_shard_decode_requests=False,
            dense_mha_metadata_layer_name="",
            topk_backend="torch",
        )

    assert captured["q_shape"] == (num_tokens, num_heads, head_dim)
    assert captured["k_shape"] == (12, head_dim)
    assert captured["k_scale_shape"] == (12,)
    assert captured["max_seqlen_k"] == max_seqlen_k
    assert captured["row_starts"] == [0, 0, 0]
    assert captured["row_ends"] == [3, 6, 10]
    # Compressed lengths are [3, 6, 10]: row 0 has 3 valid entries + 1 sentinel (-1).
    assert out[0].tolist() == [2, 1, 0, -1]
    assert out[1].tolist() == [5, 4, 3, 2]
    assert out[2].tolist() == [9, 8, 7, 6]


# ---------------------------------------------------------------------------
# CUDA GPU kernel parity tests (skipped on non-CUDA / ROCm / CPU)
# ---------------------------------------------------------------------------


@triton.jit
def _test_e4m3_decode_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(in_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, e4m3_uint8_to_f32(vals), mask=mask)


@triton.jit
def _test_e4m3_encode_kernel(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    vals = tl.load(in_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, f32_to_e4m3_uint8(vals), mask=mask)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_e4m3_software_codecs_bit_exact_gpu():
    device = torch.device("cuda")
    codes = torch.arange(256, dtype=torch.uint8, device=device)
    dec = torch.empty(256, dtype=torch.float32, device=device)
    _test_e4m3_decode_kernel[(1,)](codes, dec, 256, BLOCK=256)

    ref_dec = codes.view(torch.float8_e4m3fn).to(torch.float32)
    finite = ~torch.isnan(ref_dec)
    assert torch.equal(dec[finite].view(torch.int32), ref_dec[finite].view(torch.int32))
    assert torch.isnan(dec[~finite]).all()

    pos = ref_dec[:127]
    mids = (pos[:-1] + pos[1:]) * 0.5
    specials = torch.tensor(
        [
            0.0,
            -0.0,
            448.0,
            -448.0,
            500.0,
            -500.0,
            float("inf"),
            float("-inf"),
            float("nan"),
        ],
        dtype=torch.float32,
        device=device,
    )
    torch.manual_seed(0)
    x = torch.cat(
        [pos, -pos, mids, -mids, specials, torch.randn(4096, device=device) * 50.0]
    )
    enc = torch.empty(x.numel(), dtype=torch.uint8, device=device)
    _test_e4m3_encode_kernel[(triton.cdiv(x.numel(), 256),)](
        x, enc, x.numel(), BLOCK=256
    )

    ref_enc = x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(torch.uint8)
    non_nan = ~torch.isnan(x)
    assert torch.equal(enc[non_nan], ref_enc[non_nan])
    assert ((enc[~non_nan] & 0x7F) == 0x7F).all()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("num_heads", [32, 64])
@pytest.mark.parametrize("emulate_fp8", [True, False])
def test_fp8_mqa_logits_triton_matches_reference(monkeypatch, num_heads, emulate_fp8):
    if not emulate_fp8 and not current_platform.has_device_capability(89):
        pytest.skip("native fp8e4nv requires SM89+")
    monkeypatch.setattr(
        "vllm.v1.attention.ops.mqa_logits_triton._should_emulate_fp8",
        lambda: emulate_fp8,
    )
    torch.manual_seed(42)
    device = torch.device("cuda")
    num_rows, num_keys, head_dim = 17, 192, 128
    max_seqlen_k = 140

    q = (torch.randn(num_rows, num_heads, head_dim, device=device) * 2.0).to(
        torch.float8_e4m3fn
    )
    k = (torch.randn(num_keys, head_dim, device=device) * 2.0).to(torch.float8_e4m3fn)
    k_scale = torch.rand(num_keys, dtype=torch.float32, device=device) * 0.5 + 0.1
    weights = torch.randn(num_rows, num_heads, dtype=torch.float32, device=device)

    # Mix empty, short, ragged, and offset rows.
    ks_list = [0, 0, 0, 5, 10, 32, 64, 64, 0, 12, 50, 50, 100, 0, 7, 19, 40]
    ke_list = [0, 1, 17, 69, 145, 172, 192, 64, 133, 88, 190, 51, 192, 64, 71, 150, 180]
    cu_ks = torch.tensor(ks_list, dtype=torch.int32, device=device)
    cu_ke = torch.tensor(ke_list, dtype=torch.int32, device=device)

    out = fp8_mqa_logits_triton(q, (k, k_scale), weights, cu_ks, cu_ke, max_seqlen_k)
    ref = _ref_fp8_mqa_logits(q, k, k_scale, weights, cu_ks, cu_ke, max_seqlen_k)

    assert out.shape == (num_rows, max_seqlen_k)
    assert out.dtype == torch.float32
    for i in range(num_rows):
        length = max(0, min(ke_list[i] - ks_list[i], max_seqlen_k))
        if length > 0:
            torch.testing.assert_close(
                out[i, :length], ref[i, :length], rtol=1e-2, atol=1e-2
            )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize(
    "next_n,use_2d_ctx", [(1, True), (2, True), (4, True), (2, False)]
)
@pytest.mark.parametrize("strided_cache", [False, True])
def test_fp8_paged_mqa_logits_triton_matches_reference(
    monkeypatch, next_n, use_2d_ctx, strided_cache
):
    # Exercise the SM80 software E4M3 path on every CUDA GPU.
    monkeypatch.setattr(
        "vllm.v1.attention.ops.mqa_logits_triton._should_emulate_fp8",
        lambda: True,
    )
    torch.manual_seed(123)
    device = torch.device("cuda")
    batch_size, num_heads, head_dim = 3, 32, 128
    block_size = 64
    num_blocks = 12
    max_model_len = 512

    if strided_cache:
        # Simulate multi-layer interleaved layout where stride(0) > block_size * (D+4).
        backing = torch.zeros(
            (num_blocks, 2, block_size, head_dim + 4), dtype=torch.uint8, device=device
        )
        kv_cache = backing[:, 0, :, :].unsqueeze(2)
    else:
        kv_cache = torch.zeros(
            (num_blocks, block_size, 1, head_dim + 4), dtype=torch.uint8, device=device
        )

    # Populate each block with valid FP8 keys + FP32 scales in the exact CUDA layout.
    k_fp8 = (torch.randn(num_blocks, block_size, head_dim, device=device) * 2.0).to(
        torch.float8_e4m3fn
    )
    k_scales = (
        torch.rand(num_blocks, block_size, dtype=torch.float32, device=device) + 0.1
    )
    for blk in range(num_blocks):
        page_flat = kv_cache[blk].reshape(-1)
        page_flat[: block_size * head_dim].copy_(
            k_fp8[blk].view(torch.uint8).reshape(-1)
        )
        page_flat[block_size * head_dim : block_size * (head_dim + 4)].copy_(
            k_scales[blk].view(torch.uint8).reshape(-1)
        )

    q = (torch.randn(batch_size, next_n, num_heads, head_dim, device=device) * 2.0).to(
        torch.float8_e4m3fn
    )
    weights = torch.randn(
        batch_size * next_n, num_heads, dtype=torch.float32, device=device
    )
    block_tables = torch.tensor(
        [
            [0, 3, 6, 9],
            [1, 4, 7, 10],
            [2, 5, 8, 11],
        ],
        dtype=torch.int32,
        device=device,
    )
    base_lens = [15, 135, 200]
    if use_2d_ctx:
        ctx_lens = torch.tensor(
            [
                [max(0, bl - (next_n - 1 - n)) for n in range(next_n)]
                for bl in base_lens
            ],
            dtype=torch.int32,
            device=device,
        )
    else:
        ctx_lens = torch.tensor(base_lens, dtype=torch.int32, device=device)

    out = fp8_paged_mqa_logits_triton(
        q, kv_cache, weights, ctx_lens, block_tables, max_model_len=max_model_len
    )
    ref = _ref_fp8_paged_mqa_logits(
        q, kv_cache, weights, ctx_lens, block_tables, max_model_len=max_model_len
    )

    assert out.shape == (batch_size * next_n, max_model_len)
    for b in range(batch_size):
        for n in range(next_n):
            row = b * next_n + n
            seq_len = (
                int(ctx_lens[b, n].item())
                if use_2d_ctx
                else max(0, base_lens[b] - (next_n - 1 - n))
            )
            torch.testing.assert_close(
                out[row, :seq_len], ref[row, :seq_len], rtol=1e-2, atol=1e-2
            )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rope_dim", [64, 0])
@pytest.mark.parametrize("num_heads", [8, 32])
@pytest.mark.parametrize("num_kv_splits", [1, 4])
def test_triton_mla_sparse_fwd_matches_reference(
    dtype, rope_dim, num_heads, num_kv_splits
):
    torch.manual_seed(7)
    device = torch.device("cuda")
    kv_lora_rank = 512
    dim_qk = kv_lora_rank + rope_dim
    num_tokens = 5
    num_kv_rows = 400
    index_topk = 140  # deliberately non-power-of-2 / non-multiple of BLOCK_N=32
    sm_scale = (dim_qk) ** -0.5

    q = torch.randn(num_tokens, num_heads, dim_qk, dtype=dtype, device=device)
    kv = torch.randn(num_kv_rows, dim_qk, dtype=dtype, device=device)
    indices = torch.randint(
        0, num_kv_rows, (num_tokens, index_topk), dtype=torch.int32, device=device
    )
    # Row 0: partially padded tail; Row 1: leading -1s; Row 2: all -1s (empty!).
    indices[0, 50:] = -1
    indices[1, :40] = -1
    indices[2, :] = -1

    out, lse = triton_mla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale=sm_scale,
        kv_lora_rank=kv_lora_rank,
        return_lse=True,
        num_kv_splits=num_kv_splits,
    )
    ref_out, ref_lse = _ref_mla_sparse_fwd(q, kv, indices, sm_scale, kv_lora_rank)

    assert out.dtype == dtype
    assert lse is not None and lse.dtype == torch.float32
    assert not torch.isnan(out).any()
    # All-masked row must produce exact zeros and -inf LSE.
    assert torch.equal(out[2], torch.zeros_like(out[2]))
    assert torch.isneginf(lse[2]).all()

    atol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 2e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(out, ref_out, rtol=rtol, atol=atol)
    valid_rows = [0, 1, 3, 4]
    torch.testing.assert_close(
        lse[valid_rows], ref_lse[valid_rows], rtol=1e-2, atol=1e-2
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
def test_triton_mla_sparse_cuda_graph_capture_and_replay(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.attention.ops.mqa_logits_triton._should_emulate_fp8",
        lambda: True,
    )
    torch.manual_seed(99)
    device = torch.device("cuda")
    batch_size, next_n, idx_heads, idx_dim = 2, 1, 32, 128
    block_size, num_blocks, max_model_len = 64, 8, 256

    kv_idx = torch.zeros(
        (num_blocks, block_size, 1, idx_dim + 4), dtype=torch.uint8, device=device
    )
    k_fp8 = torch.randn(num_blocks, block_size, idx_dim, device=device).to(
        torch.float8_e4m3fn
    )
    k_scales = torch.ones(num_blocks, block_size, dtype=torch.float32, device=device)
    for blk in range(num_blocks):
        page_flat = kv_idx[blk].reshape(-1)
        page_flat[: block_size * idx_dim].copy_(
            k_fp8[blk].view(torch.uint8).reshape(-1)
        )
        page_flat[block_size * idx_dim : block_size * (idx_dim + 4)].copy_(
            k_scales[blk].view(torch.uint8).reshape(-1)
        )

    q_idx = torch.randn(batch_size, next_n, idx_heads, idx_dim, device=device).to(
        torch.float8_e4m3fn
    )
    w_idx = torch.randn(batch_size, idx_heads, dtype=torch.float32, device=device)
    bt = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=device)
    ctx_lens = torch.tensor([[80], [110]], dtype=torch.int32, device=device)

    q_mla = torch.randn(batch_size, 16, 576, dtype=torch.bfloat16, device=device)
    kv_mla = torch.randn(256, 576, dtype=torch.bfloat16, device=device)
    topk_idx = torch.randint(0, 256, (batch_size, 64), dtype=torch.int32, device=device)

    # Warmup before capture.
    fp8_paged_mqa_logits_triton(q_idx, kv_idx, w_idx, ctx_lens, bt, max_model_len)
    triton_mla_sparse_fwd(q_mla, kv_mla, topk_idx, sm_scale=576**-0.5, return_lse=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits_out = fp8_paged_mqa_logits_triton(
            q_idx, kv_idx, w_idx, ctx_lens, bt, max_model_len
        )
        attn_out, lse_out = triton_mla_sparse_fwd(
            q_mla, kv_mla, topk_idx, sm_scale=576**-0.5, return_lse=True
        )

    # Replay with different device-side lengths and indices.
    ctx_lens.copy_(torch.tensor([[140], [30]], dtype=torch.int32, device=device))
    topk_idx.copy_(
        torch.randint(0, 256, (batch_size, 64), dtype=torch.int32, device=device)
    )
    graph.replay()
    torch.accelerator.synchronize()

    ref_logits = _ref_fp8_paged_mqa_logits(
        q_idx, kv_idx, w_idx, ctx_lens, bt, max_model_len
    )
    ref_attn, ref_lse = _ref_mla_sparse_fwd(q_mla, kv_mla, topk_idx, 576**-0.5, 512)
    torch.testing.assert_close(
        logits_out[0, :140], ref_logits[0, :140], rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        logits_out[1, :30], ref_logits[1, :30], rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(attn_out, ref_attn, rtol=2e-2, atol=2e-2)
    assert lse_out is not None
    torch.testing.assert_close(lse_out, ref_lse, rtol=1e-2, atol=1e-2)


def _ref_rope_2d(
    x: torch.Tensor,
    pos: torch.Tensor,
    cos_sin: torch.Tensor,
    interleave: bool,
) -> torch.Tensor:
    rot = cos_sin.shape[-1]
    half = rot // 2
    cs = cos_sin[pos.long()].unsqueeze(1)
    cos, sin = cs[..., :half], cs[..., half:]
    out = x.float().clone()
    r = out[..., :rot]
    if interleave:
        x1, x2 = r[..., 0::2].clone(), r[..., 1::2].clone()
        r[..., 0::2] = x1 * cos - x2 * sin
        r[..., 1::2] = x2 * cos + x1 * sin
    else:
        x1, x2 = r[..., :half].clone(), r[..., half:].clone()
        r[..., :half] = x1 * cos - x2 * sin
        r[..., half:] = x2 * cos + x1 * sin
    return out


def _fp8_max_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    def _ordered_key(t: torch.Tensor) -> torch.Tensor:
        u = t.contiguous().view(torch.uint8).to(torch.int64)
        return torch.where(u >= 0x80, 0xFF - u, u + 0x80)

    return int((_ordered_key(a) - _ordered_key(b)).abs().max().item())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA")
@pytest.mark.parametrize(
    "has_indexer,index_rope_interleave",
    [(True, False), (True, True), (False, False)],
)
def test_fused_q_sm80_emulate_fp8_bf16_query_path(
    monkeypatch, has_indexer: bool, index_rope_interleave: bool
):
    """Regression test for SM80 engine init failure in `_fused_q_kernel`:
    with `quantize_mqa=False` and SM < 8.9 (`EMULATE_FP8=True`), Triton must
    compile without `fp8e4nv` casts and produce accurate RoPE + indexer FP8
    outputs.
    """
    monkeypatch.setattr(
        current_platform,
        "has_device_capability",
        lambda cap, device_id=0: cap <= 80,
    )
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_tokens, num_q_heads, rope_dim, kv_lora_rank = 7, 16, 64, 512
    num_idx_heads, idx_head_dim = 32, 128
    max_pos = 128

    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    q_pe = torch.randn(
        num_tokens, num_q_heads, rope_dim, dtype=torch.bfloat16, device=device
    )
    ql_nope = torch.randn(
        num_tokens, num_q_heads, kv_lora_rank, dtype=torch.bfloat16, device=device
    )
    q_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    half = rope_dim // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    index_q = (
        torch.randn(
            num_tokens,
            num_idx_heads,
            idx_head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        if has_indexer
        else None
    )
    index_weights = (
        torch.randn(num_tokens, num_idx_heads, dtype=torch.float32, device=device)
        if has_indexer
        else None
    )
    softmax_scale = idx_head_dim**-0.5
    head_scale = num_idx_heads**-0.5

    iq_fp8, iw_out, mqa_q = dsv32_kernels.fused_q(
        positions,
        q_pe,
        cos_sin,
        index_q,
        cos_sin if has_indexer else None,
        ql_nope,
        q_scale,
        index_weights,
        softmax_scale,
        head_scale,
        has_indexer=has_indexer,
        index_rope_interleave=index_rope_interleave,
        quantize_mqa=False,
    )

    assert mqa_q.shape == q_pe.shape
    assert mqa_q.dtype == torch.bfloat16
    assert not torch.isnan(mqa_q).any()
    ref_q_pe = _ref_rope_2d(q_pe, positions, cos_sin, interleave=True)
    torch.testing.assert_close(mqa_q.float(), ref_q_pe, rtol=1e-2, atol=1e-2)

    if has_indexer:
        assert index_q is not None and index_weights is not None
        assert iq_fp8.shape == index_q.shape
        assert iq_fp8.dtype == current_platform.fp8_dtype()
        assert not torch.isnan(iq_fp8.float()).any()
        ref_iq = _ref_rope_2d(
            index_q, positions, cos_sin, interleave=index_rope_interleave
        )
        amax = ref_iq.abs().amax(dim=-1, keepdim=True)
        fp8_max = (
            224.0 if current_platform.fp8_dtype() == torch.float8_e4m3fnuz else 448.0
        )
        scale = torch.exp2(
            torch.ceil(torch.log2(torch.clamp(amax, min=1e-4) / fp8_max))
        )
        ref_iq_fp8 = (ref_iq / scale).to(current_platform.fp8_dtype())
        assert _fp8_max_ulp_diff(iq_fp8, ref_iq_fp8) <= 1
        ref_iw = index_weights * scale.squeeze(-1) * softmax_scale * head_scale
        torch.testing.assert_close(iw_out, ref_iw, rtol=1e-3, atol=1e-3)
