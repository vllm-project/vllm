# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

import pytest
import torch

from vllm.config import (
    AttentionConfig,
    CacheConfig,
    ModelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype


def _require_deep_gemm() -> None:
    from vllm.utils.deep_gemm import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("kpool indexer requires DeepGEMM")


def _build(indexer_cls, backend: str):
    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        return indexer_cls(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="cuda"),
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
    """The kpool indexer must read kernel_config.sparse_indexer_topk_backend
    and hand it to the shared SparseIndexerTopk dispatcher, rather than
    hard-coding its own cooperative/persistent/per_row choice."""
    _require_deep_gemm()
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize("backend", ["auto", "aiter", "per_row", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend_rocm(
    backend: str,
) -> None:
    """The AMD kpool indexer must go through the same dispatcher, so the
    AITER decode top-k is reachable from kernel_config."""
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool

    assert _build(SparseAttnIndexerKpool, backend).topk_backend == backend


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only dispatch")
@pytest.mark.parametrize(
    "configured,full_cudagraph,expected",
    [
        ("auto", False, "aiter"),
        ("auto", True, "auto"),
        ("aiter", True, "aiter"),
        ("per_row", False, "per_row"),
    ],
)
def test_kpool_decode_topk_backend_never_auto_selects_aiter_under_full_cudagraph(
    monkeypatch: pytest.MonkeyPatch,
    configured: str,
    full_cudagraph: bool,
    expected: str,
) -> None:
    """A FULL cudagraph bakes in one kernel for every replay, so "auto" cannot
    use the context length to pick AITER and must stay on the in-tree kernel.
    Explicit backends keep working, and outside FULL the long-context heuristic
    still narrows to AITER."""
    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.models.glm5next.amd import sparse_indexer

    monkeypatch.setattr(
        rocm_aiter_ops, "is_indexer_top_k_enabled", lambda: True, raising=False
    )
    monkeypatch.setattr(
        rocm_aiter_ops,
        "is_indexer_top_k_supported",
        lambda **kwargs: True,
        raising=False,
    )

    assert (
        sparse_indexer._kpool_decode_topk_backend(
            configured,
            num_rows=8,
            max_valid_seq_len=64 * 1024,
            select_k=512,
            index_kpool=4,
            full_cudagraph=full_cudagraph,
        )
        == expected
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [7, 2048])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA graph validation requires GPU",
            ),
        ),
    ],
)
@torch.inference_mode()
def test_kpool_key_norm_does_not_compile_a_rank_local_reduction(
    default_vllm_config, dist_init, tmp_path, dtype, num_tokens, device
) -> None:
    """Reach the real norm through Indexer.forward, stopping before GPU-only ops."""
    from transformers import Glm5NextTextConfig

    from vllm.models.glm5next.common.attention import Indexer

    if current_platform.is_cuda():
        _require_deep_gemm()
    torch.manual_seed(0)
    hf_config = Glm5NextTextConfig(
        architectures=["Glm5NextForCausalLM"],
        index_head_dim=128,
        index_n_heads=32,
        index_topk=512,
        index_kpool=4,
        max_position_embeddings=512,
    )
    hf_config.save_pretrained(tmp_path)
    vllm_config = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=512,
            enforce_eager=True,
            dtype=dtype,
        ),
        cache_config=CacheConfig(block_size=512, enable_prefix_caching=False),
        attention_config=AttentionConfig(sparse_mla_prefill_workspace_factor=2),
    )
    with (
        set_current_vllm_config(vllm_config),
        set_default_torch_dtype(dtype),
        torch.device(device),
    ):
        indexer = Indexer(
            vllm_config,
            hf_config,
            hidden_size=8,
            q_lora_rank=8,
            quant_config=None,
            cache_config=CacheConfig(block_size=512, enable_prefix_caching=False),
            topk_indices_buffer=torch.empty(num_tokens, 512, dtype=torch.int32),
            prefix="key_norm_test",
        )
        hidden = torch.randn(num_tokens, 8)
        for layer in (indexer.wq_b, indexer.wk, indexer.weights_proj):
            layer.weight.normal_()
        indexer.process_weights_after_loading()
        for layer in (indexer.wq_b, indexer.wk_weights_proj):
            layer.quant_method.process_weights_after_loading(layer)
        assert indexer.k_norm.weight is not None
        assert indexer.k_norm.bias is not None
        indexer.k_norm.weight.normal_()
        indexer.k_norm.bias.normal_()

    captured = []

    class NormReached(Exception):
        pass

    def stop_after_norm(module, args, output):
        captured.append((args[0], output))
        raise NormReached

    torch.compiler.reset()
    with (
        set_current_vllm_config(vllm_config),
        indexer.k_norm.register_forward_hook(stop_after_norm),
        pytest.raises(NormReached),
        torch.compiler.set_stance("fail_on_recompile"),
    ):
        indexer(hidden, hidden, positions=None, rotary_emb=None)

    key, output = captured[0]
    assert key.stride() == (160, 1)
    assert output.dtype == dtype
    assert indexer.k_norm.weight.dtype == indexer.k_norm.bias.dtype == torch.float32
    expected = torch.nn.functional.layer_norm(
        key.float(), (128,), indexer.k_norm.weight, indexer.k_norm.bias, 1e-6
    ).to(dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)

    if device == "cuda":
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.accelerator.current_stream())
        with torch.cuda.stream(warmup_stream):
            indexer.k_norm(key)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = indexer.k_norm(key)
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(replayed, expected, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm-only workspace")
def test_kpool_profiling_reserves_decode_logits_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiling must reserve worst-case decode logits in WorkspaceManager so a
    subsequent decode step of SparseAttnIndexerKpool does not grow a locked
    workspace when decode_rows * max_pool_len * 4 exceeds the prefill gather
    workspace."""
    from types import SimpleNamespace

    from transformers import Glm5NextTextConfig

    from vllm._aiter_ops import rocm_aiter_ops
    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.forward_context import set_forward_context
    from vllm.models.glm5next.amd.sparse_indexer import SparseAttnIndexerKpool
    from vllm.models.glm5next.common.attention import Glm5NextIndexerCache
    from vllm.models.glm5next.common.model import get_sparse_topk_buffer_width
    from vllm.v1.attention.backends.mla.indexer import (
        DeepSeekV32IndexerDecodeMetadata,
        DeepseekV32IndexerMetadata,
    )
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        lock_workspace,
        reset_workspace_manager,
    )

    monkeypatch.setattr(rocm_aiter_ops, "is_enabled", lambda: True)

    device = torch.device("cuda")
    text_cfg = Glm5NextTextConfig()
    head_dim = text_cfg.index_head_dim
    n_heads = text_cfg.index_n_heads
    topk_tokens = text_cfg.index_topk
    index_kpool = text_cfg.index_kpool
    assert head_dim is not None and n_heads is not None
    assert topk_tokens is not None and index_kpool is not None

    max_num_seqs = 16
    num_spec = 5
    decode_rows = max_num_seqs * (num_spec + 1)
    max_num_batched_tokens = 256
    max_model_len = 16384
    max_pool_len = max_model_len // index_kpool
    # Prefill gather workspace is smaller than worst-case decode logits
    # (decode_rows * max_pool_len * 4 B = 1.5 MiB).
    max_total_seq_len = max_pool_len // 2

    cfg = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
        ),
        cache_config=CacheConfig(block_size=128),
        kernel_config={"sparse_indexer_topk_backend": "torch"},
    )
    cfg.speculative_config = SimpleNamespace(num_speculative_tokens=num_spec)  # type: ignore[assignment]

    prefix = "model.layers.0.self_attn.indexer.k_cache"
    topk_buf = torch.empty(
        (max_num_batched_tokens, get_sparse_topk_buffer_width(text_cfg)),
        dtype=torch.int32,
        device=device,
    )

    reset_workspace_manager()
    init_workspace_manager(device)
    try:
        with set_current_vllm_config(cfg):
            k_cache = Glm5NextIndexerCache(
                head_dim=head_dim + 4,
                dtype=torch.uint8,
                prefix=prefix,
                cache_config=cfg.cache_config,
                index_kpool=index_kpool,
            )
            k_cache.kv_cache = torch.zeros(
                (4, 64, head_dim + 4), dtype=torch.uint8, device=device
            )
            indexer = SparseAttnIndexerKpool(
                k_cache=k_cache,
                quant_block_size=head_dim,
                scale_fmt="ue8m0",
                topk_tokens=topk_tokens,
                head_dim=head_dim,
                max_pool_len=max_pool_len,
                max_total_seq_len=max_total_seq_len,
                topk_indices_buffer=topk_buf,
                skip_k_cache_insert=True,
            )

        hidden_states = torch.zeros(
            (max_num_batched_tokens, text_cfg.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        q_quant = torch.zeros(
            (max_num_batched_tokens, n_heads, head_dim),
            dtype=current_platform.fp8_dtype(),
            device=device,
        )
        k = torch.zeros(
            (max_num_batched_tokens, head_dim), dtype=torch.bfloat16, device=device
        )
        weights = torch.zeros(
            (max_num_batched_tokens, n_heads), dtype=torch.float32, device=device
        )

        # 1. Profiling dummy run (attn_metadata=None) reserves workspace.
        with set_current_vllm_config(cfg), set_forward_context(None, cfg):
            indexer(hidden_states, q_quant, k, weights, index_kpool=index_kpool)

        lock_workspace()

        # 2. Runtime decode step at worst-case decode_rows must fit the locked
        # workspace. max_seq_len > topk_tokens bypasses short-decode fast path.
        decode_meta = DeepseekV32IndexerMetadata(
            seq_lens=torch.ones(decode_rows, dtype=torch.int32, device=device),
            max_seq_len=topk_tokens + index_kpool,
            slot_mapping=torch.full(
                (decode_rows,), -1, dtype=torch.int64, device=device
            ),
            num_decodes=decode_rows,
            num_decode_tokens=decode_rows,
            num_prefills=0,
            num_prefill_tokens=0,
            decode=DeepSeekV32IndexerDecodeMetadata(
                block_table=torch.zeros(
                    (decode_rows, 4), dtype=torch.int32, device=device
                ),
                seq_lens=torch.ones((decode_rows, 1), dtype=torch.int32, device=device),
                decode_lens=torch.ones(decode_rows, dtype=torch.int32, device=device),
                requires_padding=False,
                schedule_metadata=torch.zeros(
                    (16, 2), dtype=torch.int32, device=device
                ),
            ),
        )
        with (
            set_current_vllm_config(cfg),
            set_forward_context({prefix: decode_meta}, cfg),
        ):
            out = indexer(
                hidden_states[:decode_rows],
                q_quant[:decode_rows],
                k[:decode_rows],
                weights[:decode_rows],
                index_kpool=index_kpool,
            )
        assert out is topk_buf
    finally:
        reset_workspace_manager()
