# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, VllmConfig, set_current_vllm_config
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
    default_vllm_config, dist_init, dtype, num_tokens, device
) -> None:
    """Reach the real norm through Indexer.forward, stopping before GPU-only ops."""
    from transformers import Glm5NextTextConfig

    from vllm.models.glm5next.common.attention import Indexer

    if current_platform.is_cuda():
        _require_deep_gemm()
    torch.manual_seed(0)
    with set_default_torch_dtype(dtype), torch.device(device):
        indexer = Indexer(
            SimpleNamespace(model_config=SimpleNamespace(max_model_len=512)),
            Glm5NextTextConfig(
                index_head_dim=128, index_n_heads=32, index_topk=512, index_kpool=4
            ),
            hidden_size=8,
            q_lora_rank=8,
            quant_config=None,
            cache_config=CacheConfig(block_size=512),
            topk_indices_buffer=torch.empty(num_tokens, 512, dtype=torch.int32),
            prefix="key_norm_test",
        )
        hidden = torch.randn(num_tokens, 8)
    for layer in (indexer.wq_b, indexer.wk_weights_proj):
        layer.weight.normal_()
    indexer._wp_fp32 = (
        indexer.wk_weights_proj.weight[indexer.head_dim :, :].t().contiguous().float()
    )
    for layer in (indexer.wq_b, indexer.wk_weights_proj):
        layer.quant_method.process_weights_after_loading(layer)
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
        with torch.cuda.stream(torch.cuda.Stream()):
            indexer.k_norm(key)
        torch.accelerator.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replayed = indexer.k_norm(key)
        for _ in range(3):
            graph.replay()
            torch.testing.assert_close(replayed, expected, rtol=0, atol=0)
