# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the GLM kpool indexer's top-k backend wiring."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="CUDA-only dispatch"
)


def _require_deep_gemm() -> None:
    from vllm.utils.deep_gemm import has_deep_gemm

    if not has_deep_gemm():
        pytest.skip("kpool indexer requires DeepGEMM")


@pytest.mark.parametrize("backend", ["auto", "persistent", "cooperative", "torch"])
def test_kpool_indexer_dispatches_through_shared_topk_backend(backend: str) -> None:
    """The kpool indexer must read kernel_config.sparse_indexer_topk_backend
    and hand it to the shared SparseIndexerTopk dispatcher, rather than
    hard-coding its own cooperative/persistent/per_row choice."""
    _require_deep_gemm()
    from vllm.models.glm5next.nvidia.sparse_indexer import SparseAttnIndexerKpool

    cfg = VllmConfig(kernel_config={"sparse_indexer_topk_backend": backend})
    with set_current_vllm_config(cfg):
        op = SparseAttnIndexerKpool(
            k_cache=None,
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=4096,
            max_total_seq_len=8192,
            topk_indices_buffer=torch.empty(8, 2176, dtype=torch.int32, device="cuda"),
        )
    assert op.topk_backend == backend


@pytest.mark.skipif(
    not current_platform.is_device_capability(90), reason="requires Hopper"
)
@torch.inference_mode()
def test_kpool_auto_uses_deep_select_on_hopper() -> None:
    """GLM pool-level top-k uses the shared Hopper DeepSelect dispatch."""
    _require_deep_gemm()
    from vllm.model_executor.layers.indexer_topk import SparseIndexerTopk

    logits = torch.randn(4, 262144, dtype=torch.float32, device="cuda")
    pool_lens = torch.tensor([511, 512, 777, 2048], dtype=torch.int32, device="cuda")
    output = torch.empty(4, 512, dtype=torch.int32, device="cuda")
    selector = SparseIndexerTopk("auto")

    assert selector.resolve_backend(logits, 512, 4) == "deep_select"
    selector(logits, pool_lens, 1, output, 512, 2048)

    for row, length in enumerate(pool_lens.tolist()):
        count = min(length, 512)
        picked = output[row, :count].long()
        torch.testing.assert_close(
            logits[row, picked].sort().values,
            logits[row, :length].topk(count).values.sort().values,
            rtol=0,
            atol=0,
        )
        assert torch.all(output[row, count:] == -1)
