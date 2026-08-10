# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guards for HY V4's attention-sink wiring.

The sink is part of the architecture, so dropping it changes model outputs
without raising. Two things can silently break it: the backend not advertising
`supports_sink` (the layer then loads the weight but disables the bias), and the
kernel wrappers not forwarding ``attn_sink``. Both are asserted here.
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.models.hy_v4.nvidia import flashmla_sparse as hyv4_flashmla_sparse
from vllm.models.hy_v4.nvidia.flashmla_sparse import (
    HYV4FlashMLASparseBackend,
    HYV4FlashMLASparseImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend

from .test_weight_loading import _hf_config, _model_config


def _bare_impl(
    sinks: torch.Tensor | None,
    num_heads: int = 4,
    prefill_padding: int = 64,
    fp8_decode_padded_heads: int = 64,
) -> HYV4FlashMLASparseImpl:
    """Build an impl without the heavy workspace/platform initialization."""
    impl = object.__new__(HYV4FlashMLASparseImpl)
    impl.sinks = sinks
    impl.num_heads = num_heads
    impl.prefill_padding = prefill_padding
    impl.fp8_decode_padded_heads = fp8_decode_padded_heads
    impl.softmax_scale = 0.5
    return impl


def test_backend_advertises_sink_support_without_renaming() -> None:
    # The parent lacks sink support, which is the gap this backend closes.
    assert not FlashMLASparseBackend.supports_sink()
    assert HYV4FlashMLASparseBackend.supports_sink()
    assert HYV4FlashMLASparseBackend.get_impl_cls() is HYV4FlashMLASparseImpl
    # The name is load-bearing: shared code promotes a quantized KV cache to
    # fp8_ds_mla only for "FLASHMLA_SPARSE".
    assert HYV4FlashMLASparseBackend.get_name() == "FLASHMLA_SPARSE"
    assert HYV4FlashMLASparseBackend.is_mla()
    assert HYV4FlashMLASparseBackend.is_sparse()


@pytest.mark.parametrize(
    "sinks",
    [
        torch.zeros(4, dtype=torch.bfloat16),  # kernels require fp32
        torch.zeros(3, dtype=torch.float32),  # wrong head count
        torch.zeros((4, 1), dtype=torch.float32),  # wrong rank
    ],
)
def test_validate_sinks_rejects_unusable_tensors(sinks: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        HYV4FlashMLASparseImpl._validate_sinks(sinks, num_heads=4)


def test_sinks_for_query_pads_with_neg_inf() -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)
    q = torch.zeros(2, 4, 576)

    # Padded lanes must be -inf so the extra kernel heads get no sink effect.
    padded = impl._sinks_for_query(q, head_dim=1, kernel_heads=64)
    assert padded is not None
    assert padded.shape == (64,)
    assert torch.equal(padded[:4], sinks)
    assert torch.isneginf(padded[4:]).all()

    # Without padding the live parameter is handed through, so weights loaded
    # after construction are observed.
    assert impl._sinks_for_query(q, head_dim=1, kernel_heads=4) is sinks
    assert _bare_impl(None)._sinks_for_query(q, head_dim=1, kernel_heads=64) is None


def test_sinks_for_query_rejects_layout_mismatch() -> None:
    impl = _bare_impl(torch.arange(4, dtype=torch.float32))
    q = torch.zeros(2, 8, 576)
    with pytest.raises(ValueError, match="head count must match"):
        impl._sinks_for_query(q, head_dim=1, kernel_heads=64)

    q4 = torch.zeros(2, 4, 576)
    with pytest.raises(ValueError, match="cannot be smaller"):
        impl._sinks_for_query(q4, head_dim=1, kernel_heads=2)


def test_bf16_kernel_forwards_attn_sink(monkeypatch) -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)
    captured: dict[str, torch.Tensor | None] = {}

    def fake_sparse_fwd(q, kv, indices, scale, attn_sink=None, topk_length=None):
        captured["attn_sink"] = attn_sink
        return (torch.zeros(q.shape[0], q.shape[1], 512),)

    monkeypatch.setattr(hyv4_flashmla_sparse, "flash_mla_sparse_fwd", fake_sparse_fwd)
    impl._bf16_flash_mla_kernel(
        q=torch.zeros(2, 4, 576),
        kv_c_and_k_pe_cache=torch.zeros(8, 576),
        topk_indices=torch.zeros(2, 4, dtype=torch.int32),
    )

    attn_sink = captured["attn_sink"]
    assert attn_sink is not None, "sink was dropped before the BF16 kernel"
    # num_heads=4 is not a multiple of prefill_padding=64, so q and the sink are
    # both padded to 64 heads.
    assert attn_sink.shape == (64,)
    assert torch.equal(attn_sink[:4], sinks)


def test_fp8_kernel_forwards_attn_sink(monkeypatch) -> None:
    sinks = torch.arange(4, dtype=torch.float32)
    impl = _bare_impl(sinks)
    captured: dict[str, torch.Tensor | None] = {}

    def fake_with_kvcache(**kwargs):
        captured["attn_sink"] = kwargs["attn_sink"]
        q = kwargs["q"]
        return torch.zeros(q.shape[0], q.shape[1], q.shape[2], 512), torch.zeros(1)

    monkeypatch.setattr(
        hyv4_flashmla_sparse, "flash_mla_with_kvcache", fake_with_kvcache
    )

    class _KernelMetadata:
        dummy_block_table = torch.zeros(1, 1, dtype=torch.int32)
        cache_lens = torch.zeros(1, dtype=torch.int32)
        scheduler_metadata = None

    impl._fp8_flash_mla_kernel(
        q=torch.zeros(1, 2, 4, 576),
        kv_c_and_k_pe_cache=torch.zeros(8, 656, dtype=torch.uint8),
        topk_indices=torch.zeros(1, 2, 4, dtype=torch.int32),
        kernel_metadata=_KernelMetadata(),  # type: ignore[arg-type]
    )

    attn_sink = captured["attn_sink"]
    assert attn_sink is not None, "sink was dropped before the FP8 decode kernel"
    assert attn_sink.shape == (64,)
    assert torch.equal(attn_sink[:4], sinks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_sparse_layer_binds_sink_capable_backend(
    tmp_path, dist_init, workspace_init
) -> None:
    from vllm.models.hy_v4 import HYV4ForCausalLM

    torch.set_default_dtype(torch.bfloat16)
    hf_config = _hf_config(enable_ihc=False, sparse=True)
    assert hf_config.learnable_sink
    vllm_config = VllmConfig(model_config=_model_config(tmp_path, hf_config))

    with set_current_vllm_config(vllm_config), torch.device("cuda"):
        model = HYV4ForCausalLM(vllm_config=vllm_config, prefix="")

    attn = model.model.layers[0].self_attn
    assert attn.mla_attn.attn_backend is HYV4FlashMLASparseBackend
    # fp32 is what the kernels require; bf16 would mean the bias got disabled.
    assert attn.learnable_sink_param.dtype == torch.float32
    # The impl must hold the live parameter, not a detached copy.
    assert attn.mla_attn.impl.sinks is attn.learnable_sink_param
