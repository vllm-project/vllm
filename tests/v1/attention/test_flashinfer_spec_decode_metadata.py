# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing tests for the native FlashInfer spec-decode path.

The decode bucket contains uniform query_len > 1 batches when speculative
decoding is enabled. Under full-CUDAGraph capture, vLLM may pad the batch
with zero-length request slots (the ``zero_rows`` strategy). The metadata
builder must:

1. Route those uniform query_len > 1 batches to ``FISpecDecode`` even when
   one or more padded zero-length rows are present.
2. Pass per-request metadata to ``plan()`` (qo_indptr, paged_kv_indptr,
   paged_kv_last_page_len) without expanding the padded slots.

The original aggregate routing rule (``num_decode_tokens // num_decodes``)
silently misroutes ``[5, 5, 0]`` to ``FIDecode`` (10 % 3 == 1 → query_len
falls back to 1). The qo_indptr-delta scan added in
``flashinfer.py:build()`` fixes this; the parametrized cases below pin the
behavior.
"""

import unittest.mock

import pytest
import torch

from tests.v1.attention.utils import (
    BatchSpec,
    create_common_attn_metadata,
    create_standard_kv_cache_spec,
    create_vllm_config,
)
from vllm.config import SpeculativeConfig, set_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PerLayerParameters

if not current_platform.is_cuda():
    pytest.skip(
        "FlashInfer routing tests require CUDA.",
        allow_module_level=True,
    )

flashinfer = pytest.importorskip("flashinfer")

from vllm.v1.attention.backends.flashinfer import (  # noqa: E402
    FISpecDecode,
    FlashInferMetadataBuilder,
)

BLOCK_SIZE = 16
MODEL = "Qwen/Qwen2.5-0.5B"
NUM_SPEC_TOKENS = 4
DEVICE = torch.device("cuda:0")


def _mock_get_per_layer_parameters(vllm_config, layer_names, impl_cls):
    head_size = vllm_config.model_config.get_head_size()
    return {
        name: PerLayerParameters(
            window_left=-1,
            logits_soft_cap=0.0,
            sm_scale=1.0 / (head_size**0.5),
        )
        for name in layer_names
    }


class _RecordingPrefillWrapper:
    """Stand-in for ``BatchPrefillWithPagedKVCacheWrapper`` that records the
    kwargs passed to ``plan()``. The real wrapper allocates FlashInfer
    workspace and runs CPU-side scheduling we don't need for a routing test.
    """

    def __init__(self):
        self.plan_kwargs: dict | None = None
        self.requested_batch_size: int | None = None
        self.requested_use_cudagraph: bool | None = None
        # Fields that ``FlashInferImpl.forward`` may inspect.
        self._window_left = -1
        self._logits_soft_cap = 0.0
        self._sm_scale = 0.0
        self._causal = False

    def plan(self, **kwargs):
        # Snapshot tensor inputs so post-build mutation of the persistent
        # buffers can't fool the test.
        recorded: dict = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                recorded[k] = v.detach().cpu().clone()
            else:
                recorded[k] = v
        self.plan_kwargs = recorded
        self._window_left = kwargs.get("window_left", self._window_left)
        self._logits_soft_cap = kwargs.get("logits_soft_cap", self._logits_soft_cap)
        self._sm_scale = kwargs.get("sm_scale", self._sm_scale)
        self._causal = kwargs.get("causal", self._causal)


def _make_builder() -> FlashInferMetadataBuilder:
    vllm_config = create_vllm_config(
        model_name=MODEL,
        max_model_len=512,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=512,
    )
    vllm_config.speculative_config = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=NUM_SPEC_TOKENS,
        prompt_lookup_max=4,
        prompt_lookup_min=2,
    )
    kv_cache_spec = create_standard_kv_cache_spec(vllm_config)

    with (
        set_current_vllm_config(vllm_config),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.can_use_trtllm_attention",
            return_value=False,
        ),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.get_per_layer_parameters",
            _mock_get_per_layer_parameters,
        ),
    ):
        builder = FlashInferMetadataBuilder(
            kv_cache_spec, ["layer.0"], vllm_config, DEVICE
        )

    # Threshold should reflect 1 + num_speculative_tokens so query_len=5
    # batches stay in the decode bucket.
    assert builder.reorder_batch_threshold == 1 + NUM_SPEC_TOKENS
    return builder


def _build(builder: FlashInferMetadataBuilder, batch_spec: BatchSpec):
    common = create_common_attn_metadata(batch_spec, BLOCK_SIZE, DEVICE)
    fake_wrapper = _RecordingPrefillWrapper()

    def _fake_get_spec_decode_prefill_wrapper(batch_size, use_cudagraph):
        # batch_size must be the request count, not token count, per the
        # CUDAGraph wrapper contract documented in
        # ``_get_spec_decode_prefill_wrapper``.
        fake_wrapper.requested_batch_size = batch_size
        fake_wrapper.requested_use_cudagraph = use_cudagraph
        return fake_wrapper

    with (
        unittest.mock.patch.object(
            builder,
            "_get_spec_decode_prefill_wrapper",
            side_effect=_fake_get_spec_decode_prefill_wrapper,
        ),
        unittest.mock.patch(
            "vllm.v1.attention.backends.flashinfer.can_use_trtllm_attention",
            return_value=False,
        ),
    ):
        attn_metadata = builder.build(common_prefix_len=0, common_attn_metadata=common)

    return attn_metadata, fake_wrapper


@pytest.mark.parametrize(
    "query_lens, seq_lens, expected_qo_indptr, expected_last_page_len",
    [
        # The regression case: trailing zero row from CG padding.
        # Old aggregate logic: 10 % 3 == 1 → query_len=1 → FIDecode (bug).
        # New per-row scan: nonzero rows are [5, 5] → query_len=5 →
        # FISpecDecode.
        ([5, 5, 0], [64, 72, 0], [0, 5, 10, 10], [16, 8, 0]),
        # Sanity: no padding, all real rows.
        ([5, 5, 5], [64, 72, 80], [0, 5, 10, 15], [16, 8, 16]),
    ],
    ids=["padded_zero_row", "no_padding"],
)
def test_spec_decode_routes_to_fispecdecode(
    query_lens, seq_lens, expected_qo_indptr, expected_last_page_len
):
    """Uniform query_len > 1 in the decode bucket must produce FISpecDecode
    metadata, with per-request plan() kwargs preserving zero_rows entries."""
    builder = _make_builder()
    attn_metadata, fake_wrapper = _build(
        builder, BatchSpec(seq_lens=seq_lens, query_lens=query_lens)
    )

    assert isinstance(attn_metadata.decode, FISpecDecode), (
        f"expected FISpecDecode for query_lens={query_lens}, "
        f"got {type(attn_metadata.decode).__name__}"
    )
    assert attn_metadata.decode.wrapper is fake_wrapper

    num_reqs = len(query_lens)
    assert fake_wrapper.requested_batch_size == num_reqs
    assert fake_wrapper.plan_kwargs is not None
    assert fake_wrapper.plan_kwargs["causal"] is True

    qo = fake_wrapper.plan_kwargs["qo_indptr"]
    assert qo.tolist() == expected_qo_indptr, (
        f"qo_indptr mismatch: got {qo.tolist()}, want {expected_qo_indptr}"
    )

    last_page_len = fake_wrapper.plan_kwargs["paged_kv_last_page_len"]
    assert last_page_len.tolist() == expected_last_page_len, (
        f"paged_kv_last_page_len mismatch: got {last_page_len.tolist()}, "
        f"want {expected_last_page_len}"
    )

    paged_kv_indptr = fake_wrapper.plan_kwargs["paged_kv_indptr"]
    # zero_rows: trailing padded row contributes no extra KV pages, so the
    # last two entries of paged_kv_indptr coincide.
    if query_lens[-1] == 0:
        assert paged_kv_indptr[-1].item() == paged_kv_indptr[-2].item(), (
            f"padded row should not extend paged_kv_indptr: {paged_kv_indptr.tolist()}"
        )
