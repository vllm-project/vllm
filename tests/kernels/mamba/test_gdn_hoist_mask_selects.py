# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for ``VLLM_GDN_HOIST_MASK_SELECTS`` — the all-mode mixed-batch
mask-select hoist (host-sync attribution doc, component 1).

In mixed (spec + prefill) eager batches every GDN layer boolean-indexed
request-level metadata with the DEVICE spec mask; each such select runs
torch.nonzero -> count DtoH + cudaStreamSynchronize (9-11 per layer, 324+
per prefill-containing step). The builder now pre-selects the row slices
once per step with the already-available CPU mask (sync-free). Tests:

- builder: hoisted slices bit-equal the device-mask selects; populated only
  for mixed batches with the flag on (pure-spec / nospec / prefill-only and
  flag-off builds keep them None).
- layer: the real ``_forward_core`` on flag-on vs flag-off metadata is
  bit-identical, and the layer performs ZERO CUDA-bool-mask index ops under
  the flag (dispatch counter) vs >= 9 without it.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip(
        reason="GDN mask-hoist tests require CUDA (Triton/FLA kernels).",
        allow_module_level=True,
    )

from tests.kernels.mamba import test_gdn_all_mode_prefill as harness  # noqa: E402
from tests.kernels.mamba import (  # noqa: E402
    test_gdn_all_mode_spec_decode as spec_harness,
)
from tests.kernels.mamba.test_gdn_index_prep import (  # noqa: E402
    _prep_from_common,
    _run_layer_with_meta,
)
from tests.v1.attention.utils import BatchSpec  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402

DEVICE = torch.device("cuda")
BLOCK = spec_harness.BLOCK
NUM_SPEC = spec_harness.NUM_SPEC

MIXED_BATCH = BatchSpec(seq_lens=[192, 103], query_lens=[192, 3])
MIXED_DRAFTS = [-1, NUM_SPEC]
MIXED_ACCEPTED = [1, 2]
MIXED_PREV = [-1, 1]

SEL_FIELDS = (
    ("ns_all_state_indices_sel", "all_state_indices_tensor", False),
    ("ns_block_idx_last_computed_sel", "block_idx_last_computed_token", False),
    (
        "ns_block_idx_first_scheduled_sel",
        "block_idx_first_scheduled_token",
        False,
    ),
    ("ns_block_idx_last_scheduled_sel", "block_idx_last_scheduled_token", False),
    ("ns_num_computed_tokens_sel", "num_computed_tokens", False),
    ("spec_all_state_indices_sel", "all_state_indices_tensor", True),
    ("spec_block_idx_last_scheduled_sel", "block_idx_last_scheduled_token", True),
    ("spec_block_idx_last_computed_sel", "block_idx_last_computed_token", True),
    (
        "spec_block_idx_prev_step_sel",
        "block_idx_last_scheduled_token_prev_step",
        True,
    ),
    ("spec_block_idx_packed_anchors_sel", "block_idx_packed_anchors", True),
    (
        "spec_block_idx_packed_anchors_spec_sel",
        "block_idx_packed_anchors_spec",
        True,
    ),
)


class _CudaBoolIndexCounter(TorchDispatchMode):
    """Counts aten.index.Tensor calls indexed by a CUDA bool tensor — the
    dispatch-level signature of the device-mask selects whose internal
    nonzero forces the DtoH + streamSync."""

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
        if func.overloadpacket.__name__ == "index":
            indices = args[1] if len(args) > 1 else ()
            for t in indices or ():
                if (
                    isinstance(t, torch.Tensor)
                    and t.is_cuda
                    and t.dtype == torch.bool
                ):
                    self.count += 1
                    break
        return func(*args, **(kwargs or {}))


def _build_mixed(monkeypatch, hoist, with_prep):
    cfg = spec_harness._make_spec_config("all")
    monkeypatch.setenv("VLLM_GDN_HOIST_MASK_SELECTS", "1" if hoist else "0")
    meta, common = _build_spec_meta(
        cfg, MIXED_BATCH, MIXED_DRAFTS, MIXED_ACCEPTED, MIXED_PREV, with_prep
    )
    return cfg, meta, common


def _build_spec_meta(cfg, batch, drafts, accepted, prev, with_prep):
    from tests.v1.attention.utils import create_common_attn_metadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK,
            shapes=((16, 64),),
            dtypes=(torch.float16,),
            num_speculative_blocks=NUM_SPEC,
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg,
        device=DEVICE,
    )
    common = create_common_attn_metadata(
        batch, BLOCK, DEVICE, arange_block_indices=True
    )
    table = common.block_table_tensor
    n, _ = table.shape
    extra = torch.arange(
        n * NUM_SPEC, dtype=table.dtype, device=table.device
    ).reshape(n, NUM_SPEC) + int(table.max().item() + 1)
    common.block_table_tensor = torch.cat([table, extra], dim=1)
    common.block_table_tensor.add_(1)
    prev_t = torch.tensor(prev, dtype=torch.int32, device=DEVICE)
    kwargs = dict(
        num_decode_draft_tokens_cpu=torch.tensor(drafts, dtype=torch.int32),
        num_accepted_tokens=torch.tensor(
            accepted, dtype=torch.int32, device=DEVICE
        ),
        prev_last_scheduled_idx=prev_t,
    )
    if with_prep:
        kwargs["block_idx_prep"] = _prep_from_common(common, prev_t, BLOCK)
    with set_current_vllm_config(cfg):
        meta = builder.build(0, common, **kwargs)
    return meta, common


@pytest.mark.parametrize("with_prep", [False, True])
def test_builder_hoisted_selects_match_device_mask(monkeypatch, with_prep):
    """Every hoisted slice is bit-equal to the device-mask select of the
    corresponding unselected field (spec and non-spec sides), with and
    without the CS1 prep buffers (packed-anchor slices)."""
    torch.manual_seed(0)
    _, meta, _ = _build_mixed(monkeypatch, hoist=True, with_prep=with_prep)
    assert meta.spec_sequence_masks is not None
    mask = meta.spec_sequence_masks
    for sel_name, src_name, is_spec in SEL_FIELDS:
        src = getattr(meta, src_name)
        sel = getattr(meta, sel_name)
        if src is None:
            assert sel is None, sel_name
            continue
        ref = src[mask] if is_spec else src[~mask]
        assert sel is not None, sel_name
        assert torch.equal(sel, ref), sel_name
        assert not sel.is_cpu


def test_builder_hoist_only_for_mixed_batches(monkeypatch):
    """Flag off, pure-spec, prefill-only and nospec builds keep every
    hoisted field None."""
    torch.manual_seed(0)
    # Flag off on a mixed batch.
    _, meta, _ = _build_mixed(monkeypatch, hoist=False, with_prep=False)
    for sel_name, _, _ in SEL_FIELDS:
        assert getattr(meta, sel_name) is None, sel_name
    monkeypatch.setenv("VLLM_GDN_HOIST_MASK_SELECTS", "1")
    # Pure-spec batch (no prefills).
    cfg = spec_harness._make_spec_config("all")
    meta, _ = _build_spec_meta(
        cfg,
        BatchSpec(seq_lens=[103, 231], query_lens=[3, 3]),
        [NUM_SPEC, NUM_SPEC],
        [2, 1],
        [1, -1],
        False,
    )
    for sel_name, _, _ in SEL_FIELDS:
        assert getattr(meta, sel_name) is None, sel_name
    # Nospec mixed decode+prefill and prefill-only (no spec mask at all).
    from tests.v1.attention.utils import create_common_attn_metadata
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    from vllm.v1.kv_cache_interface import MambaSpec

    cfg_ns = harness._make_vllm_config(BLOCK, "all")
    builder = GDNAttentionMetadataBuilder(
        kv_cache_spec=MambaSpec(
            block_size=BLOCK, shapes=((16, 64),), dtypes=(torch.float16,)
        ),
        layer_names=[harness.PREFIX],
        vllm_config=cfg_ns,
        device=DEVICE,
    )
    for batch in (
        BatchSpec(seq_lens=[101, 192], query_lens=[1, 192]),
        BatchSpec(seq_lens=[320], query_lens=[320]),
    ):
        common = create_common_attn_metadata(
            batch, BLOCK, DEVICE, arange_block_indices=True
        )
        common.block_table_tensor.add_(1)
        with set_current_vllm_config(cfg_ns):
            meta = builder.build(0, common)
        for sel_name, _, _ in SEL_FIELDS:
            assert getattr(meta, sel_name) is None, sel_name


@pytest.mark.parametrize("with_prep", [False, True])
def test_layer_mixed_parity_and_zero_device_mask_selects(
    monkeypatch, with_prep
):
    """The real _forward_core on flag-on metadata is bit-identical to the
    flag-off run, and performs ZERO CUDA-bool-mask index ops (vs >= 9
    without the hoist)."""
    torch.manual_seed(0)
    weights = harness._make_weights(DEVICE)
    inputs = harness._make_inputs(195, DEVICE)
    (conv_c, _), *states = spec_harness._rand_states(DEVICE, 4)
    seeds = {
        (1, 1): (conv_c, states[0][1]),
        (1, 2): (None, states[1][1]),
        (1, 3): (None, states[2][1]),
    }

    cfg, meta_off, common = _build_mixed(
        monkeypatch, hoist=False, with_prep=with_prep
    )
    with _CudaBoolIndexCounter() as c_off:
        out_off, conv_off, ssm_off = _run_layer_with_meta(
            cfg, meta_off, common, inputs, weights, seeds, num_spec=NUM_SPEC
        )
    assert c_off.count >= 9

    cfg, meta_on, common = _build_mixed(
        monkeypatch, hoist=True, with_prep=with_prep
    )
    with _CudaBoolIndexCounter() as c_on:
        out_on, conv_on, ssm_on = _run_layer_with_meta(
            cfg, meta_on, common, inputs, weights, seeds, num_spec=NUM_SPEC
        )
    assert c_on.count == 0

    assert torch.equal(out_on, out_off)
    assert torch.equal(conv_on, conv_off)
    assert torch.equal(ssm_on, ssm_off)
