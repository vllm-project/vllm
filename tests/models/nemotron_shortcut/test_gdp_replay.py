# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDP replay (`--use-replayssm` on GDP layers) wiring, on CPU with the
FlashInfer op faked: the tracker commit rule, tracker binding and the per-step
commit claim, the gate parameters the op reads,
and what `NemotronShortcutGDPMixer` hands the op for spec, decode and prefill rows.
The op's arithmetic is FlashInfer's and is covered by its own tests."""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.models.nemotron_shortcut.nvidia import gdp as gdp_module
from vllm.models.nemotron_shortcut.nvidia.gdp import (
    NemotronShortcutGDPMixer,
    gdp_replay_state_dtype,
    gdp_replay_state_shape,
)
from vllm.models.nemotron_shortcut.nvidia.gdp_replay import (
    FLUSH_MIN,
    GdpReplayTrackers,
    is_prefix_view,
)

from .utils import make_config

HEADS, NUM_BLOCKS = 2, 12


def _bound_trackers(num_blocks: int, conv_rows: int) -> GdpReplayTrackers:
    trackers = GdpReplayTrackers()
    trackers.bind(torch.zeros(num_blocks, conv_rows, 4))
    return trackers


def _reference_commit(base, hist_len, bank, prev_query_len, accepted, modulus):
    """COMMON.md §4.3 of FlashInfer's gdp_conv_replay, plus the deferral:
    nothing is committed for a block whose last step was not a replay step."""
    if prev_query_len == 0:
        return base, hist_len, bank
    flush = hist_len >= FLUSH_MIN
    if flush:
        base = (base + hist_len) % modulus
        bank ^= 1
        hist_len = 0
    return base, hist_len + accepted, bank


def test_commit_follows_the_op_contract_over_many_steps():
    generator = torch.Generator().manual_seed(0)
    modulus, tokens = 12, 8
    trackers = _bound_trackers(NUM_BLOCKS, modulus)
    slots = torch.tensor([3, 5, 9], dtype=torch.int32)
    expected = {int(s): (0, 0, 0) for s in slots}
    prev_query_len = {int(s): 0 for s in slots}
    for _ in range(200):
        query_lens = torch.randint(1, tokens + 1, (3,), generator=generator)
        accepted = torch.minimum(
            torch.randint(1, tokens + 1, (3,), generator=generator),
            torch.tensor([max(prev_query_len[int(s)], 1) for s in slots]),
        )
        base, hist_len, bank = trackers.commit(slots, accepted, query_lens)
        for row, slot in enumerate(slots.tolist()):
            expected[slot] = _reference_commit(
                *expected[slot], prev_query_len[slot], int(accepted[row]), modulus
            )
            prev_query_len[slot] = int(query_lens[row])
            assert (int(base[row]), int(hist_len[row]), int(bank[row])) == (
                expected[slot]
            )
            # The op's bounds: banked history fits one bank of 8 + T slots.
            assert int(hist_len[row]) <= 8 + tokens
            assert 0 <= int(base[row]) < modulus


def test_commit_skips_reset_blocks_and_records_query_lengths():
    trackers = _bound_trackers(NUM_BLOCKS, 4)
    trackers.hist_len[2], trackers.prev_query_len[2] = 5, 4
    trackers.hist_len[3], trackers.prev_query_len[3] = 5, 4
    trackers.reset(torch.tensor([3], dtype=torch.int32))

    slots = torch.tensor([2, 3], dtype=torch.int32)
    _, hist_len, _ = trackers.commit(
        slots, torch.tensor([2, 2]), torch.tensor([4, 1], dtype=torch.int32)
    )

    assert hist_len.tolist() == [7, 0]
    assert trackers.prev_query_len[slots.long()].tolist() == [4, 1]


def test_padding_rows_touch_only_the_null_block():
    trackers = _bound_trackers(NUM_BLOCKS, 4)
    trackers.hist_len[1:] = 3
    trackers.prev_query_len[1:] = 2
    before = trackers.hist_len[1:].clone()

    trackers.commit(torch.zeros(4, dtype=torch.int32), torch.ones(4), torch.ones(4))
    trackers.reset(torch.zeros(4, dtype=torch.int32))

    assert torch.equal(trackers.hist_len[1:], before)


def test_binding_sizes_the_trackers_and_keeps_captured_addresses():
    trackers = GdpReplayTrackers()
    trackers.bind(torch.zeros(NUM_BLOCKS, 8, 4))
    base = trackers.base
    assert base.numel() == NUM_BLOCKS and trackers.base_modulus == 8

    # Every layer binds, and an extensible cache rebinds after capture.
    trackers.bind(torch.zeros(NUM_BLOCKS, 8, 4))
    trackers.bind(torch.zeros(NUM_BLOCKS - 2, 8, 4))
    assert trackers.base is base

    # Growing would drop the cursors live requests and captured graphs hold.
    with pytest.raises(RuntimeError, match="would drop them"):
        trackers.bind(torch.zeros(NUM_BLOCKS + 4, 8, 4))


def test_a_prefix_view_shares_start_block_shape_and_strides():
    pools = (torch.zeros(NUM_BLOCKS, 3, 4), torch.zeros(NUM_BLOCKS, 5))
    committed = tuple(pool[: NUM_BLOCKS - 2] for pool in pools)
    assert is_prefix_view(pools, committed)
    assert is_prefix_view(pools, pools)

    # More blocks, other storage, other strides, other block shape.
    assert not is_prefix_view(committed, pools)
    assert not is_prefix_view(pools, tuple(view.clone() for view in committed))
    assert not is_prefix_view(pools, tuple(pool[::2] for pool in pools))
    assert not is_prefix_view(pools, (pools[0].view(NUM_BLOCKS, 12)[:4], pools[1][:4]))


def test_the_first_layer_to_see_a_steps_metadata_commits():
    step, other_group, next_step = (
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
    )
    assert GdpReplayTrackers.claim(step)
    assert not GdpReplayTrackers.claim(step)
    assert GdpReplayTrackers.claim(other_group)
    assert GdpReplayTrackers.claim(next_step)


# --- The mixer -----------------------------------------------------------


@pytest.fixture
def forward_context(dist_init, monkeypatch):
    monkeypatch.setattr(gdp_module, "missing_gdp_kernels", lambda: ())
    monkeypatch.setattr(gdp_module, "gdp_replay_kernel_missing", lambda: False)
    with torch.inference_mode(), set_forward_context(None, get_current_vllm_config()):
        yield


class _FakeOp:
    """Records each call and returns a recognizable output."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        x, out_rows = args[0], args[10]
        batch = out_rows.size(0)
        tokens = x.size(0) // batch
        values = torch.arange(batch * tokens * HEADS * 64, dtype=x.dtype)
        values = values.view(batch, tokens, HEADS, 64)
        # zero_inactive_out: padding requests (NULL_BLOCK_ID) get +0.
        values[out_rows == 0] = 0
        out = kwargs.get("out")
        if out is None:
            return values
        assert out.shape == values.shape and out.is_contiguous()
        return out.copy_(values)


@pytest.fixture
def fake_op(monkeypatch):
    op = _FakeOp()
    module = types.ModuleType("flashinfer.gdp_conv_replay_decode")
    module.gdp_conv_replay = op
    if "flashinfer" not in sys.modules:
        monkeypatch.setitem(sys.modules, "flashinfer", types.ModuleType("flashinfer"))
    monkeypatch.setitem(sys.modules, "flashinfer.gdp_conv_replay_decode", module)
    return op


def _mixer(
    num_spec: int, prefix: str, trackers: GdpReplayTrackers | None = None
) -> NemotronShortcutGDPMixer:
    config = make_config(
        hidden_size=32,
        layer_types=["linear_attention"],
        gdp_num_heads=HEADS,
        gdp_num_groups=HEADS,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        gdp_num_householder=3,
        gdp_conv_kernel=4,
        gdp_norm_eps=1e-5,
    )
    mixer = NemotronShortcutGDPMixer(
        config,
        cache_config=CacheConfig(use_replayssm=True),
        prefix=prefix,
        gdp_replay_trackers=trackers,
    )
    mixer.num_spec = num_spec
    mixer.replay_capture_sizes = []
    shapes = gdp_replay_state_shape(1, HEADS, HEADS, 128, 64, 4, 3, num_spec)
    dtypes = gdp_replay_state_dtype(torch.bfloat16, "auto")
    mixer.kv_cache = tuple(
        torch.zeros(NUM_BLOCKS, *shape, dtype=dtype)
        for shape, dtype in zip(shapes, dtypes)
    )
    assert mixer.gdp_replay_trackers is not None
    mixer.gdp_replay_trackers.bind(mixer.kv_cache[0])
    return mixer


def _bind_raw(mixer: NemotronShortcutGDPMixer, raw: torch.Tensor) -> None:
    dtypes = gdp_replay_state_dtype(torch.bfloat16, "auto")
    mixer.get_state_dtype = lambda: dtypes  # type: ignore[method-assign]
    mixer.bind_kv_cache(raw)


def _raw_pages(mixer: NemotronShortcutGDPMixer) -> torch.Tensor:
    page = sum(state[0].numel() * state.element_size() for state in mixer.kv_cache)
    return torch.zeros(NUM_BLOCKS, 1, 1, page, dtype=torch.int8)


def test_a_committed_rebind_keeps_the_capacity_views(forward_context):
    """The replay op compiles per pool shape, block count included. When an
    extensible KV cache rebinds a committed prefix after graph capture, the
    layer keeps the capacity views, so the op meets the shapes it compiled."""
    mixer = _mixer(3, "model.layers.0.capacity_views")
    raw = _raw_pages(mixer)
    _bind_raw(mixer, raw)
    capacity = mixer.kv_cache
    assert capacity[0].shape[0] == NUM_BLOCKS

    _bind_raw(mixer, raw[: NUM_BLOCKS - 2])
    assert mixer.kv_cache is capacity

    # A remap that moves the storage, as a defragmenting commit can, takes the
    # new views.
    moved = raw.clone()[: NUM_BLOCKS - 2]
    _bind_raw(mixer, moved)
    assert mixer.kv_cache[0].data_ptr() == moved.data_ptr()
    assert mixer.kv_cache[0].shape[0] == NUM_BLOCKS - 2


def test_shared_replayssm_tracker_setup_skips_gdp_layers(forward_context):
    """--use-replayssm makes the model runner bind the shared ReplaySSM trackers
    (Mamba-2, Gated DeltaNet) on every layer that sets MambaBase.use_replayssm.
    GDP replay keeps its own cursors, so a GDP layer must not be picked up."""
    from vllm.v1.worker.replayssm_utils import initialize_replayssm_ring_trackers

    name = "model.layers.0.shared_trackers"
    mixer = _mixer(3, name)
    assert mixer.use_gdp_replayssm and not mixer.use_replayssm
    groups = [SimpleNamespace(layer_names=[name])]
    initialize_replayssm_ring_trackers(
        SimpleNamespace(kv_cache_groups=groups), {name: mixer}
    )
    assert not hasattr(mixer, "_replayssm_ring_start")


def test_without_replay_a_rebind_takes_the_new_views(forward_context):
    mixer = _mixer(3, "model.layers.0.plain_rebind")
    mixer.gdp_replay_trackers = None
    raw = _raw_pages(mixer)
    _bind_raw(mixer, raw)
    _bind_raw(mixer, raw[: NUM_BLOCKS - 2])
    assert mixer.kv_cache[0].shape[0] == NUM_BLOCKS - 2


def _replay_group(num_spec: int):
    """Two GDP layers sharing the model's trackers."""
    trackers = GdpReplayTrackers()
    first, second = (
        _mixer(num_spec, f"model.layers.{i}.gdp_replay_{num_spec}", trackers)
        for i in (0, 1)
    )
    return first, second


def test_a_groups_first_layer_plans_the_step_for_the_rest():
    trackers = _bound_trackers(NUM_BLOCKS, 8)
    metadata = SimpleNamespace()
    loc = torch.tensor([0, 4, 5], dtype=torch.int32)
    step = trackers.commit_step(
        metadata,
        torch.tensor([5, 7], dtype=torch.int32),
        loc,
        torch.ones(2, dtype=torch.int32),
        tokens=4,
        padded=3,
        num_rows=5,
    )
    assert GdpReplayTrackers.step_of(metadata) is step
    assert step.slots.tolist() == [5, 7, 0] and step.valid_len.tolist() == [4, 1, 0]
    # Request 1's single token lands at row 4 of the 2 x 4 rectangle.
    assert step.dst is not None and step.dst.tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(AssertionError, match="plans each step"):
        GdpReplayTrackers.step_of(SimpleNamespace())


@pytest.mark.parametrize("padded", [1, 2, 3, 5])
def test_every_planned_input_starts_16_byte_aligned(padded):
    """gdp_conv_replay rejects index and cursor tensors that don't start on a
    16-B boundary, at every padded batch size."""
    trackers = _bound_trackers(NUM_BLOCKS, 8)
    step = trackers.commit_step(
        SimpleNamespace(),
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        tokens=4,
        padded=padded,
        num_rows=4,
    )
    planned = (step.slots, step.valid_len, step.base, step.hist_len, step.ring_bank)
    for tensor in planned:
        assert tensor.numel() == padded and tensor.data_ptr() % 16 == 0


def test_a_standalone_layer_keeps_its_own_trackers(forward_context):
    first, second = _replay_group(num_spec=3)
    alone = _mixer(3, "model.layers.5.gdp_replay_alone")
    assert first.gdp_replay_trackers is second.gdp_replay_trackers
    assert alone.gdp_replay_trackers is not first.gdp_replay_trackers


def _run(mixer, metadata, projected):
    mixer._gdn_metadata = lambda: metadata
    core_out = torch.zeros(projected.size(0), HEADS, 64, dtype=projected.dtype)
    mixer._forward_replay(projected, core_out)
    return core_out


def _projected(rows: int) -> torch.Tensor:
    width = HEADS * (64 + 704 + 4)
    return torch.randn(rows, width).to(torch.bfloat16)


def _spec_metadata(slots, query_lens, num_accepted, num_spec):
    loc = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    loc[1:] = torch.tensor(query_lens).cumsum(0)
    tokens = int(loc[-1])
    return SimpleNamespace(
        num_actual_tokens=tokens,
        spec_sequence_masks=torch.ones(len(slots), dtype=torch.bool),
        spec_sequence_masks_cpu=torch.ones(len(slots), dtype=torch.bool),
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=len(slots),
        num_spec_decode_tokens=tokens,
        spec_token_indx=torch.arange(tokens),
        non_spec_token_indx=torch.empty(0, dtype=torch.long),
        spec_state_indices_tensor=torch.tensor(slots, dtype=torch.int32)
        .unsqueeze(1)
        .repeat(1, num_spec + 1),
        spec_query_start_loc=loc,
        num_accepted_tokens=torch.tensor(num_accepted, dtype=torch.int32),
    )


def test_gate_params_are_the_loaded_parameters(forward_context):
    mixer = _mixer(3, "gate_params")
    mixer.A_log.weight_loader(mixer.A_log, torch.tensor([0.5, 1.5]))
    mixer.dt_bias.weight_loader(mixer.dt_bias, torch.tensor([-1.0, 2.0]))
    assert mixer.gate_params.tolist() == [[0.5, 1.5], [-1.0, 2.0]]


def test_spec_rows_take_whole_in_proj_rows(forward_context, fake_op):
    first, second = _replay_group(num_spec=3)
    trackers = first.gdp_replay_trackers
    # Block 5 committed 3 tokens last step; block 7 is due to flush.
    trackers.base[5], trackers.hist_len[5], trackers.prev_query_len[5] = 1, 3, 4
    trackers.base[7], trackers.hist_len[7], trackers.prev_query_len[7] = 2, 9, 4
    metadata = _spec_metadata([5, 7], [4, 4], [2, 3], num_spec=3)
    projected = _projected(8)

    core_out = _run(first, metadata, projected)
    _run(second, metadata, projected)

    assert len(fake_op.calls) == 2
    for args, kwargs in fake_op.calls:
        x, ab, slots, base, hist_len, valid_len = (
            args[0],
            args[1],
            args[4],
            args[11],
            args[12],
            args[13],
        )
        # x and ab are the conv and gate columns of the same in_proj rows.
        assert x.data_ptr() == projected.data_ptr() + HEADS * 64 * 2
        assert ab.data_ptr() == projected.data_ptr() + HEADS * 768 * 2
        assert x.shape == (8, 704 * HEADS) and ab.shape == (8, 4 * HEADS)
        assert x.stride() == ab.stride() == projected.stride()
        assert slots.tolist() == [5, 7] and args[10] is slots
        # Block 5: 3 + 2 accepted; block 7 flushed: base 2 + 9 mod 8 rows, bank 1.
        assert base.tolist() == [1, 3]
        assert hist_len.tolist() == [5, 3]
        assert kwargs["ring_bank"].tolist() == [0, 1]
        assert valid_len.tolist() == [4, 4]
        assert kwargs["null_block_id"] == 0 and kwargs["zero_inactive_out"]
        assert args[5] is first.gate_params or args[5] is second.gate_params
    # The second layer read the committed trackers without committing again.
    assert trackers.hist_len[[5, 7]].tolist() == [5, 3]
    assert torch.equal(core_out.flatten(), torch.arange(8 * HEADS * 64).to(core_out))


def test_ragged_windows_are_padded_for_the_op(forward_context, fake_op):
    first, _ = _replay_group(num_spec=3)
    metadata = _spec_metadata([5, 7], [4, 1], [1, 1], num_spec=3)
    projected = _projected(5)

    core_out = _run(first, metadata, projected)

    x, valid_len = fake_op.calls[0][0][0], fake_op.calls[0][0][13]
    assert x.size(0) == 8 and valid_len.tolist() == [4, 1]
    rows = HEADS * 64
    full = torch.arange(8 * rows).view(8, HEADS, 64).to(core_out)
    # Request 1's single token sits at padded row 4.
    assert torch.equal(core_out, full[[0, 1, 2, 3, 4]])


@pytest.mark.parametrize("accepted,hist_len", [(None, 3), (2, 4)])
def test_decode_and_prefill_rows_without_spec(
    forward_context, fake_op, monkeypatch, accepted, hist_len
):
    """A plain decode row commits what its previous step accepted: one token
    without drafts (no counts in the metadata), else the given count."""
    first, _ = _replay_group(num_spec=0)
    trackers = first.gdp_replay_trackers
    trackers.hist_len[7], trackers.prev_query_len[7] = 4, 1
    prefill_calls = []

    def fake_prefill(*args):
        prefill_calls.append(args)
        out = args[10]
        assert out is not None and out.shape == (3, HEADS, 64)
        return out.fill_(7.0)

    monkeypatch.setattr(first, "_prefill", fake_prefill)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        spec_sequence_masks=None,
        num_decodes=1,
        num_decode_tokens=1,
        num_prefills=1,
        non_spec_state_indices_tensor=torch.tensor([5, 7], dtype=torch.int32),
        prefill_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        prefill_state_indices=torch.tensor([7], dtype=torch.int32),
        prefill_has_initial_state=torch.tensor([False]),
        num_accepted_tokens=(
            None if accepted is None else torch.tensor([accepted, 1], dtype=torch.int32)
        ),
    )
    trackers.hist_len[5], trackers.prev_query_len[5] = 2, 1

    core_out = _run(first, metadata, _projected(4))

    (args, kwargs) = fake_op.calls[0]
    assert args[0].size(0) == 1 and args[4].tolist() == [5]
    assert args[12].tolist() == [hist_len] and args[13].tolist() == [1]
    (prefill_args,) = prefill_calls
    assert prefill_args[7].tolist() == [7] and prefill_args[9] is None
    # Prefilled block 7 starts over; decoded block 5 kept its history.
    assert trackers.hist_len[[5, 7]].tolist() == [hist_len, 0]
    assert trackers.prev_query_len[7] == 0
    assert torch.all(core_out[1:] == 7.0)


def test_eager_batches_pad_to_a_captured_size(forward_context, fake_op):
    first, _ = _replay_group(num_spec=3)
    # Captured sizes are token counts: 16 tokens hold 4 requests of 4 tokens.
    first.replay_capture_sizes = [4, 16, 32]
    metadata = _spec_metadata([5, 7], [4, 4], [1, 1], num_spec=3)
    projected = _projected(8)

    core_out = _run(first, metadata, projected)

    args, _ = fake_op.calls[0]
    assert args[0].size(0) == 16
    assert torch.equal(args[0][:8], projected[:, HEADS * 64 : HEADS * 768])
    assert not args[0][8:].any()
    assert args[4].tolist() == [5, 7, 0, 0] and args[13].tolist() == [4, 4, 0, 0]
    assert torch.equal(core_out.flatten(), torch.arange(8 * HEADS * 64).to(core_out))


def _conv_rows_of(rows: torch.Tensor) -> int:
    """Where a view of in_proj rows starts its conv channels."""
    return rows.data_ptr() + HEADS * 64 * 2


def test_profile_runs_zero_the_output(forward_context, fake_op):
    first, _ = _replay_group(num_spec=3)
    first._gdn_metadata = lambda: None
    core_out = torch.full((4, HEADS, 64), float("nan"), dtype=torch.bfloat16)
    first._forward_replay(_projected(4), core_out)
    assert not fake_op.calls and not core_out.any()


def test_full_windows_write_the_layer_output_in_place(forward_context, fake_op):
    """The op writes the layer's output buffer itself; graph padding rows past
    the step's tokens are zeroed."""
    first, _ = _replay_group(num_spec=3)
    metadata = _spec_metadata([5, 7], [4, 4], [1, 1], num_spec=3)
    projected = _projected(10)
    core_out = torch.full((10, HEADS, 64), float("nan"), dtype=torch.bfloat16)
    first._gdn_metadata = lambda: metadata
    first._forward_replay(projected, core_out)

    args, kwargs = fake_op.calls[0]
    assert kwargs["out"].data_ptr() == core_out.data_ptr()
    assert args[0].data_ptr() == _conv_rows_of(projected) and args[0].size(0) == 8
    assert torch.equal(
        core_out[:8].flatten(), torch.arange(8 * HEADS * 64).to(core_out)
    )
    assert not core_out[8:].any()


def _mixed_metadata(spec_first: bool):
    """Two spec requests (blocks 5 and 7, 4 tokens each) and a 3-token prefill
    (block 9): either after the spec requests, or between them."""
    masks = [True, True, False] if spec_first else [True, False, True]
    lens = [4, 4, 3] if spec_first else [4, 3, 4]
    spec_tokens = [
        t
        for r, (spec, n) in enumerate(zip(masks, lens))
        for t in range(sum(lens[:r]), sum(lens[:r]) + n)
        if spec
    ]
    prefill_tokens = [t for t in range(11) if t not in spec_tokens]
    return SimpleNamespace(
        num_actual_tokens=11,
        spec_sequence_masks=torch.tensor(masks),
        spec_sequence_masks_cpu=torch.tensor(masks),
        num_prefills=1,
        num_decodes=0,
        num_prefill_tokens=3,
        num_decode_tokens=0,
        num_spec_decodes=2,
        num_spec_decode_tokens=8,
        spec_token_indx=torch.tensor(spec_tokens),
        non_spec_token_indx=torch.tensor(prefill_tokens),
        spec_state_indices_tensor=torch.tensor([[5] * 4, [7] * 4], dtype=torch.int32),
        spec_query_start_loc=torch.tensor([0, 4, 8], dtype=torch.int32),
        num_accepted_tokens=torch.ones(2, dtype=torch.int32),
        non_spec_query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        non_spec_state_indices_tensor=torch.tensor([9], dtype=torch.int32),
        has_initial_state=torch.tensor([False]),
    )


def _fake_prefill(mixer, monkeypatch):
    calls = []

    def fake_prefill(*args):
        calls.append(args)
        out = args[10]
        return (out if out is not None else args[0].new_empty(3, HEADS, 64)).fill_(7)

    monkeypatch.setattr(mixer, "_prefill", fake_prefill)
    return calls


@pytest.mark.parametrize("spec_first", [True, False])
def test_mixed_steps_place_spec_and_prefill_rows(
    forward_context, fake_op, monkeypatch, spec_first
):
    """With the spec requests first (the batch reorder's usual order), both
    parts read their in_proj rows and write their outputs in place; otherwise
    their rows are gathered and scattered."""
    first, _ = _replay_group(num_spec=3)
    prefill_calls = _fake_prefill(first, monkeypatch)
    metadata = _mixed_metadata(spec_first)
    projected = _projected(12)
    core_out = _run(first, metadata, projected)

    (args, kwargs), (prefill_args,) = fake_op.calls[0], prefill_calls
    in_place = (
        args[0].data_ptr() == _conv_rows_of(projected),
        kwargs["out"] is not None,
        prefill_args[0].data_ptr() == _conv_rows_of(projected[8]),
        prefill_args[10] is not None,
    )
    assert in_place == (spec_first,) * 4
    spec = torch.arange(8 * HEADS * 64).view(8, HEADS, 64).to(core_out)
    assert torch.equal(core_out[metadata.spec_token_indx], spec)
    assert torch.all(core_out[metadata.non_spec_token_indx] == 7)
    assert not core_out[11:].any()
    # The prefilled block starts over.
    assert first.gdp_replay_trackers.hist_len[9] == 0


def test_padding_requests_take_the_rows_that_follow(
    forward_context, fake_op, monkeypatch
):
    """An eager step pads to a captured batch size with the rows after the
    spec rows when the buffers have them, instead of copying."""
    first, _ = _replay_group(num_spec=3)
    first.replay_capture_sizes = [16]
    _fake_prefill(first, monkeypatch)
    projected = _projected(20)
    core_out = _run(first, _mixed_metadata(True), projected)

    args, kwargs = fake_op.calls[0]
    assert args[0].data_ptr() == _conv_rows_of(projected) and args[0].size(0) == 16
    assert kwargs["out"].data_ptr() == core_out.data_ptr()
    assert args[4].tolist() == [5, 7, 0, 0]
    spec = torch.arange(8 * HEADS * 64).view(8, HEADS, 64).to(core_out)
    assert torch.equal(core_out[:8], spec)
    assert torch.all(core_out[8:11] == 7) and not core_out[11:].any()
