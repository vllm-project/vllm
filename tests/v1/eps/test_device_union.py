# ABOUTME: Validates device-side EPS union scorer behaviour.
# ABOUTME: Ensures JL gating keeps high-energy groups and honors strict mode.

import pytest

try:  # pragma: no cover - torch required
    import torch
except ModuleNotFoundError:  # pragma: no cover
    pytest.skip("torch not available", allow_module_level=True)

from vllm.config.eps import EpsConfig
from vllm.v1.eps import build_eps_forward_context, eps_context
from vllm.v1.eps.config import to_runtime_config
from vllm.v1.eps.device_union import apply_device_union, union_select_for_request
from vllm.v1.eps.state import EpsJLState


def _make_state(*, num_layers: int, num_heads: int, num_groups: int, head_dim: int, sketch_dim: int):
    state = EpsJLState.init(
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        num_groups=num_groups,
        sketch_dim=sketch_dim,
        device=torch.device("cpu"),
    )
    state.Phi.copy_(torch.eye(head_dim, sketch_dim).repeat(num_heads, 1, 1))
    return state


def test_device_union_prefers_high_energy_group():
    cfg = to_runtime_config(
        EpsConfig(
            enabled=True,
            method="jl",
            scope="union",
            heads="all",
            group_blocks=1,
            last_n=0,
            alpha=1.1,
        )
    )

    state = _make_state(num_layers=1, num_heads=2, num_groups=2, head_dim=2, sketch_dim=2)
    state.G[0, :, 0] = torch.eye(2)
    state.G[0, :, 1] = torch.eye(2) * 64.0
    state.frob2[0, :, 0] = 1.0
    state.frob2[0, :, 1] = 64.0

    q = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    block_ids = torch.tensor([10, 11], dtype=torch.int64)
    block_mapping = {10: 0, 11: 1}

    decision = union_select_for_request(
        cfg=cfg,
        layer_index=0,
        q_attn=q,
        state=state,
        block_ids=block_ids,
        block_mapping=block_mapping,
        seq_len=32,
        block_size=16,
        num_attn_heads=2,
        num_kv_heads=2,
    )

    assert decision.kept_block_ids.tolist() == [11]
    assert decision.new_seq_len == 16
    assert decision.groups_total == 2


def test_device_union_strict_keeps_unseen_groups():
    cfg = to_runtime_config(
        EpsConfig(
            enabled=True,
            method="jl",
            scope="union",
            heads="all",
            group_blocks=1,
            last_n=0,
            alpha=1.1,
            strict=True,
        )
    )

    state = _make_state(num_layers=1, num_heads=1, num_groups=2, head_dim=2, sketch_dim=2)
    state.G[0, 0, 1] = torch.eye(2) * 25.0
    state.frob2[0, 0, 1] = 25.0

    q = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    block_ids = torch.tensor([5, 6], dtype=torch.int64)
    block_mapping = {5: 0, 6: 1}

    decision = union_select_for_request(
        cfg=cfg,
        layer_index=0,
        q_attn=q,
        state=state,
        block_ids=block_ids,
        block_mapping=block_mapping,
        seq_len=32,
        block_size=16,
        num_attn_heads=1,
        num_kv_heads=1,
    )

    assert decision.kept_block_ids.tolist() == [5, 6]
    assert decision.new_seq_len == 32
    assert decision.group_keep_mask.tolist() == [True, True]

class _FakeLayer:
    def __init__(self):
        self._eps_layer_name = "layer0"
        self._eps_layer_index = 0
        self._eps_kv_group_id = 0


class _FakeMetadata:
    def __init__(self, block_table, seq_lens, query_start_loc, max_seq_len):
        self.num_actual_tokens = 1
        self.query_start_loc = query_start_loc
        self.seq_lens = seq_lens
        self.max_seq_len = max_seq_len
        self.block_table = block_table
        self.use_cascade = False


def test_apply_device_union_updates_metadata():
    cfg = to_runtime_config(
        EpsConfig(
            enabled=True,
            method="jl",
            scope="union",
            heads="all",
            group_blocks=1,
            last_n=0,
            alpha=1.1,
        )
    )

    state = _make_state(num_layers=1, num_heads=1, num_groups=2, head_dim=2, sketch_dim=2)
    state.G[0, 0, 1] = torch.eye(2) * 64.0
    state.frob2[0, 0, 1] = 64.0

    block_size = 16
    request_states = [[state]]
    request_block_ids = [([0, 1],)]
    ctx = build_eps_forward_context(
        cfg=cfg,
        layer_lookup={"layer0": (0, 0)},
        group_block_sizes=[block_size],
        request_ids=["req0"],
        request_states=request_states,
        request_block_ids=request_block_ids,
        token_request_indices=torch.tensor([0], dtype=torch.int32),
        cudagraph_capture=False,
    )
    ctx.device_union_groups = {0}

    block_table = torch.tensor([[0, 1]], dtype=torch.int32)
    seq_lens = torch.tensor([32], dtype=torch.int32)
    metadata = _FakeMetadata(
        block_table=block_table,
        seq_lens=seq_lens,
        query_start_loc=torch.tensor([0, 1], dtype=torch.int64),
        max_seq_len=32,
    )

    query_tokens = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)

    with eps_context(ctx):
        applied = apply_device_union(
            layer=_FakeLayer(),
            query_tokens=query_tokens,
            num_heads=1,
            num_kv_heads=1,
            attn_metadata=metadata,
        )

    assert applied
    assert metadata.block_table[0, 0].item() == 1
    assert metadata.block_table[0, 1].item() == cfg.sentinel
    assert metadata.seq_lens[0].item() == block_size
    assert metadata.max_seq_len == block_size
    counters = ctx.device_counters
    assert counters is not None
    assert counters.blocks_kept == 1
    assert counters.blocks_total == 2
