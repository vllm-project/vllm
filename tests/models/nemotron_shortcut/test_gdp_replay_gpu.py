# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDP replay vs the existing CAKE decode path on B200 / B300.

Both paths start from the same post-prefill state (checkpoint and conv taps)
and run the same MTP verify windows with random acceptance, enough steps for
several checkpoint flushes. Each step's GDP output must match within the FP16
checkpoint's tolerance. Two more tests start replay from the real prefill,
on one rank and from a TP1 prefill into TP2 decode ranks, against an FP32
prefill over the whole sequence. Needs FlashInfer with `gdp_conv_replay_decode`
and `cake_gdp`.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CacheConfig, get_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.models.nemotron_shortcut.nvidia.gdp import (
    NemotronShortcutGDPMixer,
    gdp_replay_state_dtype,
    gdp_replay_state_shape,
    gdp_state_dtype,
    gdp_state_shape,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_default_torch_dtype

from .utils import make_config

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="gdp_conv_replay runs on sm_100a / sm_103a",
)

HEADS, NUM_SPEC, BATCH, STEPS = 6, 3, 4, 32
TOKENS = NUM_SPEC + 1


@pytest.fixture
def forward_context(dist_init):
    pytest.importorskip("flashinfer.gdp_conv_replay_decode")
    pytest.importorskip("flashinfer.cake_gdp")
    with torch.inference_mode(), set_forward_context(None, get_current_vllm_config()):
        yield


def _mixer(replay: bool, prefix: str, heads: int = HEADS) -> NemotronShortcutGDPMixer:
    config = make_config(
        hidden_size=64,
        layer_types=["linear_attention"],
        gdp_num_heads=heads,
        gdp_num_groups=heads,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        gdp_num_householder=3,
        gdp_conv_kernel=4,
        gdp_norm_eps=1e-5,
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        mixer = NemotronShortcutGDPMixer(
            config, cache_config=CacheConfig(use_replayssm=replay), prefix=prefix
        )
    mixer.num_spec = NUM_SPEC
    mixer.replay_capture_sizes = []
    return mixer


def _caches(
    replay: bool, num_blocks: int, heads: int = HEADS
) -> tuple[torch.Tensor, ...]:
    args = (1, heads, heads, 128, 64, 4, 3, NUM_SPEC)
    if replay:
        shapes = gdp_replay_state_shape(*args)
        dtypes = gdp_replay_state_dtype(torch.bfloat16, "auto")
    else:
        shapes = gdp_state_shape(*args)
        dtypes = gdp_state_dtype(torch.bfloat16, "auto")
    return tuple(
        torch.zeros(num_blocks, *shape, dtype=dtype, device="cuda")
        for shape, dtype in zip(shapes, dtypes)
    )


def _metadata(blocks: torch.Tensor, num_accepted: torch.Tensor) -> SimpleNamespace:
    """Uniform verify windows of TOKENS tokens; `blocks` is [BATCH, TOKENS]."""
    tokens = BATCH * TOKENS
    return SimpleNamespace(
        num_actual_tokens=tokens,
        spec_sequence_masks=torch.ones(BATCH, dtype=torch.bool, device="cuda"),
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=BATCH,
        spec_token_indx=torch.arange(tokens, device="cuda"),
        non_spec_token_indx=torch.empty(0, dtype=torch.long, device="cuda"),
        spec_state_indices_tensor=blocks,
        spec_query_start_loc=torch.arange(
            0, tokens + 1, TOKENS, dtype=torch.int32, device="cuda"
        ),
        num_accepted_tokens=num_accepted,
    )


def _returning(metadata: SimpleNamespace):
    return lambda: metadata


def test_replay_matches_the_cake_decode_path(forward_context):
    torch.manual_seed(0)
    cake, replay = _mixer(False, "gdp_cake"), _mixer(True, "model.layers.0.gdp")
    conv_weight = torch.randn_like(cake.conv1d.weight) * 0.3
    a_log = torch.log(torch.empty(HEADS, device="cuda").uniform_(1, 16))
    dt_bias = torch.randn(HEADS, device="cuda") * 0.5
    for mixer in (cake, replay):
        mixer.conv1d.weight.data.copy_(conv_weight)
        mixer.A_log.data.copy_(a_log)
        mixer.dt_bias.data.copy_(dt_bias)

    # CAKE keeps one block per window token; replay one block per request.
    cake_blocks = 1 + torch.arange(BATCH * TOKENS, dtype=torch.int32, device="cuda")
    cake_blocks = cake_blocks.view(BATCH, TOKENS)
    replay_blocks = 1 + torch.arange(BATCH, dtype=torch.int32, device="cuda")
    replay_blocks = replay_blocks.unsqueeze(1).repeat(1, TOKENS)
    cake.kv_cache = _caches(False, 1 + BATCH * TOKENS)
    replay.kv_cache = _caches(True, 1 + BATCH)
    assert replay.gdp_replay_trackers is not None
    replay.gdp_replay_trackers.bind(replay.kv_cache[0])

    # The post-prefill state both paths start from.
    checkpoint = (torch.randn(BATCH, HEADS, 64, 128, device="cuda") * 0.05).half()
    taps = torch.randn(BATCH, 3, cake.conv_dim, device="cuda").to(torch.bfloat16)
    cake.kv_cache[1][cake_blocks[:, 0].long()] = checkpoint.float()
    cake.kv_cache[0][cake_blocks[:, 0].long(), :3] = taps
    replay.kv_cache[1][replay_blocks[:, 0].long()] = checkpoint
    replay.kv_cache[0][replay_blocks[:, 0].long(), :3] = taps

    width = HEADS * (64 + 704 + 4)
    num_accepted = torch.ones(BATCH, dtype=torch.int32, device="cuda")
    trackers = replay.gdp_replay_trackers
    assert trackers is not None
    flushes, bank = 0, 0
    for step in range(STEPS):
        projected = torch.randn(BATCH * TOKENS, width, device="cuda").to(torch.bfloat16)
        _, mixed_vkq, b, a = projected.split(
            [cake.value_dim, cake.conv_dim, 3 * HEADS, HEADS], dim=-1
        )
        cake_out = torch.zeros(BATCH * TOKENS, HEADS, 64, dtype=torch.bfloat16)
        cake_out = cake_out.cuda()
        replay_out = torch.zeros_like(cake_out)

        cake_md = _metadata(cake_blocks, num_accepted)
        cake._gdn_metadata = _returning(cake_md)
        # The conv update writes its output into its input, so CAKE gets its own
        # copy of the conv columns: `projected` is replay's input too.
        cake._forward(mixed_vkq.clone(), b.clone(), a.clone(), cake_out)
        replay_md = _metadata(replay_blocks, num_accepted)
        replay._gdn_metadata = _returning(replay_md)
        replay._forward_replay(projected, replay_out)

        error = (replay_out.float() - cake_out.float()).norm()
        assert error <= 3e-2 * cake_out.float().norm(), f"step {step}: {error}"
        num_accepted = torch.randint(
            1, TOKENS + 1, (BATCH,), dtype=torch.int32, device="cuda"
        )
        # A flush moves the history to the other bank.
        flushes += int(trackers.ring_bank[1]) != bank
        bank = int(trackers.ring_bank[1])

    assert flushes >= 3


# ---------------------------------------------------------------------------
# Prefill -> replay handoff, on one rank and across a prefill / decode TP change.

PROMPT_LENS = (2, 5, 17, 33)  # 2 < conv taps: the prompt's missing taps are 0
HANDOFF_STEPS = 8


def _width(heads: int) -> int:
    """One in_proj row: z | conv (V3 | K3 | Q) | b | a."""
    return heads * (64 + 704 + 4)


def _shared_weights(heads: int) -> dict[str, torch.Tensor]:
    conv_dim = heads * 704
    return {
        "conv": torch.randn(conv_dim, 1, 4, device="cuda") * 0.3,
        "a_log": torch.log(torch.empty(heads, device="cuda").uniform_(1, 16)),
        "dt_bias": torch.randn(heads, device="cuda") * 0.5,
    }


def _load(mixer: NemotronShortcutGDPMixer, weights: dict[str, torch.Tensor]) -> None:
    mixer.conv1d.weight.data.copy_(weights["conv"])
    mixer.A_log.data.copy_(weights["a_log"])
    mixer.dt_bias.data.copy_(weights["dt_bias"])


def _head_conv_channels(heads: int, lo: int, hi: int) -> torch.Tensor:
    """Conv channels of heads [lo, hi) in the householder-major order
    V [3, H, 64] | K [3, H, 128] | Q [H, 128]: seven runs, as the head-wise
    in_proj loader gives a TP rank and FlashInfer reads a head."""
    runs = [((u * heads + lo) * 64, (u * heads + hi) * 64) for u in range(3)]
    k0 = 3 * 64 * heads
    runs += [
        (k0 + (u * heads + lo) * 128, k0 + (u * heads + hi) * 128) for u in range(3)
    ]
    q0 = k0 + 3 * 128 * heads
    runs.append((q0 + lo * 128, q0 + hi * 128))
    return torch.cat([torch.arange(a, b) for a, b in runs]).cuda()


def _head_row_columns(heads: int, lo: int, hi: int) -> torch.Tensor:
    """In_proj row columns a TP rank holding heads [lo, hi) computes."""
    value_dim, conv_dim = heads * 64, heads * 704
    gates = value_dim + conv_dim
    parts = [
        torch.arange(lo * 64, hi * 64, device="cuda"),
        value_dim + _head_conv_channels(heads, lo, hi),
    ]
    parts += [
        torch.arange(gates + u * heads + lo, gates + u * heads + hi, device="cuda")
        for u in range(3)
    ]
    parts.append(
        torch.arange(gates + 3 * heads + lo, gates + 3 * heads + hi, device="cuda")
    )
    return torch.cat(parts)


def _sequences(heads: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-request prompts, and per-step verify windows of TOKENS tokens."""

    def rows(n: int) -> torch.Tensor:
        return torch.randn(n, _width(heads), device="cuda").to(torch.bfloat16)

    prompts = [rows(n) for n in PROMPT_LENS]
    steps = [rows(len(PROMPT_LENS) * TOKENS) for _ in range(HANDOFF_STEPS)]
    return prompts, steps


def _loc(lens) -> torch.Tensor:
    loc = torch.zeros(len(lens) + 1, dtype=torch.int32, device="cuda")
    loc[1:] = torch.tensor(lens, device="cuda").cumsum(0)
    return loc


def _split_row(mixer: NemotronShortcutGDPMixer, rows: torch.Tensor):
    _, conv, b, a = rows.split(
        [mixer.value_dim, mixer.conv_dim, 3 * mixer.num_heads, mixer.num_heads],
        dim=-1,
    )
    return conv, b, a


def _reference(
    tag: str, heads: int, weights, prompts, steps
) -> list[list[torch.Tensor]]:
    """FP32-state prefill over each request's whole sequence (prompt, then every
    window fully accepted): the outputs every step's window must reproduce."""
    ref = _mixer(False, f"gdp_ref_{tag}", heads)
    _load(ref, weights)
    batch = len(prompts)
    ref.kv_cache = _caches(False, 1 + batch, heads)
    seqs = [
        torch.cat([p] + [s[r * TOKENS : (r + 1) * TOKENS] for s in steps])
        for r, p in enumerate(prompts)
    ]
    conv, b, a = _split_row(ref, torch.cat(seqs))
    out = ref._prefill(
        conv.contiguous(),
        b.contiguous(),
        a.contiguous(),
        ref.conv1d.weight.view(ref.conv_dim, 4),
        ref.kv_cache[0].transpose(-1, -2),
        ref.kv_cache[1],
        _loc([len(s) for s in seqs]),
        1 + torch.arange(batch, dtype=torch.int32, device="cuda"),
        torch.zeros(batch, dtype=torch.bool, device="cuda"),
        None,
    ).view(-1, heads, 64)
    loc = _loc([len(s) for s in seqs]).tolist()
    windows = []
    for step in range(len(steps)):
        windows.append(
            [
                out[
                    loc[r] + len(p) + step * TOKENS : loc[r]
                    + len(p)
                    + (step + 1) * TOKENS
                ]
                for r, p in enumerate(prompts)
            ]
        )
    return windows


def _replay_mixer(
    heads: int, prefix: str, weights, batch: int
) -> NemotronShortcutGDPMixer:
    name = f"model.layers.0.{prefix}"
    mixer = _mixer(True, name, heads)
    _load(mixer, weights)
    mixer.kv_cache = _caches(True, 1 + batch, heads)
    assert mixer.gdp_replay_trackers is not None
    mixer.gdp_replay_trackers.bind(mixer.kv_cache[0])
    return mixer


def _decode(mixer: NemotronShortcutGDPMixer, steps) -> tuple[list[torch.Tensor], int]:
    """Run every window through the replay op, all tokens accepted. Also
    returns how often block 1 flushed (its ring bank switched)."""
    batch = len(PROMPT_LENS)
    blocks = 1 + torch.arange(batch, dtype=torch.int32, device="cuda")
    blocks = blocks.unsqueeze(1).repeat(1, TOKENS)
    accepted = torch.full((batch,), TOKENS, dtype=torch.int32, device="cuda")
    outs, flushes, bank = [], 0, 0
    trackers = mixer.gdp_replay_trackers
    assert trackers is not None
    for rows in steps:
        out = torch.zeros(batch * TOKENS, mixer.num_heads, 64, dtype=torch.bfloat16)
        out = out.cuda()
        mixer._gdn_metadata = _returning(_metadata(blocks, accepted))
        mixer._forward_replay(rows.clone(), out)
        outs.append(out)
        flushes += int(trackers.ring_bank[1]) != bank
        bank = int(trackers.ring_bank[1])
    return outs, flushes


def _assert_windows_match(outs, windows, heads_slice=slice(None)) -> None:
    for step, (out, ref) in enumerate(zip(outs, windows)):
        expected = torch.cat(ref)[:, heads_slice].float()
        error = (out.float() - expected).norm()
        assert error <= 3e-2 * expected.norm(), f"step {step}: {error}"


def test_replay_continues_a_real_prefill(forward_context):
    """The regular prefill leaves the last three prompt inputs in conv rows
    0..2 and the post-prompt checkpoint; replay must continue from exactly
    that, across several flushes."""
    torch.manual_seed(1)
    batch = len(PROMPT_LENS)
    weights = _shared_weights(HEADS)
    prompts, steps = _sequences(HEADS)
    windows = _reference("handoff", HEADS, weights, prompts, steps)

    replay = _replay_mixer(HEADS, "gdp_handoff", weights, batch)
    replay._replay_prefill(
        torch.cat(prompts).clone(),
        _loc(PROMPT_LENS),
        1 + torch.arange(batch, dtype=torch.int32, device="cuda"),
        torch.zeros(batch, dtype=torch.bool, device="cuda"),
        None,
        commits=True,
    )
    trackers = replay.gdp_replay_trackers
    assert trackers is not None
    assert trackers.hist_len[1 : 1 + batch].tolist() == [0] * batch

    outs, flushes = _decode(replay, steps)
    _assert_windows_match(outs, windows)
    # 8 windows of 4 tokens, flushing at hist_len >= 9.
    assert flushes >= 2


@pytest.mark.parametrize("naive_cut", [False, True])
def test_replay_decodes_a_prefill_from_a_different_tp(forward_context, naive_cut):
    """P/D with prefill at TP1 and decode at TP2. A decode rank must receive its
    heads' conv channels: seven runs of the householder-major layout, not one
    contiguous slice. The checkpoint splits by heads. Each decode rank starts
    with zeroed replay trackers, as a freshly prefilled block does."""
    torch.manual_seed(2)
    heads, ranks, batch = 4, 2, len(PROMPT_LENS)  # 2 heads per decode rank
    local = heads // ranks
    weights = _shared_weights(heads)
    prompts, steps = _sequences(heads)
    tag = "naive" if naive_cut else "heads"
    windows = _reference(tag, heads, weights, prompts, steps)

    prefill = _replay_mixer(heads, f"gdp_prefill_tp1_{tag}", weights, batch)
    blocks = 1 + torch.arange(batch, dtype=torch.int32, device="cuda")
    prefill._replay_prefill(
        torch.cat(prompts).clone(),
        _loc(PROMPT_LENS),
        blocks,
        torch.zeros(batch, dtype=torch.bool, device="cuda"),
        None,
        commits=True,
    )
    taps = prefill.kv_cache[0][blocks.long(), :3]  # [batch, 3, 704 * heads]
    checkpoint = prefill.kv_cache[1][blocks.long()]  # [batch, heads, 64, 128]

    for rank in range(ranks):
        lo, hi = rank * local, (rank + 1) * local
        channels = _head_conv_channels(heads, lo, hi)
        rank_weights = {
            "conv": weights["conv"][channels],
            "a_log": weights["a_log"][lo:hi],
            "dt_bias": weights["dt_bias"][lo:hi],
        }
        decode = _replay_mixer(local, f"gdp_decode_r{rank}_{tag}", rank_weights, batch)
        sent = channels
        if naive_cut:
            width = local * 704
            sent = torch.arange(rank * width, (rank + 1) * width, device="cuda")
        # Only rows 0..2 are sent. Poison the rest of the decode ring: the op
        # must write those rows before it reads them.
        decode.kv_cache[0][blocks.long(), 3:] = float("nan")
        decode.kv_cache[0][blocks.long(), :3] = taps[:, :, sent]
        decode.kv_cache[1][blocks.long()] = checkpoint[:, lo:hi]

        columns = _head_row_columns(heads, lo, hi)
        outs, _ = _decode(decode, [s[:, columns] for s in steps])
        if naive_cut:
            with pytest.raises(AssertionError):
                _assert_windows_match(outs, windows, slice(lo, hi))
        else:
            _assert_windows_match(outs, windows, slice(lo, hi))
