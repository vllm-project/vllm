# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CPU tests for vllm/v1/worker/kv_compression.py.

Runs without a full vLLM install (stubs the vllm-internal imports), so the
compression math can be validated on a machine without GPU/vLLM deps:

    python tests/v1/worker/test_kv_compression_standalone.py

When kvpress is importable, the KeyDiff scores and the filtering decision
are additionally cross-checked against kvpress's KeyDiffPress and the
FilteringPress thresholding rule.
"""

import importlib.util
import pathlib
import sys
import types

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "vllm" / "v1" / "worker" / "kv_compression.py"


def _load_module():
    """Load kv_compression.py with vllm-internal imports stubbed out."""
    if "vllm.v1.worker.kv_compression" in sys.modules:
        return sys.modules["vllm.v1.worker.kv_compression"]
    try:
        from vllm.v1.worker import kv_compression

        return kv_compression
    except Exception:
        pass

    logger_mod = types.ModuleType("vllm.logger")

    class _Logger:
        def info(self, *a, **k):
            pass

        def warning(self, *a, **k):
            pass

    logger_mod.init_logger = lambda name: _Logger()

    kci_mod = types.ModuleType("vllm.v1.kv_cache_interface")

    class FullAttentionSpec:  # noqa: D401 - stub
        pass

    class KVCacheConfig:  # noqa: D401 - stub
        pass

    kci_mod.FullAttentionSpec = FullAttentionSpec
    kci_mod.KVCacheConfig = KVCacheConfig

    for name, mod in [
        ("vllm", types.ModuleType("vllm")),
        ("vllm.logger", logger_mod),
        ("vllm.v1", types.ModuleType("vllm.v1")),
        ("vllm.v1.kv_cache_interface", kci_mod),
        ("vllm.v1.worker", types.ModuleType("vllm.v1.worker")),
    ]:
        sys.modules.setdefault(name, mod)
    sys.modules["vllm.logger"] = logger_mod
    sys.modules["vllm.v1.kv_cache_interface"] = kci_mod

    spec = importlib.util.spec_from_file_location(
        "vllm.v1.worker.kv_compression", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["vllm.v1.worker.kv_compression"] = module
    return module


kvc = _load_module()

torch.manual_seed(42)

BLOCK_SIZE = 16
NUM_HEADS = 4
HEAD_SIZE = 32
DEVICE = "cpu"
DTYPE = torch.float32


def make_paged_cache(num_blocks, num_layers=3):
    return [
        torch.randn(2, num_blocks, BLOCK_SIZE, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        for _ in range(num_layers)
    ]


def write_sequence(kv_caches, block_row, keys_per_layer, values_per_layer):
    """Write dense per-layer K/V into the paged cache at positions 0..T-1."""
    seq_len = keys_per_layer[0].shape[0]
    slots = kvc._slots_for_positions(
        block_row, BLOCK_SIZE, seq_len, torch.device(DEVICE)
    )
    for kv_cache, keys, values in zip(kv_caches, keys_per_layer, values_per_layer):
        kvc.scatter_slots(kv_cache[0], slots, keys)
        kvc.scatter_slots(kv_cache[1], slots, values)
    return slots


def test_gather_scatter_roundtrip():
    kv_caches = make_paged_cache(num_blocks=8, num_layers=1)
    block_row = np.array([5, 2, 7, 0], dtype=np.int32)
    seq_len = 37
    keys = torch.randn(seq_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    values = torch.randn(seq_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    slots = write_sequence(kv_caches, block_row, [keys], [values])
    assert torch.equal(kvc.gather_slots(kv_caches[0][0], slots), keys)
    assert torch.equal(kvc.gather_slots(kv_caches[0][1], slots), values)
    print("test_gather_scatter_roundtrip PASSED")


def test_keydiff_scores_match_kvpress():
    try:
        from kvpress import KeyDiffPress
    except ImportError:
        print("test_keydiff_scores_match_kvpress SKIPPED (kvpress not installed)")
        return
    keys = torch.randn(50, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    ours = kvc.keydiff_scores(keys)  # [H, T]
    # kvpress layout: [batch, heads, seq, dim]
    kv_keys = keys.permute(1, 0, 2).unsqueeze(0)
    ref = (
        KeyDiffPress()
        .score(
            module=None,
            hidden_states=None,
            keys=kv_keys,
            values=None,
            attentions=None,
            kwargs={},
        )
        .squeeze(0)
    )  # [H, T]
    torch.testing.assert_close(ours, ref, atol=1e-6, rtol=1e-5)
    print("test_keydiff_scores_match_kvpress PASSED")


def test_compact_request_kv():
    num_layers = 3
    kv_caches = make_paged_cache(num_blocks=16, num_layers=num_layers)
    seq_len = 100
    ratio = 0.5
    block_row = np.array([3, 9, 1, 12, 6, 0, 15], dtype=np.int32)
    keys_pl = [
        torch.randn(seq_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        for _ in range(num_layers)
    ]
    values_pl = [
        torch.randn(seq_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        for _ in range(num_layers)
    ]
    write_sequence(kv_caches, block_row, keys_pl, values_pl)

    n_target = int(seq_len * (1 - ratio))
    n_kept = kvc.compact_request_kv(kv_caches, block_row, BLOCK_SIZE, seq_len, n_target)
    assert n_kept == n_target, n_kept

    kept_slots = kvc._slots_for_positions(
        block_row, BLOCK_SIZE, n_kept, torch.device(DEVICE)
    )
    for layer in range(num_layers):
        keys, values = keys_pl[layer], values_pl[layer]
        # Reference: per-head top-n_kept by KeyDiff score, in temporal order.
        scores = kvc.keydiff_scores(keys)  # [H, T]
        ref_idx = scores.topk(n_kept, dim=-1).indices.sort(dim=-1).values
        got_keys = kvc.gather_slots(kv_caches[layer][0], kept_slots)  # [n, H, D]
        got_values = kvc.gather_slots(kv_caches[layer][1], kept_slots)
        for h in range(NUM_HEADS):
            torch.testing.assert_close(got_keys[:, h], keys[ref_idx[h], h])
            torch.testing.assert_close(got_values[:, h], values[ref_idx[h], h])
    print("test_compact_request_kv PASSED")


def test_compact_noop_when_ratio_keeps_all():
    # A 1-token sequence can never be compacted below 1 token.
    kv_caches = make_paged_cache(num_blocks=1, num_layers=1)
    block_row = np.array([0], dtype=np.int32)
    keys = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    values = torch.randn(1, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    write_sequence(kv_caches, block_row, [keys], [values])
    before = kv_caches[0].clone()
    # n_kept target of 0 is clamped to 1, which keeps everything.
    n_kept = kvc.compact_request_kv(kv_caches, block_row, BLOCK_SIZE, 1, 0)
    assert n_kept == 1
    assert torch.equal(kv_caches[0], before)
    print("test_compact_noop_when_ratio_keeps_all PASSED")


def test_masked_keydiff_scores_match_per_head():
    """masked_keydiff_scores(valid) == keydiff_scores on each head's valid keys."""
    torch.manual_seed(3)
    num_tokens = 30
    keys = torch.randn(num_tokens, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    lengths = torch.tensor([30, 17, 5, 23])
    cols = torch.arange(num_tokens)
    valid = cols.unsqueeze(0) < lengths.unsqueeze(1)
    valid[:, -1] = True  # include the "new token" column

    scores = kvc.masked_keydiff_scores(keys, valid)
    for h in range(NUM_HEADS):
        valid_pos = valid[h].nonzero(as_tuple=True)[0]
        ref = kvc.keydiff_scores(keys[valid_pos, h : h + 1, :]).squeeze(0)
        torch.testing.assert_close(scores[h, valid_pos], ref, atol=1e-6, rtol=1e-5)
        assert (scores[h, ~valid[h]] == float("-inf")).all()
    print("test_masked_keydiff_scores_match_per_head PASSED")


def test_filtering_step_multistep_vs_kvpress():
    """Stateful multi-step equivalence with real kvpress FilteringPress.

    Runs 60 decode steps on 2 independent layers and checks, after every
    step, that our paged-cache filtering produces exactly the same
    per-(layer, head) lengths and the same per-head packed K/V contents as
    kvpress FilteringPress + PaddedTensor (fill_padding=False), and that
    the shared physical length equals the max kvpress buffer length.
    """
    try:
        from types import SimpleNamespace

        from kvpress import FilteringPress, KeyDiffPress
    except ImportError:
        print(
            "test_filtering_step_multistep_vs_kvpress SKIPPED "
            "(kvpress FilteringPress not available)"
        )
        return

    torch.manual_seed(11)
    num_layers = 2
    ratio = 0.5
    prompt_len = 24
    num_steps = 60

    # --- vLLM side: paged cache with prompt written densely.
    kv_caches = make_paged_cache(num_blocks=16, num_layers=num_layers)
    block_row = np.array([3, 9, 1, 12, 6, 0], dtype=np.int32)
    keys_pl = [
        torch.randn(prompt_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        for _ in range(num_layers)
    ]
    values_pl = [
        torch.randn(prompt_len, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        for _ in range(num_layers)
    ]
    write_sequence(kv_caches, block_row, keys_pl, values_pl)
    lengths = torch.full((num_layers, NUM_HEADS), prompt_len, dtype=torch.long)
    num_kv_discarded = 0

    # --- kvpress side: one FilteringPress (keeps per-layer _lengths) and
    # one buffer per layer, [1, H, T, D].
    fp = FilteringPress(
        base_press=KeyDiffPress(),
        target_compression_ratio=ratio,
        fill_padding=False,
    )
    bufs_k = [k.permute(1, 0, 2).unsqueeze(0).clone() for k in keys_pl]
    bufs_v = [v.permute(1, 0, 2).unsqueeze(0).clone() for v in values_pl]

    deltas = []
    for step in range(num_steps):
        logical_total = prompt_len + step + 1
        new_k = [
            torch.randn(NUM_HEADS, HEAD_SIZE, dtype=DTYPE) for _ in range(num_layers)
        ]
        new_v = [
            torch.randn(NUM_HEADS, HEAD_SIZE, dtype=DTYPE) for _ in range(num_layers)
        ]

        # kvpress: append the token to each layer buffer and compress.
        for layer in range(num_layers):
            module = SimpleNamespace(layer_idx=layer, head_dim=HEAD_SIZE)
            keys_in = torch.cat(
                [bufs_k[layer], new_k[layer].unsqueeze(0).unsqueeze(2)], dim=2
            )
            values_in = torch.cat(
                [bufs_v[layer], new_v[layer].unsqueeze(0).unsqueeze(2)], dim=2
            )
            bufs_k[layer], bufs_v[layer] = fp.compress(
                module,
                None,
                keys_in,
                values_in,
                None,
                {"position_ids": torch.arange(logical_total)},
            )

        # vLLM: write the token at the shared cache column, then filter.
        num_cached = logical_total - num_kv_discarded
        new_col_slots = kvc._slots_for_positions(
            block_row, BLOCK_SIZE, num_cached, torch.device(DEVICE)
        )[-1:]
        for layer in range(num_layers):
            kvc.scatter_slots(
                kv_caches[layer][0], new_col_slots, new_k[layer].unsqueeze(0)
            )
            kvc.scatter_slots(
                kv_caches[layer][1], new_col_slots, new_v[layer].unsqueeze(0)
            )
        new_shared_len = kvc.filtering_step(
            kv_caches,
            block_row,
            BLOCK_SIZE,
            lengths,
            num_cached,
            logical_total,
            ratio,
        )
        delta = num_cached - new_shared_len
        assert delta in (0, 1), delta
        deltas.append(delta)
        num_kv_discarded += delta

        # --- equivalence checks after every step.
        for layer in range(num_layers):
            kvpress_lengths = fp._lengths[layer].squeeze(0)
            assert torch.equal(lengths[layer], kvpress_lengths), (
                f"step {step} layer {layer}: {lengths[layer]} vs {kvpress_lengths}"
            )
            max_len = int(lengths[layer].max().item())
            assert bufs_k[layer].shape[2] == max_len
            slots = kvc._slots_for_positions(
                block_row, BLOCK_SIZE, max_len, torch.device(DEVICE)
            )
            got_k = kvc.gather_slots(kv_caches[layer][0], slots)  # [C, H, D]
            got_v = kvc.gather_slots(kv_caches[layer][1], slots)
            for h in range(NUM_HEADS):
                head_len = int(lengths[layer][h].item())
                torch.testing.assert_close(
                    got_k[:head_len, h], bufs_k[layer][0, h, :head_len]
                )
                torch.testing.assert_close(
                    got_v[:head_len, h], bufs_v[layer][0, h, :head_len]
                )
        assert new_shared_len == max(
            bufs_k[layer].shape[2] for layer in range(num_layers)
        )

    # Behavioral sanity: both keeps and skips occurred, heads diverged, and
    # the realized shared length is meaningfully below the logical length.
    assert 0 < sum(deltas) < num_steps, deltas
    assert lengths.min() < lengths.max()
    logical_final = prompt_len + num_steps
    shared_final = logical_final - num_kv_discarded
    assert shared_final < logical_final
    print(
        "test_filtering_step_multistep_vs_kvpress PASSED "
        f"(skipped {sum(deltas)}/{num_steps} columns, shared len "
        f"{shared_final}/{logical_final}, per-head lengths {lengths.tolist()})"
    )


def test_filtering_step_keeps_all_when_under_target():
    """If every head has fewer valid tokens than n_kept, all heads accept."""
    kv_caches = make_paged_cache(num_blocks=4, num_layers=1)
    block_row = np.array([0, 1, 2, 3], dtype=np.int32)
    num_cached = 10
    keys = torch.randn(num_cached, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    values = torch.randn(num_cached, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
    write_sequence(kv_caches, block_row, [keys], [values])
    lengths = torch.full((1, NUM_HEADS), num_cached - 1, dtype=torch.long)
    # logical_total large => n_kept >= num_cached => threshold is the
    # minimum valid score => every head accepts, no column is freed.
    new_len = kvc.filtering_step(
        kv_caches, block_row, BLOCK_SIZE, lengths, num_cached, 30, 0.5
    )
    assert new_len == num_cached
    assert (lengths == num_cached).all()
    print("test_filtering_step_keeps_all_when_under_target PASSED")


def test_filtering_realized_ratio_converges():
    """Long-run realized compression tracks the target (in expectation).

    Mirrors the setup of 05a_filtering_bias_analysis.md (KeyDiff scoring on
    random keys, total_tokens_seen + round): per-head keep rate should sit
    near (1 - ratio). The shared physical length is the max over heads, so
    the memory ratio sits somewhat above the per-head ratio.
    """
    torch.manual_seed(5)
    ratio = 0.5
    prompt_len = 16
    num_steps = 1200
    head_size = 16

    kv_caches = [torch.randn(2, 128, BLOCK_SIZE, NUM_HEADS, head_size, dtype=DTYPE)]
    block_row = np.arange(128, dtype=np.int32)
    keys = torch.randn(prompt_len, NUM_HEADS, head_size, dtype=DTYPE)
    values = torch.randn(prompt_len, NUM_HEADS, head_size, dtype=DTYPE)
    slots = kvc._slots_for_positions(
        block_row, BLOCK_SIZE, prompt_len, torch.device(DEVICE)
    )
    kvc.scatter_slots(kv_caches[0][0], slots, keys)
    kvc.scatter_slots(kv_caches[0][1], slots, values)

    lengths = torch.full((1, NUM_HEADS), prompt_len, dtype=torch.long)
    num_kv_discarded = 0
    for step in range(num_steps):
        logical_total = prompt_len + step + 1
        num_cached = logical_total - num_kv_discarded
        col_slots = kvc._slots_for_positions(
            block_row, BLOCK_SIZE, num_cached, torch.device(DEVICE)
        )[-1:]
        kvc.scatter_slots(
            kv_caches[0][0],
            col_slots,
            torch.randn(1, NUM_HEADS, head_size, dtype=DTYPE),
        )
        kvc.scatter_slots(
            kv_caches[0][1],
            col_slots,
            torch.randn(1, NUM_HEADS, head_size, dtype=DTYPE),
        )
        new_len = kvc.filtering_step(
            kv_caches, block_row, BLOCK_SIZE, lengths, num_cached, logical_total, ratio
        )
        num_kv_discarded += num_cached - new_len

    logical_final = prompt_len + num_steps
    per_head_ratio = lengths.float().mean().item() / logical_final
    shared_ratio = (logical_final - num_kv_discarded) / logical_final
    # Loose bands: Monte Carlo with a single seed.
    assert 0.35 < per_head_ratio < 0.65, per_head_ratio
    assert 0.35 < shared_ratio < 0.75, shared_ratio
    print(
        "test_filtering_realized_ratio_converges PASSED "
        f"(per-head keep ratio {per_head_ratio:.3f}, "
        f"shared/memory ratio {shared_ratio:.3f}, target keep {1 - ratio})"
    )


def test_iterative_compaction():
    """Repeated retroactive compaction across growth phases.

    Mirrors kvpress CompressionRatioDecodingPress: each compaction keeps
    the per-head top int(logical_total * (1 - ratio)) of the *current*
    cache (which already holds composite per-head survivors of earlier
    compactions). Verified against an independently maintained per-head
    reference list.
    """
    torch.manual_seed(21)
    ratio = 0.5
    kv_caches = make_paged_cache(num_blocks=16, num_layers=1)
    block_row = np.arange(16, dtype=np.int32)
    device = torch.device(DEVICE)

    # Independent reference: per-head lists of (key, value) rows.
    ref_k = [[] for _ in range(NUM_HEADS)]
    ref_v = [[] for _ in range(NUM_HEADS)]
    num_cached = 0
    logical_total = 0

    def append_tokens(n):
        nonlocal num_cached, logical_total
        keys = torch.randn(n, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        values = torch.randn(n, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
        slots = kvc._slots_for_positions(block_row, BLOCK_SIZE, num_cached + n, device)[
            num_cached:
        ]
        kvc.scatter_slots(kv_caches[0][0], slots, keys)
        kvc.scatter_slots(kv_caches[0][1], slots, values)
        for h in range(NUM_HEADS):
            for i in range(n):
                ref_k[h].append(keys[i, h])
                ref_v[h].append(values[i, h])
        num_cached += n
        logical_total += n

    def compact_and_check():
        nonlocal num_cached
        n_target = max(1, int(logical_total * (1 - ratio)))
        n_kept = kvc.compact_request_kv(
            kv_caches, block_row, BLOCK_SIZE, num_cached, n_target
        )
        # Reference: per-head KeyDiff top-n over the head's current list,
        # kept in temporal order.
        for h in range(NUM_HEADS):
            head_keys = torch.stack(ref_k[h]).unsqueeze(1)  # [T, 1, D]
            scores = kvc.keydiff_scores(head_keys).squeeze(0)  # [T]
            kept = scores.topk(n_kept).indices.sort().values.tolist()
            ref_k[h] = [ref_k[h][i] for i in kept]
            ref_v[h] = [ref_v[h][i] for i in kept]
        num_cached = n_kept
        slots = kvc._slots_for_positions(block_row, BLOCK_SIZE, n_kept, device)
        got_k = kvc.gather_slots(kv_caches[0][0], slots)
        got_v = kvc.gather_slots(kv_caches[0][1], slots)
        for h in range(NUM_HEADS):
            torch.testing.assert_close(got_k[:, h], torch.stack(ref_k[h]))
            torch.testing.assert_close(got_v[:, h], torch.stack(ref_v[h]))
        return n_kept

    # Prefill chunk 1 + chunk 2 (continuation), compaction after each —
    # block-wise iterative compression — then decode with periodic
    # compaction.
    append_tokens(30)
    compact_and_check()  # logical 30 -> keep 15
    append_tokens(30)  # continuation chunk: cache 45, logical 60
    compact_and_check()  # keep 30
    for _ in range(3):
        append_tokens(8)  # decode growth
        compact_and_check()
    assert logical_total == 84
    assert num_cached == max(1, int(84 * (1 - ratio)))
    print(
        "test_iterative_compaction PASSED "
        f"(logical {logical_total}, cached {num_cached})"
    )


class _FakeBlockTable:
    def __init__(self, block_table_np, block_size):
        self.block_table = types.SimpleNamespace(np=block_table_np)
        self.block_size = block_size


class _FakeInputBatch:
    def __init__(self, req_ids, num_computed, block_table_np, block_size):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = np.array(num_computed, dtype=np.int32)
        self._bt = _FakeBlockTable(block_table_np, block_size)

    @property
    def block_table(self):
        return [self._bt]


def _fake_req_state(prompt_len):
    return types.SimpleNamespace(
        num_prompt_tokens=prompt_len,
        num_kv_discarded=0,
        kv_compressed=False,
        kv_filter_lengths=None,
        kv_last_compaction_total=0,
    )


def _run_manager_step(mgr, req_state, block_row, computed_before, scheduled):
    input_batch = _FakeInputBatch(
        ["req"], [computed_before], block_row.reshape(1, -1), BLOCK_SIZE
    )
    scheduler_output = types.SimpleNamespace(num_scheduled_tokens={"req": scheduled})
    return mgr.run_post_forward(input_batch, {"req": req_state}, scheduler_output)


def test_manager_phase_coverage():
    """Both algorithms cover prefill, continuation (chunked prefill) and
    decode through KVCompressionManager's trigger logic."""
    from vllm.v1.worker.kv_compression import KVCompressionManager

    torch.manual_seed(33)
    ratio = 0.5
    interval = 16
    prompt_len = 48  # three chunks of 16

    def write_tokens(kv_caches, block_row, req_state, computed_before, n):
        start = computed_before - req_state.num_kv_discarded
        slots = kvc._slots_for_positions(
            block_row, BLOCK_SIZE, start + n, torch.device(DEVICE)
        )[start:]
        for kv_cache in kv_caches:
            kvc.scatter_slots(
                kv_cache[0], slots, torch.randn(n, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
            )
            kvc.scatter_slots(
                kv_cache[1], slots, torch.randn(n, NUM_HEADS, HEAD_SIZE, dtype=DTYPE)
            )

    for algorithm in ("full_replacement", "filtering"):
        mgr = KVCompressionManager(
            algorithm=algorithm,
            compression_ratio=ratio,
            compression_interval=interval,
        )
        mgr.kv_caches = make_paged_cache(num_blocks=16, num_layers=2)
        block_row = np.arange(16, dtype=np.int32)
        req_state = _fake_req_state(prompt_len)

        # Prefill chunk 1 (tokens 0..15): interval reached -> compaction.
        write_tokens(mgr.kv_caches, block_row, req_state, 0, 16)
        out = _run_manager_step(mgr, req_state, block_row, 0, 16)
        assert out == {"req": 8}, (algorithm, out)  # 16 -> keep 8

        # Prefill chunk 2 (continuation, tokens 16..31): compaction again.
        write_tokens(mgr.kv_caches, block_row, req_state, 16, 16)
        out = _run_manager_step(mgr, req_state, block_row, 16, 16)
        # cache was 8 + 16 = 24, keep int(32 * 0.5) = 16 -> discard 8
        assert out == {"req": 8}, (algorithm, out)

        # Final prefill chunk (tokens 32..47): unconditional compaction at
        # prefill completion. cache 16 + 16 = 32, keep 24 -> discard 8.
        write_tokens(mgr.kv_caches, block_row, req_state, 32, 16)
        out = _run_manager_step(mgr, req_state, block_row, 32, 16)
        assert out == {"req": 8}, (algorithm, out)
        assert req_state.num_kv_discarded == 24

        # Decode steps.
        deltas = []
        for step in range(interval + 1):
            computed_before = prompt_len + step
            write_tokens(mgr.kv_caches, block_row, req_state, computed_before, 1)
            out = _run_manager_step(mgr, req_state, block_row, computed_before, 1)
            deltas.append(out.get("req", 0))

        if algorithm == "full_replacement":
            # No compaction until the interval elapses (logical 64 at
            # decode step 15), then one batch drop: cache 24 + 16 = 40,
            # keep int(64 * 0.5) = 32 -> discard 8.
            expected = [0] * len(deltas)
            expected[15] = 8
            assert deltas == expected, deltas
            assert req_state.kv_filter_lengths is None
        else:
            # Per-head filtering runs every decode step.
            assert req_state.kv_filter_lengths is not None
            assert all(d in (0, 1) for d in deltas), deltas
            assert 0 < sum(deltas) <= len(deltas), deltas
        print(f"test_manager_phase_coverage[{algorithm}] PASSED (deltas {deltas})")


if __name__ == "__main__":
    test_gather_scatter_roundtrip()
    test_keydiff_scores_match_kvpress()
    test_compact_request_kv()
    test_compact_noop_when_ratio_keeps_all()
    test_masked_keydiff_scores_match_per_head()
    test_filtering_step_multistep_vs_kvpress()
    test_filtering_step_keeps_all_when_under_target()
    test_filtering_realized_ratio_converges()
    test_iterative_compaction()
    test_manager_phase_coverage()
    print("ALL TESTS PASSED")
