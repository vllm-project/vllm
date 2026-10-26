# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLM-5.3 low-latency switch (vllm/models/glm53/nvidia/switch.py)."""

import subprocess
import sys
from types import SimpleNamespace

import pytest

import vllm.platforms
from vllm.config import CUDAGraphMode
from vllm.models.glm53.nvidia import switch


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.setattr(switch, "_resolved", None)
    return monkeypatch


def _config(
    tp=8,
    pp=1,
    kv="fp8",
    backend="FLASHINFER_MLA_SPARSE",
    v2=True,
    seqs=1,
    ep=False,
    dp=1,
    dcp=1,
    pcp=1,
    ubatching=False,
    lora=None,
    kv_transfer=None,
    moe="flashinfer_trtllm",
    sizes=None,
    drafts=3,
    eager=False,
    graphs=CUDAGraphMode.FULL_DECODE_ONLY,
    sleep=False,
    method="mtp",
    per_batch_size=None,
    adaptive=False,
    local_argmax=True,
):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            enable_expert_parallel=ep,
            data_parallel_size=dp,
            decode_context_parallel_size=dcp,
            prefill_context_parallel_size=pcp,
            use_ubatching=ubatching,
        ),
        model_config=SimpleNamespace(enforce_eager=eager, enable_sleep_mode=sleep),
        cache_config=SimpleNamespace(cache_dtype=kv),
        attention_config=SimpleNamespace(backend=SimpleNamespace(name=backend)),
        scheduler_config=SimpleNamespace(max_num_seqs=seqs),
        lora_config=lora,
        kv_transfer_config=kv_transfer,
        kernel_config=SimpleNamespace(moe_backend=moe),
        use_v2_model_runner=v2,
        speculative_config=SimpleNamespace(
            num_speculative_tokens=drafts,
            use_local_argmax_reduction=local_argmax,
            moe_backend=None,
            method=method,
            enable_adaptive_verification=adaptive,
            uses_dynamic_speculative_decoding=lambda: per_batch_size is not None,
        ),
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=sizes,
            max_cudagraph_capture_size=None,
            cudagraph_mode=graphs,
        ),
    )


def test_off_by_default(clean_env):
    assert switch.from_env() == switch.LowLatencyDecode()
    assert not switch.current().enabled


def test_resolve_off_needs_no_collective(clean_env):
    record = switch.resolve(_config(tp=4, kv="auto", v2=False))
    assert record == switch.LowLatencyDecode()
    assert switch.current() is record


def test_unsupported_lists_every_reason(clean_env):
    missing = switch._unsupported(_config(tp=4, kv="auto", v2=False))
    joined = "; ".join(missing)
    for need in ("tensor parallel size 8", "FLASHINFER_MLA_SPARSE", "Model Runner V2"):
        assert need in joined


@pytest.mark.parametrize(
    ("override", "need"),
    [
        ({"seqs": 2}, "--max-num-seqs 1"),
        ({"ep": True}, "no expert parallelism"),
        ({"dp": 2}, "no data, prefill or decode context parallelism"),
        ({"dcp": 2}, "no data, prefill or decode context parallelism"),
        ({"pcp": 2}, "no data, prefill or decode context parallelism"),
        ({"ubatching": True}, "or micro-batching"),
        ({"eager": True}, "without --enforce-eager"),
        ({"graphs": CUDAGraphMode.NONE}, "FULL CUDA graphs for decode"),
        ({"graphs": CUDAGraphMode.PIECEWISE}, "FULL CUDA graphs for decode"),
        ({"graphs": CUDAGraphMode.FULL}, "FULL CUDA graphs for decode"),
        ({"sleep": True}, "no sleep mode"),
        ({"method": "eagle3"}, "MTP speculative decoding"),
        ({"method": "dflash"}, "MTP speculative decoding"),
        ({"per_batch_size": [(1, 4, 3)]}, "static draft count"),
        ({"adaptive": True}, "without adaptive verification"),
        ({"lora": object()}, "no LoRA"),
        ({"kv_transfer": object()}, "no KV transfer connector"),
        ({"moe": "deep_gemm"}, "--moe-backend flashinfer_trtllm or auto"),
        ({"moe": "triton"}, "--moe-backend flashinfer_trtllm or auto"),
    ],
)
def test_refuses_unservable_deployments(clean_env, override, need):
    assert not any(need in m for m in switch._unsupported(_config()))
    assert any(need in m for m in switch._unsupported(_config(**override)))


def test_full_and_piecewise_graphs_are_supported(clean_env):
    config = _config(graphs=CUDAGraphMode.FULL_AND_PIECEWISE)
    assert not any("CUDA graphs" in m for m in switch._unsupported(config))


def test_cute_drafter_needs_local_argmax_reduction(clean_env):
    need = (
        "MTP with use_local_argmax_reduction (the drafter hands over its greedy drafts)"
    )
    assert need not in switch._unsupported(_config())
    assert need in switch._unsupported(_config(local_argmax=False))
    assert need in switch._unsupported(_config(local_argmax=False, seqs=3), (4, 8, 16))
    for drafts in (1, 2):  # the CuTe drafter's greedy drafts at every K
        config = _config(local_argmax=False, seqs=2, drafts=drafts)
        assert need in switch._unsupported(config, (4, 8))


def test_refuses_batch_invariant(clean_env):
    clean_env.setenv("VLLM_BATCH_INVARIANT", "1")
    assert any("batch-invariant" in m for m in switch._unsupported(_config()))


@pytest.mark.parametrize(
    ("kv", "backend", "supported"),
    [
        ("fp8", "FLASHINFER_MLA_SPARSE", True),
        ("fp8_e4m3", "FLASHINFER_MLA_SPARSE", True),
        ("fp8_ds_mla", "FLASHMLA_SPARSE", False),  # the removed 656-byte rows
        ("fp8", "FLASHMLA_SPARSE", False),
    ],
)
def test_only_flashinfer_fp8_rows(clean_env, kv, backend, supported):
    missing = switch._unsupported(_config(kv=kv, backend=backend))
    assert any("--kv-cache-dtype fp8" in need for need in missing) != supported


def test_latency_specialized_turns_on_the_path(clean_env):
    config = _config()
    config.additional_config = {"decode_backend": "latency_specialized"}
    assert switch.from_config(config).enabled
    config.additional_config = {}
    assert switch.from_config(config) == switch.LowLatencyDecode()
    config.additional_config = {"decode_backend": "cute"}
    with pytest.raises(ValueError, match="latency_specialized"):
        switch.from_config(config)


DEFAULT_SIZES = [1, 2, 4, 8, 16, 24, 32]  # vLLM's generated list: it starts with a 1


@pytest.mark.parametrize(
    ("seqs", "sizes", "drafts", "tiers"),
    [
        (1, None, 3, ()),  # one request: the q4 build, whatever the capture sizes
        (1, [1, 2, 4], 3, ()),
        (2, DEFAULT_SIZES, 3, (4, 8)),  # derived: every tier up to the one that fits
        (2, None, 3, (4, 8)),
        (3, DEFAULT_SIZES, 3, (4, 8, 16)),  # 12 rows
        (4, DEFAULT_SIZES, 3, (4, 8, 16)),  # 16 rows
        (2, DEFAULT_SIZES, 7, (8, 16)),  # K = 7: a request is eight rows
        (8, DEFAULT_SIZES, 0, (4, 8)),  # MTP off: B = 1 on q4, B = 2-8 packed
        (1, None, 0, ()),  # MTP off, one request: the q4 build
        (1, None, 1, ()),  # K = 1 and 2: the q4 build with dead rows
        (1, None, 2, ()),
        (4, DEFAULT_SIZES, 1, (4, 8)),  # K = 1: B = 2-4 packed (B = 2 on the q4 MoE)
        (5, DEFAULT_SIZES, 2, (4, 8, 16)),  # K = 2: B = 2 packed on tier 8
        (1, None, 4, (8,)),  # K >= 4: one request fills tier 8 (aligned, two groups)
        (1, None, 7, (8,)),
        (2, DEFAULT_SIZES, 4, (8, 16)),
        (3, DEFAULT_SIZES, 4, (8, 16)),  # 15 packed rows of 16
        (2, DEFAULT_SIZES, 5, (8, 16)),  # 6 rows each: 6, 12
        (2, [2, 4], 1, (4, 8)),  # a user list is the graphs' token counts (K = 1)
        (2, [8], 3, (8,)),  # row tiers set by the user are honored
        (4, [16], 3, (16,)),
        (2, [16], 7, (16,)),
        (2, [8, 12], 3, (8,)),  # 12 is above the graphs: vLLM's stock graph
        # Larger steps run vLLM's stock forward: the tiers are the served counts'.
        (5, DEFAULT_SIZES, 3, (4, 8, 16)),
        (8, DEFAULT_SIZES, 3, (4, 8, 16)),
        (16, DEFAULT_SIZES, 3, (4, 8, 16)),
        (16, DEFAULT_SIZES, 1, (4, 8, 16)),
        (4, DEFAULT_SIZES, 7, (8, 16)),
        (9, DEFAULT_SIZES, 0, (4, 8, 16)),  # MTP off: B = 9-16 packed on tier 16
        (17, DEFAULT_SIZES, 0, (4, 8, 16)),  # at most 16 requests on the segment
        (4, DEFAULT_SIZES, 4, (8, 16)),  # K = 4: B = 4 is 20 rows
        (4, [16, 32], 3, (16,)),  # a user list keeps its stock sizes
    ],
)
def test_row_tiers_are_derived_or_honored(clean_env, seqs, sizes, drafts, tiers):
    config = _config(seqs=seqs, sizes=sizes, drafts=drafts)
    config.additional_config = {"decode_backend": "latency_specialized"}
    record = switch.from_config(config)
    assert record.enabled
    assert record.row_tiers == tiers


@pytest.mark.parametrize(
    ("seqs", "sizes", "drafts", "flag"),
    [
        (
            3,
            [8],
            3,
            "--cudagraph-capture-sizes 4 8 12",
        ),  # the largest holds 8 of 12 rows
        (
            4,
            [32, 64],
            3,
            "--cudagraph-capture-sizes 4 8 16",
        ),  # above the ceiling
        (2, [2, 4], 3, "--cudagraph-capture-sizes 4 8"),  # a 2-token graph is no tier
        (2, [3, 6], 1, "--cudagraph-capture-sizes 2 4"),  # K = 1: graphs of 2 and 4
        (1, DEFAULT_SIZES, 8, "K = 7"),  # nine tokens: no build
    ],
)
def test_unservable_deployments_are_an_error(clean_env, seqs, sizes, drafts, flag):
    config = _config(seqs=seqs, sizes=sizes, drafts=drafts)
    config.additional_config = {"decode_backend": "latency_specialized"}
    with pytest.raises(ValueError, match=flag):
        switch.from_config(config)


def test_other_backends_leave_the_capture_sizes_alone(clean_env):
    config = _config(seqs=2, sizes=DEFAULT_SIZES)
    config.additional_config = {}
    assert switch.from_config(config) == switch.LowLatencyDecode()
    assert config.compilation_config.cudagraph_capture_sizes == DEFAULT_SIZES


@pytest.mark.parametrize(
    ("seqs", "mtp", "sizes"),
    [
        (1, True, ()),  # one request: the q4 build, no row tiers
        (2, True, (1, 2)),
        (4, True, (1, 2, 4)),
        (8, True, (1, 2, 4, 8)),  # B = 5-8: vLLM's forward and drafter
        (16, True, (1, 2, 4, 8, 16)),
        (12, True, (1, 2, 4, 8)),  # 12 would make a 12-token graph inside B = 3-4's
        (3, True, (1, 2, 3)),  # B = 3: vLLM's drafter on the graph of 3
        (6, True, (1, 2, 4, 6)),
        (4, False, ()),  # no MTP
    ],
)
def test_the_drafters_tiers_add_their_decode_sizes_to_the_capture_sizes(
    seqs, mtp, sizes
):
    assert switch.draft_capture_sizes(seqs, mtp) == sizes
    tiers = switch.draft_tiers(seqs, 3)
    assert switch.drafter_serves(mtp) == mtp
    assert seqs == 1 or not mtp or set(tiers) <= set(sizes)
    assert set(tiers) <= {1, *switch.DRAFT_REQUESTS}


GRAPHS = {
    1: (
        16,
        (4, 8, 16),
        [
            (1, 4, [1], 0),
            (4, 8, [2, 3, 4], 4),
            (8, 8, [5, 6, 7, 8], 8),
            (16, 16, list(range(9, 17)), 0),
        ],
    ),
    2: (
        8,
        (4, 8, 16),
        [(2, 4, [1], 0), (4, 8, [2], 4), (8, 8, [3, 4], 8), (16, 16, [5, 6, 7, 8], 0)],
    ),
    3: (5, (4, 8, 16), [(3, 4, [1], 0), (6, 8, [2], 8), (15, 16, [3, 4, 5], 0)]),
    4: (4, (4, 8, 16), [(4, 4, [1], 0), (8, 8, [2], 0), (16, 16, [3, 4], 0)]),
    5: (3, (8, 16), [(5, 8, [1], 0), (15, 16, [2, 3], 0)]),
    6: (2, (8, 16), [(6, 8, [1], 0), (12, 16, [2], 0)]),
    7: (2, (8, 16), [(7, 8, [1], 0), (14, 16, [2], 0)]),
    8: (2, (8, 16), [(8, 8, [1], 0), (16, 16, [2], 0)]),
}


@pytest.mark.parametrize("per_request", sorted(GRAPHS))
def test_each_run_of_request_counts_has_one_graph(per_request):
    most, tiers, expected = GRAPHS[per_request]
    assert switch.derived_tiers(most, per_request) == tiers
    found = switch.graphs(most, per_request)
    assert [
        (g.tokens, g.tier, list(g.requests), g.build.w8a16_rows) for g in found
    ] == expected
    for requests in range(1, most + 1):
        padded = min(g.tokens for g in found if g.tokens >= requests * per_request)
        graph = {g.tokens: g for g in found}[padded]
        assert graph.build == switch.shape_build(requests, per_request)
        assert requests in graph.requests
    # A larger request count runs vLLM's stock forward: the same graphs.
    assert switch.served_requests(most + 1, per_request) == most
    assert switch.graphs(most + 1, per_request) == found
    assert switch.derived_tiers(most + 1, per_request) == tiers


@pytest.mark.parametrize("per_request", sorted(GRAPHS))
def test_graphs_claim_their_builds(per_request):
    most, tiers, expected = GRAPHS[per_request]
    assert switch.tier_claims(most, per_request) == {
        tokens: switch.Build(tier, w8a16) for tokens, tier, _, w8a16 in expected
    }


def test_one_predicate_decides_which_shapes_the_segment_serves():
    assert all(switch.serves_shape(b, 1) for b in range(1, 17))
    assert switch.serves_shape(2, 2) and switch.serves_shape(2, 3)
    assert switch.serves_shape(1, 4) and switch.serves_shape(1, 5)
    assert not switch.serves_shape(17, 1) and not switch.serves_shape(7, 5)
    assert not switch.serves_shape(9, 2)
    assert not switch.serves_shape(1, 9)
    # At most R = B (1 + K) = 16 rows and B = 16 requests.
    assert switch.serves_shape(4, 4) and not switch.serves_shape(5, 4)
    assert switch.serves_shape(3, 5) and not switch.serves_shape(4, 5)
    assert switch.serves_shape(2, 8) and not switch.serves_shape(3, 8)
    assert switch.serves_shape(8, 2) and not switch.serves_shape(6, 3)
    rows = switch.shape_w8a16_rows
    assert rows(2, 1) == rows(4, 1) == rows(2, 2) == 4
    assert rows(5, 1) == rows(8, 1) == rows(3, 2) == rows(4, 2) == rows(2, 3) == 8
    assert rows(5, 2) == rows(3, 3) == rows(16, 1) == 0  # nine or more rows: W8A8
    assert rows(1, 1) == rows(2, 4) == 0  # the q4 build, an aligned tier: their own
    line = switch.describe_shapes(4, 2)
    assert "K=1: B = [1]: the q4 build" in line
    assert "B = [2]: packed rows on tier 8 (q4 MoE, at most 4 live rows)" in line
    assert "B = [3, 4]: packed rows on tier 8 (q8 MoE, at most 8 live rows)" in line
    assert "stock" not in line
    line = switch.describe_shapes(6, 4)
    assert "B = [3, 4]: whole row groups on tier 16" in line
    assert "B = [5, 6]: vLLM's stock forward (R > 16 or B > 16)" in line
    line = switch.describe_shapes(9, 4)
    assert "B = [5, 6, 7, 8, 9]: vLLM's stock forward (R > 16 or B > 16)" in line
    line = switch.describe_shapes(17, 1)
    assert "B = [9, 10, 11, 12, 13, 14, 15, 16]: packed rows on tier 16;" in line
    assert "B = [17]: vLLM's stock forward (R > 16 or B > 16)" in line
    line = switch.describe_shapes(4, 5)
    assert "B = [4]: vLLM's stock forward (R > 16 or B > 16)" in line


@pytest.mark.parametrize(
    ("per_request", "lone"), [(1, 4), (3, 4), (4, 4), (5, 8), (7, 8), (8, 8)]
)
def test_a_lone_request_keeps_whole_row_groups(per_request, lone):
    assert switch.lone_rows(per_request) == lone
    assert switch.packs(per_request) == (per_request % 4 != 0)


@pytest.mark.parametrize(
    ("tier", "per_request", "packed"),
    [
        (4, 1, False),  # the q4 build
        (8, 1, True),  # two to eight requests of K = 0-2 on tier 8
        (8, 3, True),
        (8, 5, False),  # one request fills tier 8: aligned
        (8, 7, False),
        (16, 4, False),  # whole row groups
        (16, 1, True),
        (16, 5, True),
    ],
)
def test_a_tier_runs_packed_builds_for_partial_row_groups(tier, per_request, packed):
    assert switch.packed_tier(tier, per_request) == packed


@pytest.mark.parametrize(
    ("seqs", "drafts", "sizes"),
    [
        (5, 2, (1, 2, 4, 5)),  # 5 rounds to the 6-token graph of two requests
        (2, 2, (1, 2)),
        (16, 2, (1, 2, 4, 16)),  # 8 would split B = 3-5's graph; 16 is stock's
        (8, 1, ()),  # one draft: no draft decode
    ],
)
def test_short_k_decode_sizes_are_on_the_deployments_graphs(
    clean_env, seqs, drafts, sizes
):
    assert switch.drafter_serves(True, drafts)
    assert switch.draft_capture_sizes(seqs, True, drafts) == sizes
    if drafts > 1:  # K = 1 has no draft decode
        assert set(switch.draft_tiers(seqs, drafts)) <= set(sizes)
    per_request = 1 + drafts
    claims = switch.tier_claims(seqs, per_request)
    for size in sizes:
        rounded = -(-size // per_request) * per_request
        assert size <= seqs and (rounded in claims or rounded > max(claims))


def test_the_drafter_serves_k_1_to_7(clean_env):
    assert all(switch.drafter_serves(True, k) for k in range(1, 8))
    assert not switch.drafter_serves(True, 0) and not switch.drafter_serves(True, 8)
    assert not switch.drafter_serves(False, 3)
    assert switch.draft_capture_sizes(4, True, 3) == (1, 2, 4)
    assert switch.draft_capture_sizes(4, True, 2) == (1, 2, 4)


@pytest.mark.parametrize(
    ("seqs", "drafts", "sizes"),
    [
        (1, 3, ()),  # the q4 build keeps vLLM's default sizes
        (1, 4, (1,)),  # a lone request of 5 tokens has row tiers: its draft decode
        (1, 7, (1,)),
        (2, 5, (1, 2)),
        (4, 3, (1, 2, 4)),
        (4, 7, (1, 2, 4)),  # no drafter tier of four (32 rows): vLLM's drafter at 3-4
        (13, 3, (1, 2, 4, 8)),  # 13 would take B = 3 (12 tokens) from the graph of 16
        (3, 4, (1, 2, 3)),  # B = 2 pads to the 3-request graph: vLLM's drafter
    ],
)
def test_the_drafters_decode_sizes_at_every_k(clean_env, seqs, drafts, sizes):
    assert switch.drafter_serves(True, drafts)
    assert switch.draft_capture_sizes(seqs, True, drafts) == sizes
    assert set(switch.draft_tiers(seqs, drafts)) <= set(sizes) or seqs == 1


def test_a_drafter_tier_is_the_top_of_its_target_graph():
    assert switch.draft_tiers(16, 3) == (1, 2, 4)  # B = 5-16: vLLM's drafter
    assert switch.draft_tiers(6, 3) == (1, 2, 4)
    assert switch.draft_tiers(2, 4) == (1, 2)  # two requests own the 10-token graph
    assert switch.draft_tiers(3, 4) == (1,)  # two pad to the 15-token graph of three
    assert switch.draft_tiers(16, 7) == (1, 2)
    assert switch.draft_tiers(8, 1) == (1, 2)  # B = 3-8: vLLM's drafter
    assert switch.draft_tiers(1, 1) == (1,) and switch.draft_tiers(1, 2) == (1,)
    assert switch.draft_tiers(4, 2) == (1, 2, 4)
    assert switch.draft_tiers(5, 2) == (1, 2)  # B = 3-5 share the 15-token graph


def test_one_drafter_tier_binds_another_target_tier_than_its_step():
    foreign = set()
    for seqs in (1, 2, 3, 4, 8, 16):
        for drafts in switch.DRAFT_KS:
            rows = drafts + 1
            # A tier's catch-up replays at its target graph's rows.
            tiers = [
                tier
                for tier in switch.draft_tiers(seqs, drafts)
                if switch.serves_shape(tier, rows)
            ]
            claims = switch.tier_claims(seqs, rows)
            assert all(tier * rows in claims for tier in tiers)
            for b in range(1, seqs + 1):
                tier = next((size for size in tiers if size >= b), None)
                if tier is None:
                    continue
                if switch.shape_build(b, rows) != switch.shape_build(tier, rows):
                    foreign.add((seqs, drafts, tier))
    assert foreign == set()


def test_derived_tiers_become_the_capture_sizes(clean_env):
    config = _config(seqs=4, sizes=DEFAULT_SIZES)
    switch._set_capture_sizes(config, (4, 8, 16))
    assert config.compilation_config.cudagraph_capture_sizes == [4, 8, 16]
    assert config.compilation_config.max_cudagraph_capture_size == 16
    config = _config(seqs=2, sizes=[8])
    switch._set_capture_sizes(config, (8,))  # already row tiers: untouched
    assert config.compilation_config.cudagraph_capture_sizes == [8]
    config = _config(seqs=1, sizes=DEFAULT_SIZES)
    switch._set_capture_sizes(config, ())  # one request: untouched
    assert config.compilation_config.cudagraph_capture_sizes == DEFAULT_SIZES


@pytest.mark.parametrize(
    ("moe", "selected"),
    [("auto", "flashinfer_trtllm"), ("flashinfer_trtllm", "flashinfer_trtllm")],
)
def test_moe_backend_auto_selects_flashinfer_trtllm(clean_env, moe, selected):
    config = _config(moe=moe)
    assert not any("--moe-backend" in m for m in switch._unsupported(config))
    switch._select_moe_backend(config)
    assert config.kernel_config.moe_backend == selected


@pytest.mark.parametrize(("seqs", "drafts"), [(4, 1), (16, 3), (1, 4), (4, 3)])
def test_the_drafter_inherits_the_targets_moe_backend(clean_env, seqs, drafts):
    config = _config(seqs=seqs, drafts=drafts, moe="auto")
    switch._select_moe_backend(config)
    assert config.speculative_config.moe_backend is None
    assert config.kernel_config.moe_backend == "flashinfer_trtllm"


def test_an_explicit_drafter_moe_backend_is_kept(clean_env):
    config = _config(seqs=4, drafts=1)
    config.speculative_config.moe_backend = "triton"
    switch._select_moe_backend(config)
    assert config.speculative_config.moe_backend == "triton"
    config = _config()
    config.speculative_config = None  # MTP off
    switch._select_moe_backend(config)
    assert config.speculative_config is None


def test_other_platforms_refuse_the_path_and_keep_deepseek_v32(monkeypatch):
    import vllm.models.glm53 as glm53
    from vllm.models import deepseek_v32
    from vllm.models.glm53 import generic

    other = SimpleNamespace(
        is_cuda=lambda: True, is_device_capability_family=lambda family: False
    )
    monkeypatch.setattr(vllm.platforms, "_current_platform", other)
    cls = glm53.GlmMoeDsaForCausalLM
    assert cls is generic.GlmMoeDsaForCausalLM
    assert issubclass(cls, deepseek_v32.GlmMoeDsaForCausalLM)
    assert "__init__" in vars(cls)
    assert [name for name in vars(cls) if not name.startswith("_")] == []
    calls: list[str] = []

    def stock_init(self, *, vllm_config, prefix=""):
        calls.append(prefix)

    monkeypatch.setattr(deepseek_v32.GlmMoeDsaForCausalLM, "__init__", stock_init)
    config = SimpleNamespace(additional_config={})
    cls(vllm_config=config, prefix="model")
    assert calls == ["model"]
    for backend, match in (("latency_specialized", "SM100 or SM103"), ("x", "auto")):
        config.additional_config = {"decode_backend": backend}
        with pytest.raises(ValueError, match=match):
            cls(vllm_config=config)
    assert calls == ["model"]


# The modules the default (auto) path imports; the kernels load only when enabled.
AUTO_MODULES = {
    "",
    ".model",
    ".switch",
    ".ops",
    ".ops.launch",
    ".ops.cute_dsl",
    ".ops.cute_dsl.registry",
    ".ops.cute_dsl.row_table",
    ".ops.cute_dsl.moe_q_shape",
    ".ops.cute_dsl.w8a8_shape",
}


def test_auto_imports_no_segment_modules():
    code = f"""
import sys
sys.modules["cutlass"] = None  # the switch imports without CuTe DSL
from vllm.models.glm53.nvidia import switch
del sys.modules["cutlass"]
import vllm.models.glm53.nvidia.model
prefix = "vllm.models.glm53.nvidia"
loaded = {{m[len(prefix):] for m in sys.modules if m.startswith(prefix)}}
assert loaded <= {AUTO_MODULES!r}, sorted(loaded - {AUTO_MODULES!r})
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("drafts", switch.DRAFT_KS)
def test_every_drafter_tier_is_a_built_request_tier(drafts):
    from vllm.models.glm53.nvidia.ops.cute_dsl import registry

    for seqs in range(1, 17):
        tiers = switch.draft_tiers(seqs, drafts)
        assert tiers[0] == 1 and list(tiers) == sorted(set(tiers))
        assert set(tiers[1:]) <= set(registry.DRAFT_REQUESTS)
    assert switch.draft_tiers(4, 4) == (1,)  # 10 tokens are no graph; 20 rows: stock
