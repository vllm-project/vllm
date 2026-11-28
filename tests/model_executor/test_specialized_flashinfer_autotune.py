# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.model_executor.warmup import flashinfer_autotune_cache as autotune_cache
from vllm.models.deepseek_v41.nvidia.specialized_decode.runtime.autotune import (
    preload_rank_flashinfer_autotune_cache,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def specialized_run(autotune_run, monkeypatch):
    run = autotune_run(pp=1, tp=4)
    monkeypatch.setattr(
        autotune_cache,
        "_flashinfer_autotune_path",
        lambda config, config_hash, *, per_rank: run.cache_path(run.rank),
    )
    monkeypatch.setattr(
        current_platform, "update_block_size_for_backend", lambda _: None
    )
    return run


@pytest.mark.parametrize(
    "early,late",
    [
        ("miss", "hit"),
        ("full_hit", "same"),
        ("hit", "same"),
        ("hit", "rejected_rank"),
        ("hit", "changed_key"),
        ("miss", "miss"),
        ("missing_rank", "hit"),
        ("rejected_rank", "hit"),
        ("unreadable_rank", "hit"),
        ("miss", "rejected_rank"),
    ],
)
def test_specialized_autotune_preserves_cache_and_collectives(
    specialized_run, monkeypatch, tmp_path, early, late
):
    run = specialized_run
    table = {"specialized_gemm": 17, "shared_gemm": 23}
    if early == "full_hit":
        table["specialized_only"] = 31
    if early != "miss":
        for rank in run.ranks:
            if early != "missing_rank" or rank != 3:
                run.cache_path(rank).write_text(json.dumps(table))
    if early == "rejected_rank":
        run.rejected_load_ranks = {3}
    if early == "unreadable_rank":
        run.unreadable_ranks = {3}

    def prewarm(rank):
        state = run.states[rank]
        path = preload_rank_flashinfer_autotune_cache(state.vllm_config, state.model)
        state.flashinfer_autotune_preload = path
        tuner = run.tuners[rank]
        run.set_group(run.tensor_group())
        tuner.profile("specialized_gemm")
        tuner.profile("specialized_only")
        return path

    preloaded = run.on_ranks(prewarm)
    run.rejected_load_ranks.clear()
    run.unreadable_ranks.clear()
    if (early, late) == ("miss", "hit"):
        run.cache_dir = tmp_path / "late_key"
        run.cache_dir.mkdir()
    if late in ("hit", "rejected_rank"):
        for rank in run.ranks:
            run.cache_path(rank).write_text(json.dumps(table))
        if late == "rejected_rank":
            run.rejected_load_ranks = {3}
    elif late == "changed_key":
        paths = {rank: run.cache_path(rank) for rank in run.ranks}
        paths[3] = tmp_path / "late_key" / paths[3].name
        paths[3].parent.mkdir()
        paths[3].write_text(json.dumps({**table, "specialized_gemm": 99}))
        monkeypatch.setattr(run, "cache_path", paths.__getitem__)

    key_changed = late == "changed_key"
    run.execute(
        expected_error="FlashInfer autotune cache key changed" if key_changed else None
    )
    run.assert_collectives_match()
    assert all(
        state.flashinfer_autotune_preload is None for state in run.states.values()
    )
    early_hit = early in ("full_hit", "hit")
    assert [path is not None for path in preloaded] == [early_hit] * 4
    expect_sweep = late == "miss" or (late == "rejected_rank" and not early_hit)
    assert sorted(run.dummy_run_ranks) == (list(run.ranks) if expect_sweep else [])
    for rank, tuner in run.tuners.items():
        if early_hit:
            # One live read, no second load by the late warmup, even if
            # the path would reject a new read after the early hit.
            assert run.validation_paths[rank] == tuner.load_paths == [preloaded[rank]]
        if key_changed:
            assert not tuner.cache
            assert not tuner._dirty
            continue
        if late == "rejected_rank" and not early_hit:
            assert tuner.cache == {"shared_gemm": 0}
            continue
        assert tuner.cache["specialized_gemm"] == (17 if early_hit else 0)
        assert tuner.cache["specialized_only"] == (31 if early == "full_hit" else 0)
        assert ("shared_gemm" in tuner._file_configs) == (early_hit or late == "hit")
        if late == "hit" and not early_hit:
            assert tuner._file_configs["specialized_gemm"] == 17
            assert tuner.cache["shared_gemm"] == 23
    want_saves = [] if early == "full_hit" or key_changed else list(run.ranks)
    assert sorted(rank for rank, _, _ in run.saves) == want_saves
    for rank, path, saved in run.saves:
        assert path == run.cache_path(rank)
        assert saved == run.tuners[rank].cache


@pytest.mark.parametrize(
    "case",
    [
        "clean",
        "profiled",
        "file_loaded",
        "namespaced",
        "auto_fit",
        "graph_downgrade",
        "breakable",
    ],
)
def test_specialized_autotune_resolves_final_rank_key_when_ready(
    autotune_run, autotune_model_state, monkeypatch, tmp_path, case
):
    run = autotune_run(pp=1, tp=1)
    state = autotune_model_state(max_model_len=-1 if case == "auto_fit" else 64)
    tuner = run.tuners[0]
    existing_layer = {
        "profiled": tuner.profiling_cache,
        "file_loaded": tuner._file_configs,
        "namespaced": tuner._namespaced_records,
    }.get(case)
    if existing_layer is not None:
        existing_layer["already_warm"] = 7
    monkeypatch.setenv("VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    config = state.vllm_config
    if case in ("graph_downgrade", "breakable"):
        config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
        config.compilation_config.use_breakable_cudagraph = case == "breakable"
    # The weight daemon seeds from a runner that does not have a model yet.
    runner = SimpleNamespace(vllm_config=config)

    def path_for_rank():
        return run.on_ranks(
            lambda _: autotune_cache.resolve_rank_flashinfer_autotune_file(runner)
        )[0]

    def update_block_size(config):
        config.cache_config.block_size = 128

    default_path = path_for_rank()
    monkeypatch.setattr(
        current_platform, "update_block_size_for_backend", update_block_size
    )
    default_block_size = config.cache_config.block_size
    update_block_size(config)
    final_path = path_for_rank()
    assert final_path != default_path
    config.cache_config.block_size = default_block_size
    final_path.write_text('{"specialized_gemm": 17}')

    path = run.on_ranks(
        lambda _: preload_rank_flashinfer_autotune_cache(config, state.model)
    )[0]
    if case in ("clean", "breakable"):
        assert path == final_path
        assert tuner.cache == {"specialized_gemm": 17}
    else:
        assert path is None
        assert not tuner.load_paths and not run.validation_paths
        if existing_layer is not None:
            assert existing_layer == {"already_warm": 7}


@pytest.mark.parametrize("rejection", ["validation", "replacement"])
def test_early_vote_rejects_before_touching_the_live_tuner(specialized_run, rejection):
    run = specialized_run
    for rank, tuner in run.tuners.items():
        run.cache_path(rank).write_text('{"specialized_gemm": 17}')
        tuner.protocol_records["choice"] = rank
    if rejection == "validation":
        run.rejected_load_ranks = {3}
    else:
        run.replace_during_validation = {3}

    def preload(rank):
        state = run.states[rank]
        return preload_rank_flashinfer_autotune_cache(state.vllm_config, state.model)

    assert run.on_ranks(preload) == [None] * 4
    run.assert_collectives_match()
    for rank, tuner in run.tuners.items():
        assert run.validation_paths[rank] == [run.cache_path(rank)]
        assert not tuner.load_paths and not tuner.cache
        assert tuner.protocol_records == {"choice": rank}


def test_early_install_rejection_stops_every_rank_before_tuning(specialized_run):
    run = specialized_run
    for rank in run.ranks:
        run.cache_path(rank).write_text('{"specialized_gemm": 17}')
    run.rejected_install_ranks = {3}

    def preload(rank):
        state = run.states[rank]
        with pytest.raises(RuntimeError, match="FlashInfer autotune cache changed"):
            preload_rank_flashinfer_autotune_cache(state.vllm_config, state.model)

    run.on_ranks(preload)
    run.assert_collectives_match()
    assert not run.saves and not run.profile_groups
