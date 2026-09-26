# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from contextlib import contextmanager, nullcontext
from inspect import signature
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
import torch

import vllm.model_executor.warmup.kernel_warmup as warmup
import vllm.utils.flashinfer as fi_utils
from vllm.model_executor.warmup.kernel_warmup import (
    _flashinfer_autotune_token_counts,
    _run_flashinfer_autotune_dummy_runs,
)

pytestmark = pytest.mark.cpu_test


class _FakeMoERunner:
    """Minimal MoERunner stand-in for token-count discovery tests."""

    moe_config: Any


def _make_moe(*, max_deferred_tokens: int = 128, enabled: bool = True):
    """Create a fake MoE layer with a deferred-finalize token limit."""
    moe = _FakeMoERunner()
    moe.moe_config = SimpleNamespace(
        use_deferred_moe_finalize=enabled,
        defer_moe_finalize_max_num_tokens=max_deferred_tokens,
    )
    return moe


def _make_runner(modules, *, max_tokens: int = 8192, linear_backend: str = "auto"):
    """Create a runner carrying only the state used by FlashInfer warmup."""
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        vllm_config=SimpleNamespace(
            kernel_config=SimpleNamespace(linear_backend=linear_backend),
            attention_config=SimpleNamespace(hisparse_config=None),
            parallel_config=SimpleNamespace(enable_elastic_ep=False, nnodes=1),
        ),
        get_model=Mock(
            return_value=SimpleNamespace(modules=Mock(return_value=modules))
        ),
        _dummy_run=Mock(),
    )


def test_flashinfer_autotune_token_counts_include_deferred_moe_limits():
    runner = _make_runner(
        [
            object(),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=128),
            _make_moe(max_deferred_tokens=64, enabled=False),
            _make_moe(max_deferred_tokens=-1),
        ],
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (8192, 32, 128)


def test_flashinfer_autotune_token_counts_are_bounded_and_deduplicated():
    runner = _make_runner(
        [_make_moe(max_deferred_tokens=4096)],
        max_tokens=32,
        linear_backend="flashinfer_cutedsl",
    )

    with patch("vllm.model_executor.layers.fused_moe.MoERunner", _FakeMoERunner):
        token_counts = _flashinfer_autotune_token_counts(runner)

    assert token_counts == (32,)


@pytest.mark.parametrize("skip_attn", [False, True])
def test_flashinfer_autotune_uses_token_buckets_for_each_dummy_run(skip_attn):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner as V2Runner
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner as V1Runner

    runner = _make_runner([])
    dummy_run_signature = signature((V2Runner if skip_attn else V1Runner)._dummy_run)
    runner._dummy_run.side_effect = lambda **kwargs: dummy_run_signature.bind(
        None, **kwargs
    )
    max_buckets = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 8192)
    deferred_buckets = (1, 2, 4, 8, 16, 32, 64, 128)

    with (
        patch(
            "vllm.model_executor.warmup.kernel_warmup."
            "_flashinfer_autotune_token_counts",
            return_value=(8192, 128),
        ),
        patch(
            "vllm.utils.flashinfer.flashinfer_get_hybrid_num_tokens_buckets",
            side_effect=(max_buckets, deferred_buckets),
        ) as get_buckets,
        patch("vllm.utils.flashinfer.autotune") as autotune,
    ):
        _run_flashinfer_autotune_dummy_runs(runner, skip_attn=skip_attn)

    assert get_buckets.call_args_list == [call(8192), call(128)]
    assert autotune.call_args_list == [
        call(tuning_buckets=max_buckets),
        call(tuning_buckets=deferred_buckets),
    ]
    assert runner._dummy_run.call_args_list == [
        call(
            num_tokens=8192,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
        call(
            num_tokens=128,
            skip_eplb=True,
            is_profile=True,
            randomize_inputs=True,
            **({"skip_attn": True} if skip_attn else {}),
        ),
    ]


@pytest.mark.parametrize(
    "has_v2,elastic_ep,node_count",
    [(False, False, 1), (True, False, 1), (True, True, 1), (True, False, 2)],
    ids=["v1", "v2", "elastic-v1", "multi-node-v1"],
)
@pytest.mark.parametrize("hisparse", [False, True])
@pytest.mark.parametrize("world_size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("skip_ops", [[], ["fp4_gemm", "bmm_fp8"]])
@pytest.mark.parametrize("failure", [None, "dummy", "replayssm", "kimi", "bf16"])
def test_flashinfer_autotune_lifecycle(
    monkeypatch,
    tmp_path,
    has_v2,
    elastic_ep,
    node_count,
    hisparse,
    world_size,
    rank,
    skip_ops,
    failure,
):
    use_v2 = has_v2 and not elastic_ep and node_count == 1
    runner = _make_runner([])
    runner.vllm_config.parallel_config.enable_elastic_ep = elastic_ep
    runner.vllm_config.attention_config.hisparse_config = object() if hisparse else None
    model = runner.get_model()
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b"cached")
    events = Mock()
    events.broadcast.return_value = b"cached"
    world = SimpleNamespace(
        rank_in_group=rank,
        world_size=world_size,
        cpu_group=object(),
        broadcast_object=events.broadcast,
        barrier=events.barrier,
    )

    in_autotune = False

    @contextmanager
    def autotune_context(**kwargs):
        nonlocal in_autotune
        assert torch.is_inference_mode_enabled()
        in_autotune = True
        events.enter()
        try:
            yield
        finally:
            in_autotune = False
            events.exit()

    def bf16_warmup(*args, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not in_autotune
        if failure == "bf16":
            raise RuntimeError("warmup failed")

    events.autotune.side_effect = autotune_context
    if failure is not None:
        getattr(events, failure).side_effect = RuntimeError("warmup failed")
    events.bf16.side_effect = bf16_warmup
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(
            AutoTuner=SimpleNamespace(get=lambda: events.tuner),
            set_autotune_process_group=events.set_group,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(autotune_v2=events.autotune, autotune_v2_reload=events.reload),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_node_count", lambda: node_count
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: has_v2)
    monkeypatch.setattr(fi_utils, "autotune", events.autotune)
    monkeypatch.setattr(warmup.envs, "VLLM_FLASHINFER_AUTOTUNE_SKIP_OPS", skip_ops)
    resolve_file = Mock(return_value=cache_path)
    resolve_root = Mock(return_value=str(tmp_path))
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_file", resolve_file)
    monkeypatch.setattr(warmup, "resolve_flashinfer_autotune_v2_root", resolve_root)
    monkeypatch.setattr(warmup, "write_flashinfer_autotune_cache", events.write)
    monkeypatch.setattr(warmup, "_run_flashinfer_autotune_dummy_runs", events.dummy)
    monkeypatch.setattr(warmup, "replayssm_autotune_warmup", events.replayssm)
    monkeypatch.setattr(warmup, "_autotune_kimi_k3_kda_qkvg", events.kimi)
    monkeypatch.setattr(warmup, "_run_flashinfer_bf16_autotune_dummy_run", events.bf16)
    monkeypatch.setattr(
        warmup, "autotune_hisparse_flashinfer_attention", events.hisparse
    )

    if failure is None:
        warmup.flashinfer_autotune(runner)
    else:
        with pytest.raises(RuntimeError, match="warmup failed"):
            warmup.flashinfer_autotune(runner)

    kwargs = {"skip_ops": set(skip_ops)} if skip_ops else {}
    expected = []
    if use_v2:
        resolve_file.assert_not_called()
        resolve_root.assert_called_once_with()
        expected.append(call.autotune(mode="tune", cache_root=str(tmp_path), **kwargs))
    else:
        resolve_root.assert_not_called()
        resolve_file.assert_called_once_with(runner)
        expected.extend(
            [
                call.broadcast(b"cached" if rank == 0 else None, src=0),
                call.write(cache_path, b"cached"),
                call.barrier(),
                call.tuner.load_configs(str(cache_path)),
                call.autotune(tune_mode=True, **kwargs),
            ]
        )
    expected.extend(
        [call.set_group(world.cpu_group if world_size > 1 else None), call.enter()]
    )
    if hisparse:
        expected.append(call.hisparse(runner))
    for name, arg, kwargs in [
        ("dummy", runner, {"skip_attn": hisparse}),
        ("replayssm", runner, {}),
        ("kimi", model, {}),
    ]:
        expected.append(getattr(call, name)(arg, **kwargs))
        if failure == name:
            break
    expected.append(call.exit())
    if failure in (None, "bf16"):
        expected.append(
            call.bf16(runner, skip_ops=set(skip_ops) or None, skip_attn=hisparse)
        )
    expected.append(call.set_group(None))
    if failure is None:
        if world_size > 1:
            expected.append(call.barrier())
        if use_v2 and world_size > 1:
            expected.extend([call.reload(), call.barrier()])
        elif not use_v2 and rank == 0:
            expected.append(call.tuner.save_configs(str(cache_path)))
    assert events.mock_calls == expected


@pytest.mark.parametrize("node_count,elastic_ep", [(1, False), (2, False), (1, True)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("model_runner_v2,max_reqs", [(False, 2), (True, 1), (True, 2)])
def test_sparse_autotune_respects_cache_selection(
    monkeypatch, tmp_path, node_count, elastic_ep, rank, model_runner_v2, max_reqs
):
    import vllm.model_executor.warmup.flashinfer_sparse_mla_warmup as sparse

    runner = _make_runner([])
    runner.max_num_reqs = max_reqs
    runner.vllm_config.use_v2_model_runner = model_runner_v2
    runner.vllm_config.kernel_config.enable_flashinfer_autotune = True
    runner.vllm_config.parallel_config.enable_elastic_ep = elastic_ep
    worker = SimpleNamespace(
        model_runner=runner,
        vllm_config=runner.vllm_config,
        execute_model=Mock(),
        sample_tokens=Mock(),
    )
    world = SimpleNamespace(
        rank_in_group=rank,
        world_size=2,
        broadcast_object=Mock(return_value=b"cached"),
        barrier=Mock(),
    )
    managed = Mock(return_value=nullcontext())
    legacy = Mock(return_value=nullcontext())
    reload = Mock()
    tuner = Mock()
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(autotune_v2=managed, autotune_v2_reload=reload),
    )
    monkeypatch.setitem(
        sys.modules,
        "flashinfer.autotuner",
        SimpleNamespace(AutoTuner=SimpleNamespace(get=lambda: tuner)),
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_world_group", lambda: world
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_node_count", lambda: node_count
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_autotune_v2", lambda: True)
    monkeypatch.setattr(sparse, "has_flashinfer", lambda: True)
    monkeypatch.setattr(
        sparse.current_platform, "is_device_capability_family", lambda _: True
    )
    monkeypatch.setattr(
        sparse, "_flashinfer_sparse_mla_decode_label", lambda *_: "DSv3.2"
    )
    monkeypatch.setattr(sparse, "resolve_flashinfer_autotune_v2_root", lambda: tmp_path)
    cache_path = tmp_path / "autotune.json"
    cache_path.write_bytes(b"cached")
    monkeypatch.setattr(
        sparse, "resolve_flashinfer_autotune_file", lambda _: cache_path
    )
    monkeypatch.setattr(sparse, "flashinfer_autotune", legacy)

    def mixed_warmup(*args, mixed_step_context=None, **kwargs):
        context = (
            mixed_step_context if mixed_step_context is not None else nullcontext()
        )
        with context:
            assert torch.is_inference_mode_enabled()
        return True

    mixed = Mock(side_effect=mixed_warmup)
    monkeypatch.setattr(sparse, "run_mixed_prefill_decode_warmup", mixed)
    assert sparse._run_flashinfer_sparse_mla_decode_autotune(worker, 16, frozenset())

    if node_count == 1 and not elastic_ep:
        managed.assert_called_once_with(mode="tune", cache_root=tmp_path)
        legacy.assert_not_called()
        world.broadcast_object.assert_not_called()
        tuner.load_configs.assert_not_called()
        reload.assert_called_once_with()
        assert world.barrier.call_count == 2
    else:
        managed.assert_not_called()
        reload.assert_not_called()
        assert legacy.call_count == (1 if rank == 0 else 0)
        world.broadcast_object.assert_called_once_with(
            b"cached" if rank == 0 else None, src=0
        )
        tuner.load_configs.assert_called_once_with(str(cache_path))
    assert mixed.call_count == (1 if model_runner_v2 and max_reqs >= 2 else 0)
    assert runner._dummy_run.call_count == (0 if mixed.called else 1)
