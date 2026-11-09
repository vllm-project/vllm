# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import pytest
import torch

from vllm.distributed.weight_transfer.base import (
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)
from vllm.platforms import current_platform


@dataclass
class _KernelUpdateInfo(WeightTransferUpdateInfo):
    weights: list[tuple[str, torch.Tensor]]


class _InProcessKernelTransferEngine(
    WeightTransferEngine[WeightTransferInitInfo, _KernelUpdateInfo]
):
    """Replace external transport while keeping real worker update sessions."""

    init_info_cls = WeightTransferInitInfo
    update_info_cls = _KernelUpdateInfo

    def init_transfer_engine(self, init_info: WeightTransferInitInfo) -> None:
        pass

    def start_weight_update(self) -> None:
        pass

    def finish_weight_update(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    @torch.no_grad()
    def receive_weights(self, update_info: _KernelUpdateInfo) -> None:
        from vllm.model_executor.model_loader.reload.inplace import (
            get_parameter_for_reload,
        )

        for name, weight in update_info.weights:
            get_parameter_for_reload(self.model, name).copy_(weight)


def _check_xpu_moe_reload(worker):
    """Reload through the worker and replay a graph holding model parameters."""
    import torch

    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    model = worker.model_runner.get_model()
    block = next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
    assert block._xpu_moe_shared_fused_plan is not None, "Prepare before graph capture"
    config = worker.vllm_config
    assert block._decode is not None and block.shared_expert is not None
    block._decode.runtime.prepare_capture(1, 1)
    x = torch.ones(1, block.gate.input_size, dtype=torch.float16, device=worker.device)
    stream = torch.xpu.Stream(device=worker.device)
    stream.wait_stream(torch.xpu.current_stream())
    with torch.inference_mode(), set_forward_context(None, config):
        with torch.xpu.stream(stream):
            block(x)
            block(x)
        torch.xpu.current_stream().wait_stream(stream)
        plan = block._xpu_moe_shared_fused_plan
        assert plan is not None, "The native provider must run in this test"
        workspace = plan.workspace(1)
        weight_ptr = plan.shared_w2.data_ptr()
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            output = block(x)
        graph.replay()
        torch.xpu.synchronize()
        before = output.clone()
        weight = block.shared_expert.down_proj.weight
        assert plan.shared_w2 is weight
        name = next(n for n, p in model.named_parameters() if p is weight)
        original = weight.clone()
        try:
            worker.reload_weights(
                [(name, torch.zeros_like(weight))], is_checkpoint_format=False
            )
            torch.testing.assert_close(plan.shared_w2, weight, rtol=0, atol=0)
            assert block._xpu_moe_shared_fused_plan is plan
            assert plan.shared_w2.data_ptr() == weight_ptr
            assert plan.workspace(1) is workspace
            x.fill_(0.5)
            expected = Qwen3NextSparseMoeBlock.forward(block, x)
            graph.replay()
            torch.xpu.synchronize()
            torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
            assert not torch.equal(output, before)
        finally:
            worker.reload_weights([(name, original)], is_checkpoint_format=False)
        graph.replay()
        torch.xpu.synchronize()
        expected = Qwen3NextSparseMoeBlock.forward(block, x)
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)
    return block.tp_size


def _check_xpu_moe_weight_updates(worker):
    """Default target/draft graphs retain independent weight update sessions."""
    from vllm.config import WeightTransferConfig
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.models.qwen3_next import Qwen3NextSparseMoeBlock
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    worker = worker.worker
    models = [worker.get_model(), worker.get_draft_model()]
    assert models[1] is not None
    blocks = [
        next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
        for model in models
    ]
    original_engine = worker.weight_transfer_engine
    worker.weight_transfer_engine = _InProcessKernelTransferEngine(
        WeightTransferConfig(backend="ipc"),
        worker.vllm_config,
        worker.device,
        models[0],
    )
    try:
        for is_draft in (False, True):
            model, block = models[is_draft], blocks[is_draft]
            assert block.shared_expert is not None
            other_block = blocks[not is_draft]
            assert other_block.shared_expert is not None
            weight = block.shared_expert.down_proj.weight
            other_weight = other_block.shared_expert.down_proj.weight
            assert weight is not other_weight
            original, untouched = weight.clone(), other_weight.clone()
            name = next(n for n, p in model.named_parameters() if p is weight)
            start = (
                worker.start_draft_weight_update
                if is_draft
                else worker.start_weight_update
            )
            x = torch.ones(
                1, block.gate.input_size, dtype=torch.float16, device=worker.device
            )
            stream = torch.xpu.Stream(device=worker.device)
            stream.wait_stream(torch.xpu.current_stream())
            with torch.inference_mode(), set_forward_context(None, worker.vllm_config):
                with torch.xpu.stream(stream):
                    block(x)
                    block(x)
                torch.xpu.current_stream().wait_stream(stream)
                assert block._decode is None, "Auto MTP/updates must remain independent"
                weight_ptr = weight.data_ptr()
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=stream):
                    output = block(x)
                graph.replay()
                torch.xpu.synchronize()
                before = output.clone()
                try:
                    start()
                    worker.update_weights(
                        {"weights": [(name, torch.zeros_like(weight))]}
                    )
                    worker.finish_weight_update()
                    assert torch.count_nonzero(weight).item() == 0
                    torch.testing.assert_close(other_weight, untouched, rtol=0, atol=0)
                    assert block._decode is None
                    assert weight.data_ptr() == weight_ptr
                    expected = Qwen3NextSparseMoeBlock.forward(block, x)
                    graph.replay()
                    torch.xpu.synchronize()
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    assert not torch.equal(output, before)
                finally:
                    start()
                    worker.update_weights({"weights": [(name, original)]})
                    worker.finish_weight_update()
                graph.replay()
                torch.xpu.synchronize()
                torch.testing.assert_close(output, before, rtol=0, atol=0)
    finally:
        worker.weight_transfer_engine = original_engine
    return blocks[0].tp_size


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
def test_xpu_qwen3_5_moe_reload_graph_tp2(vllm_runner, enable_pickle):
    """Opt-in ordinary TP2 decode graphs observe in-place kernel-format reloads."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig

    with vllm_runner(
        os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dtype="float16",
        quantization="fp8_per_tensor",
        additional_config={"decode_backend": "latency_specialized"},
        tensor_parallel_size=2,
        distributed_executor_backend="mp",
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 2]
        ),
    ) as model:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids
        assert model.collective_rpc(_check_xpu_moe_reload) == [2, 2]
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids


def _check_xpu_moe_a8_shared_expert(worker):
    from tests.kernels.moe.test_moe import _check_xpu_qwen3_5_decode
    from vllm.models.qwen3_5.xpu.model import Qwen3_5SparseMoeBlock

    model = worker.model_runner.get_model()
    block = next(m for m in model.modules() if isinstance(m, Qwen3_5SparseMoeBlock))
    for num_tokens in (1, 8):
        _check_xpu_qwen3_5_decode(
            (block, worker.vllm_config, True), num_tokens, should_fuse=False
        )
    return block.tp_size


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
def test_xpu_qwen3_5_moe_a8_shared_expert_fallback(vllm_runner, enable_pickle):
    """Real TP2 decode retains A8 shared projections, including graph replay."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig, KernelConfig

    with vllm_runner(
        os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dtype="float16",
        quantization="fp8_per_tensor",
        tensor_parallel_size=2,
        kernel_config=KernelConfig(linear_backend="xpu"),
        distributed_executor_backend="mp",
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=128,
        enable_chunked_prefill=True,
        max_num_seqs=8,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[1, 8]
        ),
    ) as model:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        outputs = model.llm.generate(["Hello"] * 8, params)
        assert all(len(output.outputs[0].token_ids) == 4 for output in outputs)
        assert model.collective_rpc(_check_xpu_moe_a8_shared_expert) == [2, 2]


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
def test_xpu_qwen3_5_default_mtp_weight_update_graph(vllm_runner, enable_pickle):
    """Disabling specialization preserves target/draft graphs and weight updates."""
    from vllm import SamplingParams
    from vllm.config import CompilationConfig

    with vllm_runner(
        os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
        dtype="float16",
        quantization="fp8_per_tensor",
        additional_config={"decode_backend": "auto"},
        tensor_parallel_size=2,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
        distributed_executor_backend="mp",
        language_model_only=True,
        max_model_len=256,
        max_num_batched_tokens=128,
        max_num_seqs=8,
        kv_cache_memory_bytes=128 * 1024 * 1024,
        enable_prefix_caching=False,
        compilation_config=CompilationConfig(
            mode=0, cudagraph_mode="FULL_DECODE_ONLY", cudagraph_capture_sizes=[2]
        ),
    ) as model:
        params = SamplingParams(temperature=0, max_tokens=4, ignore_eos=True)
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids
        assert model.collective_rpc(_check_xpu_moe_weight_updates) == [2, 2]
        assert model.llm.generate("Hello", params)[0].outputs[0].token_ids


def _profile_decode_provider(worker, runtime):
    """Observe bound native math without enabling PTI graph collection."""
    native_calls = 0
    if runtime is not None:
        plan = runtime.executors[0].plan
        assert plan is not None
        x = torch.ones(
            1, plan.w13.shape[1], dtype=plan.shared_gate.dtype, device=worker.device
        )
        ids = torch.arange(
            plan.top_k, dtype=torch.int32, device=worker.device
        ).unsqueeze(0)
        weights = torch.full(
            (1, plan.top_k), 1 / plan.top_k, dtype=torch.float32, device=worker.device
        )
        output = torch.empty_like(x)
        plan.forward(output, x, weights, ids)
        torch.xpu.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as provider_profiler:
            plan.forward(output, x, weights, ids)
            torch.xpu.synchronize()
        native_calls = sum(
            e.count
            for e in provider_profiler.key_averages()
            if e.key == "_xpu_C::moe_shared_fused_decode_interface"
        )
    return native_calls


def _start_decode_diagnostics(worker):
    import sys
    from collections import Counter, deque

    model = worker.model_runner.get_model()
    model = getattr(model, "language_model", model)
    runtime = getattr(model, "_xpu_moe_decode_runtime", None)
    if runtime is not None:
        runtime.history = deque(maxlen=512)
    worker.decode_provider_native_calls = _profile_decode_provider(worker, runtime)
    manager = worker.model_runner.cudagraph_manager
    replay_code = manager.run_fullgraph.__func__.__code__ if manager else None
    worker.decode_graph_replays = Counter()
    worker.decode_graph_steps = deque(maxlen=512)
    worker.decode_previous_profile = sys.getprofile()

    def observe_replay(frame, event, arg):
        if event == "return" and arg is not None and frame.f_code is replay_code:
            desc = frame.f_locals["desc"]
            worker.decode_graph_replays[(desc.num_tokens, desc.num_reqs)] += 1
            if runtime is not None and runtime.history:
                ids, step = runtime.history[-1]
                worker.decode_graph_steps.append((ids, step, desc.num_tokens))

    sys.setprofile(observe_replay)
    # PTI GPU tracing can stall existing command graphs on the tested driver.
    # Observe real replay calls here; profile native provider math separately.
    worker.decode_profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    )
    worker.decode_profiler.start()
    return [cache.data_ptr() for cache in worker.model_runner.kv_caches]


def _finish_decode_diagnostics(worker):
    import sys
    from dataclasses import asdict

    sys.setprofile(worker.decode_previous_profile)
    torch.xpu.synchronize()
    worker.decode_profiler.stop()
    native = sum(
        e.count
        for e in worker.decode_profiler.key_averages()
        if e.key == "_xpu_C::moe_shared_fused_decode_interface"
    )
    model = worker.model_runner.get_model()
    model = getattr(model, "language_model", model)
    runtime = getattr(model, "_xpu_moe_decode_runtime", None)
    result = {
        "runtime": runtime.snapshot() if runtime else None,
        "history": [(ids, asdict(step)) for ids, step in runtime.history]
        if runtime and runtime.history is not None
        else [],
        "native_calls": native,
        "provider_native_calls": worker.decode_provider_native_calls,
        "graph_replays": {
            f"{tokens}x{reqs}": count
            for (tokens, reqs), count in worker.decode_graph_replays.items()
        },
        "graph_steps": [
            (ids, asdict(step), padded)
            for ids, step, padded in worker.decode_graph_steps
        ],
        "captured_native_layers": (
            sum(e.native_launches > 0 for e in runtime.executors) if runtime else 0
        ),
        "cache_pointers": [cache.data_ptr() for cache in worker.model_runner.kv_caches],
    }
    worker.decode_profiler = None
    return result


def _run_transition_requests(llm):
    from vllm import SamplingParams

    engine = llm.llm_engine
    anchor = SamplingParams(temperature=0, max_tokens=64, ignore_eos=True, logprobs=3)
    short = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=3)
    engine.add_request("anchor", {"prompt_token_ids": [100] * 128 + [200]}, anchor)
    latest = {}
    added_pair = added_eight = False
    while engine.has_unfinished_requests():
        for output in engine.step():
            latest[output.request_id] = output
        count = len(latest["anchor"].outputs[0].token_ids) if "anchor" in latest else 0
        if count >= 2 and not added_pair:
            # A one-token prompt must run prefill despite matching a decode bucket.
            for i in range(2):
                engine.add_request(f"pair_{i}", {"prompt_token_ids": [100]}, short)
            added_pair = True
        if (
            added_pair
            and count >= 16
            and not added_eight
            and all(
                latest.get(f"pair_{i}") and latest[f"pair_{i}"].finished
                for i in range(2)
            )
        ):
            for i in range(8):
                engine.add_request(f"eight_{i}", "Explain arithmetic briefly.", short)
            added_eight = True
    assert added_pair and added_eight
    assert len(latest) == 11
    assert len(latest["anchor"].outputs[0].token_ids) == 64
    return [
        (
            list(latest[key].outputs[0].token_ids),
            latest[key].outputs[0].text,
            latest[key].outputs[0].logprobs,
        )
        for key in sorted(latest)
    ]


@pytest.mark.skipif(not current_platform.is_xpu(), reason="Requires XPU execution")
@pytest.mark.parametrize("graph_mode", ["NONE", "FULL_DECODE_ONLY"])
def test_xpu_qwen3_5_default_specialized_state_transitions(
    vllm_runner, enable_pickle, graph_mode
):
    """One real request crosses both MoE paths without moving its KV/SSM state."""
    from tests.models.utils import check_logprobs_close
    from vllm.config import CompilationConfig

    outputs = []
    for backend in ("auto", "latency_specialized"):
        with vllm_runner(
            os.environ.get("VLLM_TEST_QWEN35_MODEL", "Qwen/Qwen3.5-35B-A3B"),
            dtype="float16",
            quantization="fp8_per_tensor",
            additional_config={"decode_backend": backend},
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
            language_model_only=True,
            max_model_len=256,
            max_num_batched_tokens=128,
            enable_chunked_prefill=True,
            max_num_seqs=16,
            kv_cache_memory_bytes=128 * 1024 * 1024,
            enable_prefix_caching=False,
            compilation_config=CompilationConfig(
                mode=0,
                cudagraph_mode=graph_mode,
                cudagraph_capture_sizes=[1, 2, 4, 8] if graph_mode != "NONE" else [],
            ),
        ) as model:
            before = model.collective_rpc(_start_decode_diagnostics)
            outputs.append(_run_transition_requests(model.llm))
            reports = model.collective_rpc(_finish_decode_diagnostics)
            assert [r["cache_pointers"] for r in reports] == before
            if backend == "auto":
                assert all(
                    r["runtime"] is None
                    and r["native_calls"] == 0
                    and r["provider_native_calls"] == 0
                    for r in reports
                )
            else:
                for report in reports:
                    assert report["runtime"]["bindings_shared"]
                    assert report["runtime"]["specialized_weight_bytes"] == 0
                    assert report["runtime"]["persistent_scratch_bytes"] > 0
                    assert report["provider_native_calls"] > 0
                    if graph_mode == "NONE":
                        assert report["native_calls"] > 0
                    else:
                        assert sum(report["graph_replays"].values()) > 0
                        assert report["captured_native_layers"] == 40
                        assert all(
                            step["reason"] is None
                            and step["padded_rows"] == padded <= 8
                            for _, step, padded in report["graph_steps"]
                        )
                    steps = [step for ids, step in report["history"] if "anchor" in ids]
                    assert any(s["reason"] == "prefill_or_mixed" for s in steps)
                    assert any(
                        s["actual_rows"] == 1 and s["reason"] == "prefill_or_mixed"
                        for s in steps
                    ), "A one-token prompt tail must retain the stock prefill path"
                    boundary = next(
                        i
                        for i, s in enumerate(steps)
                        if s["actual_rows"] == 9 and s["reason"] == "token_rows"
                    )
                    assert any(s["reason"] is None for s in steps[:boundary])
                    assert any(s["reason"] is None for s in steps[boundary + 1 :])
                    assert any(
                        s["actual_rows"] == 3
                        and s["reason"] is None
                        and s["padded_rows"] == (4 if graph_mode != "NONE" else 3)
                        for s in steps
                    )
    check_logprobs_close(
        outputs_0_lst=outputs[0],
        outputs_1_lst=outputs[1],
        name_0="auto",
        name_1="latency_specialized",
    )
