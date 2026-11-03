# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CompilationConfig.cudagraph_launch_mode."""

import gc
import sys

import pytest
import regex as re
import torch

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_marks
from vllm.compilation import cudagraph_launch
from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
from vllm.compilation.counter import compilation_counter
from vllm.config import CacheConfig, CompilationConfig, VllmConfig
from vllm.config.mamba import MambaConfig
from vllm.platforms import current_platform


def _has_cuda_bindings() -> bool:
    try:
        from cuda.bindings import driver  # noqa: F401
    except ImportError:
        return False
    return True


requires_device_launch = pytest.mark.skipif(
    not (current_platform.is_cuda() and _has_cuda_bindings()),
    reason="device-launched CUDA graphs need NVIDIA CUDA and cuda-bindings",
)


def test_launch_mode_in_config_hash():
    # Startup plans are keyed by this hash, and device launch allocates
    # executable graphs at capture time, where memory profiling measures them.
    host = CompilationConfig()
    device = CompilationConfig(cudagraph_launch_mode="device")
    assert host.cudagraph_launch_mode == "host"
    assert host.compute_hash() != device.compute_hash()


def test_device_launch_config_validation(monkeypatch: pytest.MonkeyPatch):
    compilation_config = CompilationConfig(cudagraph_launch_mode="device")
    if current_platform.is_cuda():
        # A None entry makes `from cuda.bindings import driver` fail.
        monkeypatch.setitem(sys.modules, "cuda.bindings", None)
        match = "requires the cuda-bindings package"
    else:
        match = "requires NVIDIA CUDA"
    with pytest.raises(ValueError, match=match):
        VllmConfig(compilation_config=compilation_config)


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="stochastic rounding needs CUDA"
)
def test_device_launch_rejects_mamba_stochastic_rounding():
    # Stochastic rounding draws random numbers inside the captured decode graphs.
    # FlashInfer: MambaConfig rejects Triton stochastic rounding off SM100 itself.
    mamba_config = MambaConfig(backend="flashinfer", enable_stochastic_rounding=True)
    with pytest.raises(ValueError, match="per-replay RNG"):
        VllmConfig(
            compilation_config=CompilationConfig(cudagraph_launch_mode="device"),
            cache_config=CacheConfig(mamba_ssm_cache_dtype="float16"),
            mamba_config=mamba_config,
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="needs CUDA")
def test_device_launch_rejects_moe_routing_simulation(
    monkeypatch: pytest.MonkeyPatch,
):
    # The simulator draws random routing inside the captured MoE forward.
    monkeypatch.setenv("VLLM_MOE_ROUTING_SIMULATION_STRATEGY", "uniform_random")
    with pytest.raises(ValueError, match="VLLM_MOE_ROUTING_SIMULATION_STRATEGY"):
        VllmConfig(compilation_config=CompilationConfig(cudagraph_launch_mode="device"))


@pytest.fixture
def restore_vllm_stream():
    # torch.cuda.graph() enters and exits torch.cuda.stream(), which records
    # the default stream in vLLM's patched set_stream cache; restore it so
    # later tests in this process do not capture on the default stream.
    from vllm.utils.torch_utils import _current_stream_tls

    prev_stream = getattr(_current_stream_tls, "value", None)
    yield
    _current_stream_tls.value = prev_stream


@requires_device_launch
def test_device_launch_replay(restore_vllm_stream):
    x = torch.zeros(1024, device="cuda")
    graph = cudagraph_launch.new_cuda_graph("device")
    assert isinstance(graph, cudagraph_launch.DeviceLaunchCUDAGraph)
    with (
        compilation_counter.expect(
            num_cudagraph_device_launched=1, num_cudagraph_host_launch_fallbacks=0
        ),
        torch.cuda.graph(graph),
    ):
        y = x * 2 + 1
    assert graph.device_launched
    # The captured cudaGraph_t is freed after instantiation, as for host launch.
    with pytest.raises(RuntimeError, match="No cudaGraph_t is available"):
        graph.raw_cuda_graph()
    for i in range(3):
        x.fill_(i)
        graph.replay()
        torch.testing.assert_close(y, torch.full_like(x, 2 * i + 1))


@requires_device_launch
def test_device_launch_replays_on_current_stream(restore_vllm_stream):
    # vLLM replays on a different stream than it captured on, so the launch
    # must be ordered behind earlier work on the current stream.
    x = torch.zeros(1024, device="cuda")
    capture_stream = torch.cuda.Stream()
    replay_stream = torch.cuda.Stream()
    graph = cudagraph_launch.new_cuda_graph("device")
    with torch.cuda.graph(graph, stream=capture_stream):
        y = x * 2 + 1
    assert graph.device_launched
    with torch.cuda.stream(replay_stream):
        # A launch on any other stream would run during the sleep and read x
        # before the fill below.
        torch.cuda._sleep(100_000_000)
        x.fill_(5)
        graph.replay()
    replay_stream.synchronize()
    torch.testing.assert_close(y, torch.full_like(x, 11))


@requires_device_launch
def test_event_node_falls_back_to_host_launch(
    restore_vllm_stream, caplog_vllm, disable_log_dedup
):
    x = torch.zeros(16, device="cuda")
    # An external event becomes event record/wait nodes, which device launch
    # rejects.
    event = torch.cuda.Event(external=True)
    graph = cudagraph_launch.new_cuda_graph("device")
    with (
        compilation_counter.expect(
            num_cudagraph_device_launched=0, num_cudagraph_host_launch_fallbacks=1
        ),
        torch.cuda.graph(graph),
    ):
        y = x + 1
        event.record()
        torch.cuda.current_stream().wait_event(event)
    assert not graph.device_launched
    assert "could not be instantiated for device launch" in caplog_vllm.text
    assert re.search(r"node \S*(EVENT_RECORD|WAIT_EVENT)", caplog_vllm.text)
    x.fill_(41)
    graph.replay()
    torch.testing.assert_close(y, torch.full_like(x, 42))


@requires_device_launch
def test_breakable_capture_device_launches_each_segment(restore_vllm_stream):
    # Model runner V2 piecewise graphs: one graph per segment between eager breaks.
    x = torch.zeros(16, device="cuda")
    with (
        compilation_counter.expect(
            num_cudagraph_device_launched=2, num_cudagraph_host_launch_fallbacks=0
        ),
        torch.cuda.stream(torch.cuda.Stream()),
        BreakableCUDAGraphCapture(launch_mode="device") as capture,
    ):
        x.add_(1)
        capture.add_eager(lambda: x.mul_(2))
        x.add_(1)
    assert capture.num_graphs == 2
    # Capture runs only the eager segment (on zeros); replay runs all three.
    torch.accelerator.synchronize()
    capture.replay()
    torch.testing.assert_close(x, torch.full_like(x, 3))


@requires_device_launch
def test_recapture_without_reset_is_refused(restore_vllm_stream):
    # PyTorch cannot see the device-launch exec, so it would not refuse itself.
    x = torch.zeros(8, device="cuda")
    graph = cudagraph_launch.new_cuda_graph("device")
    with torch.cuda.graph(graph):
        x.add_(1)
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(RuntimeError, match="already owns a captured graph"),
    ):
        graph.capture_begin()


@requires_device_launch
def test_exec_destroyed_on_reset_and_gc(
    monkeypatch: pytest.MonkeyPatch, restore_vllm_stream
):
    destroyed = []
    destroy = cudagraph_launch._destroy_graph_exec

    def record_destroy(graph_exec):
        destroyed.append(graph_exec)
        destroy(graph_exec)

    monkeypatch.setattr(cudagraph_launch, "_destroy_graph_exec", record_destroy)
    x = torch.zeros(8, device="cuda")
    graph = cudagraph_launch.new_cuda_graph("device")
    with torch.cuda.graph(graph):
        x.add_(1)
    graph.reset()
    assert len(destroyed) == 1
    assert not graph.device_launched

    # A reset graph can be captured again.
    with torch.cuda.graph(graph):
        x.add_(1)
    assert graph.device_launched
    # Dropping the last reference destroys the exec without a GC pass.
    del graph
    assert len(destroyed) == 2
    gc.collect()
    assert len(destroyed) == 2


@requires_device_launch
def test_device_launch_memory_taken_at_capture(restore_vllm_stream):
    # CUDA graph memory profiling measures free memory around each capture, so
    # the executable graph must be allocated in capture_end(), not by the first
    # replay.
    x = torch.zeros(1024, device="cuda")
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    free_before = torch.accelerator.get_memory_info()[0]
    graphs = []
    with compilation_counter.expect(num_cudagraph_device_launched=32):
        for _ in range(32):
            graph = cudagraph_launch.new_cuda_graph("device")
            with torch.cuda.graph(graph):
                for _ in range(256):
                    x.add_(1)
            graphs.append(graph)
    torch.accelerator.synchronize()
    free_after_capture = torch.accelerator.get_memory_info()[0]
    for graph in graphs:
        graph.replay()
    torch.accelerator.synchronize()
    free_after_replay = torch.accelerator.get_memory_info()[0]
    taken_at_capture = free_before - free_after_capture
    taken_at_replay = free_after_capture - free_after_replay
    assert taken_at_replay < 2 * 1024 * 1024 < taken_at_capture
    # Capture does not run the kernels; each replay adds 256.
    torch.testing.assert_close(x, torch.full_like(x, 32 * 256))


@requires_device_launch
def test_device_launch_skips_rng_replay_prologue(restore_vllm_stream):
    # Pins the documented limitation: only PyTorch's own replay() sets up the
    # RNG state of a graph that draws from PyTorch's CUDA generator.
    generator = torch.cuda.default_generators[torch.accelerator.current_device_index()]
    host = torch.cuda.CUDAGraph()
    device = cudagraph_launch.new_cuda_graph("device")
    with torch.cuda.graph(host):
        torch.rand(64, device="cuda")
    with torch.cuda.graph(device):
        torch.rand(64, device="cuda")
    assert device.device_launched
    offset = generator.get_offset()
    host.replay()
    assert generator.get_offset() > offset
    offset = generator.get_offset()
    device.replay()
    assert generator.get_offset() == offset


def _graph_launch_state(worker) -> tuple[list[bool], int, int, int]:
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper
    from vllm.compilation.cuda_graph import CUDAGraphWrapper

    graphs = [
        entry.cudagraph
        for wrapper in CUDAGraphWrapper._all_instances
        for entry in wrapper.concrete_cudagraph_entries.values()
        if entry.cudagraph is not None
    ]
    # Model runner V2 captures its FULL graphs outside CUDAGraphWrapper.
    manager = getattr(worker.model_runner, "cudagraph_manager", None)
    manager_graphs = list(manager.graphs.values()) if manager is not None else []
    # Breakable capture keeps each segment's graph only as its bound replay().
    breakable_graphs = [
        graph
        for wrapper in BreakableCUDAGraphWrapper._all_instances
        for entry in wrapper.entries.values()
        if entry.capture is not None
        for seg in entry.capture.segments
        if isinstance(graph := getattr(seg, "__self__", None), torch.cuda.CUDAGraph)
    ]
    graphs += manager_graphs + breakable_graphs
    flags = [getattr(g, "device_launched", False) for g in graphs]
    fallbacks = compilation_counter.num_cudagraph_host_launch_fallbacks
    return flags, len(manager_graphs), len(breakable_graphs), fallbacks


@requires_device_launch
@pytest.mark.parametrize(
    "use_v2_model_runner,breakable,tp_size",
    [
        pytest.param(False, False, 1, id="mrv1"),
        pytest.param(True, False, 1, id="mrv2"),
        pytest.param(True, True, 1, id="mrv2-breakable"),
        pytest.param(True, False, 2, id="mrv2-tp2", marks=multi_gpu_marks(num_gpus=2)),
    ],
)
def test_device_launch_generation_matches_host(
    vllm_runner,
    enable_pickle,
    monkeypatch: pytest.MonkeyPatch,
    use_v2_model_runner: bool,
    breakable: bool,
    tp_size: int,
):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", str(int(use_v2_model_runner)))
    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", str(int(breakable)))
    # A worker in this process would otherwise see earlier tests' fallbacks.
    monkeypatch.setattr(compilation_counter, "num_cudagraph_host_launch_fallbacks", 0)
    prompts = ["Hello, my name is", "The capital of France is"]
    outputs = {}
    for launch_mode in ("host", "device"):
        with vllm_runner(
            "facebook/opt-125m",
            compilation_config={
                "cudagraph_launch_mode": launch_mode,
                # 16 covers the ~12-token prefill, 2 the batch-2 decode.
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            },
            tensor_parallel_size=tp_size,
            gpu_memory_utilization=0.4,
        ) as runner:
            outputs[launch_mode] = runner.generate_greedy_logprobs(
                prompts, max_tokens=16, num_logprobs=5
            )
            states = runner.collective_rpc(_graph_launch_state)
        assert len(states) == tp_size
        for flags, num_manager_graphs, num_breakable_graphs, fallbacks in states:
            assert flags
            if launch_mode == "host":
                assert not any(flags)
                continue
            if tp_size == 1:
                assert all(flags)
                assert fallbacks == 0
            else:
                # Graphs with NCCL collectives may keep PyTorch's instantiation,
                # depending on the all-reduce backend, and are counted as
                # fallbacks (with any discarded profiling captures).
                assert any(flags)
                assert flags.count(False) <= fallbacks
            if use_v2_model_runner:
                assert num_manager_graphs > 0
            if breakable:
                assert num_breakable_graphs > 0
    check_logprobs_close(
        outputs_0_lst=outputs["host"],
        outputs_1_lst=outputs["device"],
        name_0="host",
        name_1="device",
    )
