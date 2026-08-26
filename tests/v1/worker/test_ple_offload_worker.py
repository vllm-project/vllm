# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import msgspec
import pytest
import torch

import vllm.envs as envs
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.model_executor.layers import ple_offload_layer
from vllm.model_executor.layers.ple_offload_layer import PleOffloadLayer
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.v1.ple_offload import worker as ple_offload_worker
from vllm.v1.worker.gpu_worker import Worker


class _TestPleOffloadLayer(PleOffloadLayer):
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        del hidden_states, args, kwargs
        return input_ids.unsqueeze(-1)


class _WeightLoadingPleLayer(_TestPleOffloadLayer):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.bias = torch.nn.Parameter(torch.zeros(2))


class _WeightLoadingModel(torch.nn.Module):
    hf_to_vllm_mapper = WeightsMapper(orig_to_new_prefix={"checkpoint.": ""})

    def __init__(self) -> None:
        super().__init__()
        self.ple = _WeightLoadingPleLayer()
        self.received_checkpoint_names: list[str] = []

    def load_weights(self, weights) -> set[str]:
        """Record filtered names and run the normal automatic loader."""
        filtered_weights = list(weights)
        self.received_checkpoint_names = [name for name, _ in filtered_weights]
        return AutoWeightsLoader(self).load_weights(
            filtered_weights,
            mapper=self.hf_to_vllm_mapper,
        )


class _TestDefaultModelLoader:
    def __init__(self, checkpoint_names: list[str]) -> None:
        self.checkpoint_names = checkpoint_names

    def get_all_weights(self, model_config, model):
        """Return a small streamed checkpoint for weight-filtering tests."""
        del model_config, model
        return ((name, torch.ones(2)) for name in self.checkpoint_names)


def _load_test_ple_weights(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint_names: list[str],
) -> tuple[ple_offload_worker.PleOffloadRunner, _WeightLoadingModel]:
    """Run PLE weight discovery with a mapped synthetic checkpoint."""
    model = _WeightLoadingModel()
    loader = _TestDefaultModelLoader(checkpoint_names)
    monkeypatch.setattr(
        ple_offload_worker,
        "initialize_model",
        lambda **_: model,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "DefaultModelLoader",
        _TestDefaultModelLoader,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "get_model_loader",
        lambda _: loader,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "process_weights_after_loading",
        lambda *args: None,
    )

    runner = ple_offload_worker.PleOffloadRunner.__new__(
        ple_offload_worker.PleOffloadRunner
    )
    runner.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32),
        load_config=SimpleNamespace(),
    )
    runner._layers = {}
    runner._load_weights()
    return runner, model


def test_ple_offload_loads_mapped_checkpoint_names(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_names = [
        "checkpoint.ple.weight",
        "checkpoint.unrelated.weight",
        "checkpoint.ple.bias",
    ]

    with caplog.at_level("INFO", logger=ple_offload_worker.__name__):
        runner, model = _load_test_ple_weights(monkeypatch, checkpoint_names)

    assert model.received_checkpoint_names == [
        "checkpoint.ple.weight",
        "checkpoint.ple.bias",
    ]
    assert runner.layer_names == ["ple"]
    assert "matched 2 checkpoint tensor(s)" in caplog.text
    assert "verified 2/2 materialized parameter(s)" in caplog.text


def test_ple_offload_rejects_checkpoint_without_matching_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="filter matched no weights"):
        _load_test_ple_weights(
            monkeypatch,
            ["checkpoint.unrelated.weight"],
        )


def test_ple_offload_rejects_missing_materialized_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match=r"parameters: \['ple.bias'\]"):
        _load_test_ple_weights(
            monkeypatch,
            ["checkpoint.ple.weight"],
        )


def test_ple_offload_wait_only_waits_for_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_calls = []
    error = SimpleNamespace(value=0)
    stream = SimpleNamespace(cuda_stream=17)
    flag_tensor = torch.zeros(1, dtype=torch.int32)

    def fake_wait(*args: object) -> tuple[SimpleNamespace]:
        wait_calls.append(args)
        return (error,)

    monkeypatch.setattr(
        ple_offload_layer.torch.cuda,
        "current_stream",
        lambda: stream,
    )
    monkeypatch.setattr(
        ple_offload_layer.cuda_driver,
        "CUstream",
        lambda value: value,
    )
    monkeypatch.setattr(
        ple_offload_layer.cuda_driver,
        "CUdeviceptr",
        lambda value: value,
    )
    monkeypatch.setattr(
        ple_offload_layer.cuda_driver,
        "cuStreamWaitValue32",
        fake_wait,
    )
    monkeypatch.setattr(
        ple_offload_layer.cuda_driver,
        "cuStreamWriteValue32",
        lambda *args: pytest.fail(f"wait unexpectedly wrote the flag: {args}"),
    )

    result = ple_offload_layer._ple_offload_wait_impl(
        flag_tensor,
        torch.empty(4, 2),
        torch.empty(4, 2),
    )

    assert result is None
    assert wait_calls == [
        (
            stream.cuda_stream,
            flag_tensor.data_ptr(),
            ple_offload_layer.CpuGpuSemaphore.DONE_VALUE,
            ple_offload_layer.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        )
    ]


def test_offloaded_forward_waits_then_releases_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wait_calls = []
    reset_calls = []
    flag_tensor = torch.zeros(1, dtype=torch.int32)
    output_buffer = torch.arange(12).reshape(6, 2)

    monkeypatch.setattr(
        torch.ops.vllm,
        "ple_offload_wait",
        lambda *args: wait_calls.append(args),
    )

    layer = _TestPleOffloadLayer()
    layer._is_cpu_offloaded = True
    layer._gpu_output_buffer = output_buffer
    layer._sem = SimpleNamespace(
        flag_tensor=flag_tensor,
        reset=lambda stream: reset_calls.append(stream),
    )
    hidden_states = torch.zeros(3, 2)
    input_ids = torch.arange(3)

    output = layer(hidden_states, input_ids)
    stream = object()
    layer.release_offloaded_output(stream)  # type: ignore[arg-type]

    assert wait_calls == [
        (
            flag_tensor,
            output_buffer,
            hidden_states,
        )
    ]
    assert output.data_ptr() == output_buffer.data_ptr()
    torch.testing.assert_close(output, output_buffer[: input_ids.shape[0]])
    assert reset_calls == [stream]


def test_ple_offload_request_msgpack_round_trip() -> None:
    request = ple_offload_worker.PleOffloadRequest(
        dp_rank=2,
        num_tokens=17,
        num_reqs=3,
    )

    decoded = ple_offload_worker._PLE_OFFLOAD_REQUEST_DECODER.decode(
        msgspec.msgpack.encode(request)
    )

    assert decoded == request


@pytest.mark.parametrize(
    ("ple_layer_ids", "expected"),
    [
        ([1], True),
        ([], False),
    ],
)
def test_ple_offload_requires_ple_layers(
    monkeypatch: pytest.MonkeyPatch,
    ple_layer_ids: list[int],
    expected: bool,
) -> None:
    worker = Worker.__new__(Worker)
    worker.model_config = SimpleNamespace(  # type: ignore[assignment]
        hf_text_config=SimpleNamespace(ple_layer_ids=ple_layer_ids)
    )
    monkeypatch.setattr(envs, "VLLM_PLE_CPU_OFFLOAD", True)

    assert worker._has_ple_layers() is expected


@pytest.mark.parametrize(
    ("architecture", "enable_expert_parallel", "unsupported_setting"),
    [
        ("Qwen4ExpForCausalLM", False, None),
        ("Qwen4ExpForConditionalGeneration", True, None),
        ("UnsupportedArchitecture", True, "architecture"),
    ],
)
def test_ple_offload_accepts_supported_configurations(
    monkeypatch: pytest.MonkeyPatch,
    architecture: str,
    enable_expert_parallel: bool,
    unsupported_setting: str | None,
) -> None:
    worker = Worker.__new__(Worker)
    worker.use_v2_model_runner = True
    worker.parallel_config = SimpleNamespace(
        distributed_executor_backend="mp",
        nnodes=1,
        data_parallel_backend="mp",
        data_parallel_size_local=1,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=enable_expert_parallel,
        use_ubatching=False,
    )
    worker.model_config = SimpleNamespace(architecture=architecture)
    worker.vllm_config = SimpleNamespace(weight_transfer_config=None)
    monkeypatch.setattr(gpu_worker_module.current_platform, "is_cuda", lambda: True)

    if unsupported_setting is None:
        worker._validate_ple_offload_config()
    else:
        with pytest.raises(ValueError) as exc_info:
            worker._validate_ple_offload_config()
        assert f"Unsupported settings: {unsupported_setting}" in str(exc_info.value)
        assert architecture not in str(exc_info.value)


@pytest.mark.parametrize(
    ("dp_rank", "expected_calls"),
    [(0, 1), (1, 0)],
)
def test_only_dp0_tp0_spawns_shared_ple_offload_worker(
    monkeypatch: pytest.MonkeyPatch,
    dp_rank: int,
    expected_calls: int,
) -> None:
    calls = []
    worker = Worker.__new__(Worker)
    worker._ple_offload_enabled = True
    worker._ple_offload_worker_handle = None
    worker.rank = 0
    worker.local_rank = 0
    worker.vllm_config = SimpleNamespace()
    worker.parallel_config = SimpleNamespace(
        data_parallel_rank=dp_rank,
        data_parallel_size=2,
        tensor_parallel_size=2,
        _ple_offload_ipc_path="ipc:///tmp/test-ple-offload",
    )
    handle = object()

    def fake_make_process(*args: object) -> object:
        calls.append(args)
        return handle

    monkeypatch.setattr(
        ple_offload_worker.PleOffloadWorker,
        "make_process",
        fake_make_process,
    )

    worker.spawn_ple_offload()

    assert len(calls) == expected_calls
    if expected_calls:
        assert calls == [
            (
                worker.vllm_config,
                4,
                "ipc:///tmp/test-ple-offload",
            )
        ]
        assert worker._ple_offload_worker_handle is handle


def test_offload_distributed_sets_config_only_for_model_parallel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vllm_config = VllmConfig()
    calls = []

    # The Offload subprocess may inherit DP environment variables from a GPU
    # worker, but its isolated model-parallel world must always remain DP1.
    monkeypatch.setattr(envs, "VLLM_DP_SIZE", 2)
    monkeypatch.setattr(envs, "VLLM_DP_RANK", 1)
    monkeypatch.setattr(envs, "VLLM_DP_RANK_LOCAL", 1)

    monkeypatch.setattr(
        ple_offload_worker.dist,
        "is_initialized",
        lambda: False,
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "init_distributed_environment",
        lambda **kwargs: calls.append(
            ("world", get_current_vllm_config_or_none(), kwargs)
        ),
    )
    monkeypatch.setattr(
        ple_offload_worker,
        "ensure_model_parallel_initialized",
        lambda **kwargs: calls.append(
            ("model_parallel", get_current_vllm_config_or_none(), kwargs)
        ),
    )
    monkeypatch.setattr(
        ple_offload_worker.tempfile,
        "mkdtemp",
        lambda **_: "/tmp/test-ple-offload",
    )

    ple_offload_worker._init_offload_distributed()

    offload_config = calls[1][1]
    assert offload_config is not vllm_config
    assert offload_config.parallel_config.data_parallel_size == 1
    assert offload_config.parallel_config.tensor_parallel_size == 1
    assert offload_config.parallel_config.pipeline_parallel_size == 1
    assert calls == [
        (
            "world",
            None,
            {
                "world_size": 1,
                "rank": 0,
                "distributed_init_method": "file:///tmp/test-ple-offload/store",
                "local_rank": 0,
                "backend": "gloo",
            },
        ),
        (
            "model_parallel",
            offload_config,
            {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
                "backend": "gloo",
            },
        ),
    ]
    assert get_current_vllm_config_or_none() is None


def test_ple_offload_runner_groups_registrations_by_dp_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSocket:
        def __init__(self, registrations):
            self.registrations = iter(registrations)

        def recv(self):
            return next(self.registrations)

    class FakeStream:
        pass

    runner = ple_offload_worker.PleOffloadRunner.__new__(
        ple_offload_worker.PleOffloadRunner
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=2,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        model_config=SimpleNamespace(
            dtype=torch.float32,
            hf_text_config=SimpleNamespace(ple_embed_dim=2),
        ),
    )
    runner._layers = {
        "ple": SimpleNamespace(get_offload_output_dtype=lambda default: default)
    }
    runner._worker_targets = {}
    runner._pinned_bufs = {}
    runner._input_bufs = {}

    registrations = []
    for dp_rank in range(2):
        for tp_rank in range(2):
            registrations.append(
                ple_offload_worker.PleOffloadRegistration(
                    worker_id=dp_rank * 2 + tp_rank,
                    dp_rank=dp_rank,
                    tp_rank=tp_rank,
                    gpu_output_buffers={"ple": torch.empty(8, 2)},
                    sem_flag_tensors={"ple": torch.zeros(1, dtype=torch.int32)},
                    input_ids_buf=torch.full((8,), dp_rank, dtype=torch.int32),
                    query_start_loc_buf=torch.zeros(4, dtype=torch.int32),
                    ngram_context_buf=None,
                )
            )

    original_empty = torch.empty

    def unpinned_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(ple_offload_worker.pickle, "loads", lambda item: item)
    monkeypatch.setattr(ple_offload_worker.torch, "empty", unpinned_empty)
    monkeypatch.setattr(
        ple_offload_worker.torch.cuda,
        "Stream",
        lambda **_: FakeStream(),
    )
    monkeypatch.setattr(
        ple_offload_worker.CpuGpuSemaphore,
        "from_ipc_tensor",
        lambda _: SimpleNamespace(),
    )

    runner.accept_registrations(FakeSocket(registrations), len(registrations))

    assert set(runner._worker_targets) == {0, 1}
    assert [target.tp_rank for target in runner._worker_targets[0]["ple"]] == [0, 1]
    assert [target.tp_rank for target in runner._worker_targets[1]["ple"]] == [0, 1]
    assert set(runner._input_bufs) == {0, 1}
    assert runner._input_bufs[0].input_ids_buf[0].item() == 0
    assert runner._input_bufs[1].input_ids_buf[0].item() == 1
    assert set(runner._pinned_bufs) == {0, 1}


def test_ple_offload_runner_routes_requests_layer_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events = []

    class FakeLayer:
        def __init__(self, name: str):
            self.name = name

        def forward_impl(
            self,
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
            output_buffer,
        ):
            del hidden_states, query_start_loc, ngram_context
            events.append((self.name, int(input_ids[0].item())))
            result = input_ids.unsqueeze(-1).expand(-1, 2)
            output_buffer[: result.shape[0]].copy_(result)
            return output_buffer[: result.shape[0]]

    class FakeStream:
        def synchronize(self) -> None:
            pass

    class FakeSemaphore:
        def wait_reset(self, stream) -> None:
            del stream

        def signal(self, stream) -> None:
            del stream

    def target():
        return ple_offload_worker.PleOffloadOutputTarget(
            tp_rank=0,
            gpu_output_buffer=torch.empty(4, 2, dtype=torch.int32),
            sem=FakeSemaphore(),
            copy_stream=FakeStream(),  # type: ignore[arg-type]
        )

    runner = ple_offload_worker.PleOffloadRunner.__new__(
        ple_offload_worker.PleOffloadRunner
    )
    runner._clamp_input_ids = True
    runner._layers = {"ple0": FakeLayer("ple0"), "ple1": FakeLayer("ple1")}
    runner._worker_targets = {
        0: {"ple0": [target()], "ple1": [target()]},
        1: {"ple0": [target()], "ple1": [target()]},
    }
    runner._input_bufs = {
        0: ple_offload_worker.PleOffloadInputBuffers(
            input_ids_buf=torch.tensor([-1, 11], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 2], dtype=torch.int32),
            ngram_context_buf=None,
        ),
        1: ple_offload_worker.PleOffloadInputBuffers(
            input_ids_buf=torch.tensor([20], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 1], dtype=torch.int32),
            ngram_context_buf=None,
        ),
    }
    runner._pinned_bufs = {
        dp_rank: {
            layer_name: torch.empty(4, 2, dtype=torch.int32)
            for layer_name in runner._layers
        }
        for dp_rank in range(2)
    }
    monkeypatch.setattr(
        ple_offload_worker.torch.cuda,
        "stream",
        lambda _: nullcontext(),
    )

    runner._handle_requests(
        [
            ple_offload_worker.PleOffloadRequest(
                dp_rank=0,
                num_tokens=2,
                num_reqs=1,
            ),
            ple_offload_worker.PleOffloadRequest(
                dp_rank=1,
                num_tokens=1,
                num_reqs=1,
            ),
        ]
    )

    assert events == [
        ("ple0", 0),
        ("ple0", 20),
        ("ple1", 0),
        ("ple1", 20),
    ]
    torch.testing.assert_close(
        runner._worker_targets[0]["ple1"][0].gpu_output_buffer[:2],
        torch.tensor([[0, 0], [11, 11]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        runner._worker_targets[1]["ple1"][0].gpu_output_buffer[:1],
        torch.tensor([[20, 20]], dtype=torch.int32),
    )


def test_wait_for_ready_closes_pipe() -> None:
    context = ple_offload_worker.get_mp_context()
    ready_reader, ready_writer = context.Pipe(duplex=False)
    ready_writer.send(
        {
            "status": ple_offload_worker.PleOffloadWorker.READY_STR,
            "layer_names": ["layers.0.ple.ple_embedding"],
        }
    )
    ready_writer.close()
    handle = ple_offload_worker.PleOffloadWorkerHandle(
        proc=None,
        death_writer=None,
        ready_pipe_reader=ready_reader,
    )

    ple_offload_worker.PleOffloadWorker.wait_for_ready(handle)

    assert handle.ready_pipe_reader is None
