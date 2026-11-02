# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import cast

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm.config.lora import LoRAConfig
from vllm.lora.layers import (
    ColumnParallelLinearWithShardedLoRA,
    LoRARouteMapping,
    MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithShardedLoRA,
)
from vllm.lora.punica_wrapper.punica_cpu import PunicaWrapperCPU
from vllm.lora.punica_wrapper.utils import RoutedLoRATensors, convert_route_mapping
from vllm.lora.request import (
    LoRARequest,
    LoRARequestLike,
    LoRARoutingRequest,
    iter_lora_int_ids,
    iter_lora_requests,
)
from vllm.lora.routing_utils import add_lora_request, remove_lora_request
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    LinearMethodBase,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.parameter import ModelWeightParameter
from vllm.model_executor.utils import set_weight_attrs
from vllm.v1.worker.gpu.lora_utils import LoraState

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def routed_kernel_limit(request, monkeypatch):
    """Simulate hardware limits at the external CUDA driver boundary."""
    from vllm.platforms import current_platform

    limit = request.param
    if not current_platform.is_cuda():
        yield limit
        return
    import ctypes

    from vllm.lora.ops.triton_ops import routed_lora_op

    support = getattr(routed_lora_op, "is_routed_lora_supported", None)
    if support is not None:
        support.cache_clear()
    routed_lora_op._resident_capacity.clear()
    driver = routed_lora_op._libcuda()
    original_attribute = driver.cuDeviceGetAttribute
    original_occupancy = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor
    queries = 0

    def attribute(result, kind, device):
        if limit == "cooperative_launch" and kind.value == 95:
            ctypes.cast(result, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0
        return original_attribute(result, kind, device)

    def occupancy(result, function, threads, shared):
        nonlocal queries
        queries += 1
        if limit == "cooperative_launch":
            raise AssertionError("Unsupported cooperative launch reached preparation")
        if limit == "driver_error":
            return 1
        if limit == "capacity" or (limit == "later_slice" and queries == 2):
            ctypes.cast(result, ctypes.POINTER(ctypes.c_int))[0] = 0
            return 0
        return original_occupancy(result, function, threads, shared)

    monkeypatch.setattr(driver, "cuDeviceGetAttribute", attribute)
    monkeypatch.setattr(
        driver, "cuOccupancyMaxActiveBlocksPerMultiprocessor", occupancy
    )
    if limit == "bf16_hardware":
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (7, 5))
    if limit in ("capacity", "graph", "graph_v1"):
        # Keep the engine in this process so the external driver override applies.
        monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if limit in ("graph", "graph_v1"):
        monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if limit == "graph" else "0")
    if limit == "graph":
        monkeypatch.setenv("VLLM_GPU_SYNC_CHECK", "error")
    if limit == "shared_memory":
        from triton.runtime import driver as triton_driver
        from triton.runtime.errors import OutOfResources

        original_load = triton_driver.active.utils.load_binary

        def load_binary(name, kernel, shared, device):
            if name == "_routed_lora_kernel":
                raise OutOfResources(shared, 0, "shared memory")
            return original_load(name, kernel, shared, device)

        monkeypatch.setattr(triton_driver.active.utils, "load_binary", load_binary)
    try:
        yield limit
    finally:
        if support is not None:
            support.cache_clear()
        routed_lora_op._resident_capacity.clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [0, 1, 19, 129, 1500, 8192])
def test_routed_triton_matches_reference_and_replays_changed_routes(dtype, tokens):
    """Plans, output slices, absent routes and self-clearing scratch, including
    batches where each team pipelines several tiles, survive graph replay."""
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("Requires an NVIDIA GPU")
    from vllm.lora.ops.triton_ops.routed_lora_op import routed_lora
    from vllm.lora.punica_wrapper.utils import make_route_plans

    generator = torch.Generator().manual_seed(42)
    hidden, output, rank, adapters, routes = 33, 65, 8, 4, 3
    x = torch.randn(tokens, hidden + 1, generator=generator).cuda().to(dtype)[:, 1:]
    a = (
        (torch.randn(adapters, 1, rank, hidden, generator=generator) / hidden**0.5)
        .cuda()
        .to(dtype)
    )
    b = (
        (torch.randn(adapters, 1, output, rank, generator=generator) / rank**0.5)
        .cuda()
        .to(dtype)
    )
    initial = torch.randn(tokens, output + 4, generator=generator).cuda().to(dtype)
    y = initial.clone()

    def make_slots(shift):
        # Requests of 7 tokens share a route plan; every third token is idle.
        slots = (
            torch.arange(tokens)[:, None] // 7 + torch.arange(routes) + shift
        ) % adapters
        slots[:, -1] = -1
        slots[2::3] = -1
        return slots

    plans, sorted_tokens, starts = (
        torch.from_numpy(array).cuda()
        for array in make_route_plans(make_slots(0).numpy())
    )
    weights = torch.zeros(tokens, routes, device="cuda")

    def update(shift):
        # Shifting every adapter keeps the plan structure, so graph inputs keep
        # their shapes while slots and coefficients change.
        slots = make_slots(shift)
        new_plans, new_sorted, new_starts = make_route_plans(slots.numpy())
        plans.copy_(torch.from_numpy(new_plans))
        sorted_tokens.copy_(torch.from_numpy(new_sorted))
        starts.copy_(torch.from_numpy(new_starts))
        coefficients = torch.full((tokens, routes), 0.3 + shift * 0.1)
        coefficients[::2, 0] = 0
        weights.copy_(coefficients)
        return slots.cuda()

    def check(slots):
        expected = initial.float()
        for route in range(routes):
            for adapter in range(adapters):
                rows = slots[:, route] == adapter
                delta = (
                    x[rows].float() @ a[adapter, 0].float().T @ b[adapter, 0].float().T
                )
                expected[rows, 2 : 2 + output] += weights[rows, route, None] * delta
        torch.testing.assert_close(y, expected.to(dtype), atol=6e-2, rtol=6e-2)
        if tokens == 1:
            assert not torch.equal(y[:, 2 : 2 + output], initial[:, 2 : 2 + output])
        assert torch.equal(y[:, :2], initial[:, :2])
        assert torch.equal(y[:, 2 + output :], initial[:, 2 + output :])
        assert torch.equal(y[2::3], initial[2::3])

    def run():
        routed_lora(x, a, b, y, weights, plans, sorted_tokens, starts, 2)

    slots = update(0)
    run()
    check(slots)
    # A second eager call verifies the first left scratch and counters clean.
    y.copy_(initial)
    run()
    check(slots)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for shift in (1, 2):
        slots = update(shift)
        y.copy_(initial)
        graph.replay()
        check(slots)


@pytest.fixture(params=["opt", "llama"])
def tiny_composition_model(tmp_path, request):
    from peft import LoraConfig, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM

    torch.manual_seed(42)
    model_class, config_class = (
        (OPTForCausalLM, OPTConfig)
        if request.param == "opt"
        else (LlamaForCausalLM, LlamaConfig)
    )
    dimensions = (
        {"word_embed_proj_dim": 64, "ffn_dim": 64}
        if request.param == "opt"
        else {"intermediate_size": 64, "num_key_value_heads": 2}
    )
    base = model_class(
        config_class(
            vocab_size=64,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            max_position_embeddings=256,
            **dimensions,
        )
    ).eval()
    model_path = tmp_path / "base"
    base.save_pretrained(model_path)
    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        init_lora_weights=False,
        task_type="CAUSAL_LM",
    )
    reference = get_peft_model(base, config)
    reference.add_adapter("second", config)
    adapters_path = tmp_path / "adapters"
    reference.save_pretrained(adapters_path)
    adapters = (
        LoRARequest("first", 1, str(adapters_path)),
        LoRARequest("second", 2, str(adapters_path / "second")),
    )
    return str(model_path), reference.eval(), adapters


@torch.inference_mode()
def assert_generation_matches_reference(reference, prompt, completion, num_tokens):
    assert len(completion.token_ids) == num_tokens
    assert len(completion.logprobs) == num_tokens
    # Evaluate every prediction against its actual generated prefix, including
    # decode steps after vLLM has started reusing its KV cache.
    token_ids = prompt + list(completion.token_ids[:-1])
    expected = (
        reference(torch.tensor([token_ids]), use_cache=False)
        .logits[0, len(prompt) - 1 :]
        .log_softmax(-1)
    )
    actual = torch.tensor(
        [
            [scores[token].logprob for token in range(expected.shape[-1])]
            for scores in completion.logprobs
        ]
    )
    torch.testing.assert_close(actual, expected.cpu(), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "prefix_caching,routed_kernel_limit",
    [
        (False, "supported"),
        (True, "supported"),
        (False, "capacity"),
        (False, "graph"),
        (True, "graph"),
        (False, "graph_v1"),
    ],
    indirect=["routed_kernel_limit"],
)
def test_generate_composition_matches_peft_and_isolates_prefix_cache(
    tiny_composition_model,
    vllm_runner,
    prefix_caching,
    routed_kernel_limit,
    monkeypatch,
):
    from vllm import SamplingParams

    model_path, reference, adapters = tiny_composition_model
    prompt = [3, 4, 5, 6] * 40
    num_tokens = 8
    params = SamplingParams(
        temperature=0,
        max_tokens=num_tokens,
        ignore_eos=True,
        logprobs=64,
        detokenize=False,
    )
    graph_mode = routed_kernel_limit in ("graph", "graph_v1")
    if graph_mode:
        from vllm.platforms import current_platform

        if not current_platform.is_cuda():
            pytest.skip("Routed CUDA graphs require an NVIDIA GPU")
    replays = 0
    if graph_mode:
        replay = torch.cuda.CUDAGraph.replay

        def count_replays(graph):
            nonlocal replays
            replays += 1
            return replay(graph)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", count_replays)
    compilation_config = (
        {
            "cudagraph_mode": "FULL",
            "cudagraph_capture_sizes": [1, 2, 4],
            "max_cudagraph_capture_size": 4,
        }
        if graph_mode
        else {}
    )
    with vllm_runner(
        model_path,
        dtype="float16",
        enforce_eager=not graph_mode,
        compilation_config=compilation_config,
        skip_tokenizer_init=True,
        enable_lora=True,
        max_loras=2 if graph_mode else 4,
        max_loras_per_request=2,
        max_lora_rank=8,
        lora_target_modules=["q_proj", "v_proj"],
        max_model_len=192,
        block_size=128,
        kv_cache_memory_bytes=16 * 1024 * 1024,
        max_num_seqs=4,
        max_logprobs=64,
        enable_prefix_caching=prefix_caching,
    ) as runner:
        llm = runner.get_llm()
        for index, weights in enumerate(((0.7, 0.3), (0.3, 0.7))):
            name = f"reference_{index}"
            reference.add_weighted_adapter(
                ["default", "second"],
                list(weights),
                name,
                combination_type="cat",
            )
            reference.set_adapter(name)
            route = make_routing_request(adapters, weights)
            replays_before_generate = replays
            output = llm.generate(
                [{"prompt_token_ids": prompt}],
                params,
                lora_request=route,
                use_tqdm=False,
            )[0]
            if graph_mode:
                assert replays > replays_before_generate
            assert output.lora_request.lora_weights == weights
            assert (output.num_cached_tokens or 0) == 0
            assert_generation_matches_reference(
                reference, prompt, output.outputs[0], num_tokens
            )
            repeated = llm.generate(
                [{"prompt_token_ids": prompt}],
                params,
                lora_request=route,
                use_tqdm=False,
            )[0]
            if prefix_caching:
                assert repeated.num_cached_tokens > 0
            assert_generation_matches_reference(
                reference, prompt, repeated.outputs[0], num_tokens
            )

        selections = [route, adapters[0], None]
        replays_before_mixed = replays
        mixed = llm.generate(
            [{"prompt_token_ids": prompt}] * 3,
            params,
            lora_request=selections,
            use_tqdm=False,
        )
        if graph_mode:
            assert replays > replays_before_mixed
        for output, selection in zip(mixed, selections):
            assert output.lora_request == selection
            if selection is None:
                with reference.disable_adapter():
                    assert_generation_matches_reference(
                        reference, prompt, output.outputs[0], num_tokens
                    )
            else:
                reference.set_adapter(
                    "default" if isinstance(selection, LoRARequest) else name
                )
                assert_generation_matches_reference(
                    reference, prompt, output.outputs[0], num_tokens
                )

        if graph_mode:
            alias = LoRARequest("replacement", 3, adapters[0].lora_path)
            replacement = make_routing_request((adapters[1], alias), (0.7, 0.3))
            replays_before_replacement = replays
            replaced = llm.generate(
                [{"prompt_token_ids": prompt}],
                params,
                lora_request=replacement,
                use_tqdm=False,
            )[0]
            reference.set_adapter(name)
            assert_generation_matches_reference(
                reference, prompt, replaced.outputs[0], num_tokens
            )
            assert replays > replays_before_replacement


@pytest.mark.parametrize(
    "eager,limit,fully_sharded,ubatch_size,error",
    [
        (False, 2, True, 1, "enforce_eager"),
        (False, 2, False, 2, "microbatching"),
        (True, 2, False, 2, "microbatching"),
        (True, 1, False, 1, "max_loras_per_request"),
    ],
)
def test_input_processor_rejects_unsupported_composition(
    tiny_composition_model,
    eager,
    limit,
    fully_sharded,
    ubatch_size,
    error,
):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import EngineArgs
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.engine.input_processor import InputProcessor

    model_path, _, adapters = tiny_composition_model
    config = EngineArgs(
        model=model_path,
        skip_tokenizer_init=True,
        enforce_eager=eager,
        enable_lora=True,
        max_loras=2,
        max_loras_per_request=limit,
        fully_sharded_loras=fully_sharded,
        ubatch_size=ubatch_size,
        all2all_backend=(
            "deepep_low_latency" if ubatch_size > 1 else "allgather_reducescatter"
        ),
    ).create_engine_config()
    processor = InputProcessor(config)
    with pytest.raises(VLLMValidationError, match=error):
        processor.process_inputs(
            "rejected",
            {"prompt_token_ids": [3, 4]},
            SamplingParams(max_tokens=1),
            ("generate",),
            lora_request=make_routing_request(adapters, (0.7, 0.3)),
        )


def fake_quantize_activation(x: torch.Tensor, num_bits: int) -> torch.Tensor:
    max_int = 2 ** (num_bits - 1) - 1
    min_int = -(2 ** (num_bits - 1))
    scale = torch.amax(torch.abs(x), dim=-1, keepdim=True) / max_int
    scale = torch.clamp(scale, min=torch.finfo(x.dtype).eps)
    quantized_x = torch.clamp(torch.round(x / scale), min_int, max_int)
    return quantized_x * scale


class FakeQuantLinearMethod(LinearMethodBase):
    def __init__(self, num_bits: int) -> None:
        self.num_bits = num_bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight_loader = extra_weight_attrs.pop("weight_loader")
        weight = ModelWeightParameter(
            data=torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        set_weight_attrs(weight, extra_weight_attrs)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(fake_quantize_activation(x, self.num_bits), layer.weight, bias)


class FakeQuantConfig(QuantizationConfig):
    def __init__(self, num_bits: int = 4) -> None:
        super().__init__()
        self.num_bits = num_bits

    def get_name(self) -> QuantizationMethods:
        return cast(QuantizationMethods, "custom_quant")

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "FakeQuantConfig":
        return cls(num_bits=config.get("num_bits", 4))

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> FakeQuantLinearMethod | None:
        if isinstance(layer, LinearBase):
            return FakeQuantLinearMethod(self.num_bits)
        return None


def make_lora_request(lora_id: int) -> LoRARequest:
    return LoRARequest(
        lora_name=f"lora_{lora_id}",
        lora_int_id=lora_id,
        lora_path="/tmp/lora",
    )


@pytest.mark.parametrize("limit", [0, -1, 3])
def test_per_request_limit_rejects_invalid_config(limit):
    with pytest.raises(ValueError):
        LoRAConfig(max_loras=2, max_loras_per_request=limit)


def test_per_request_limit_defaults_to_one():
    ordinary = LoRAConfig(max_loras=4)
    routed = LoRAConfig(max_loras=4, max_loras_per_request=2)
    assert ordinary.max_loras_per_request == 1
    assert routed.max_loras_per_request == 2
    assert ordinary.compute_hash() != routed.compute_hash()


def test_per_request_limit_cli_plumbing():
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(
        [
            "--enable-lora",
            "--max-loras",
            "4",
            "--max-loras-per-request",
            "2",
        ]
    )
    args = EngineArgs.from_cli_args(parsed)
    assert args.max_loras == 4
    assert args.max_loras_per_request == 2


@pytest.mark.parametrize("weights", [(0.7, 0.3), (0.3, 0.7)])
def test_routed_request_engine_transport(weights):
    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.request import Request
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    route = make_routing_request((make_lora_request(1), make_lora_request(2)), weights)
    request = EngineCoreRequest(
        request_id="route",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        lora_routing_request=route,
    )
    decoded = MsgpackDecoder(EngineCoreRequest).decode(MsgpackEncoder().encode(request))
    resolved = decoded.resolved_lora_request
    assert isinstance(resolved, LoRARoutingRequest)
    assert resolved.lora_requests == route.lora_requests
    assert resolved.lora_weights == weights
    assert resolved.cache_fingerprint == route.cache_fingerprint
    assert Request.from_engine_core_request(decoded, None).lora_request == route
    decoded.lora_request = make_lora_request(3)
    with pytest.raises(ValueError, match="only one"):
        _ = decoded.resolved_lora_request


@pytest.mark.parametrize("adapter", [None, LoRARequest("scalar", 1, "/tmp/scalar")])
def test_ordinary_engine_transport_accepts_missing_route_field(adapter):
    import msgspec

    from vllm import SamplingParams
    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

    request = EngineCoreRequest(
        request_id="scalar",
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=adapter,
        cache_salt=None,
        data_parallel_rank=None,
    )
    encoder = MsgpackEncoder()
    raw = msgspec.msgpack.decode(encoder.encode(request)[0])
    assert raw[-1] is None
    old_payload = msgspec.msgpack.encode(raw[:-1])
    decoded = MsgpackDecoder(EngineCoreRequest).decode([old_payload])
    assert decoded.resolved_lora_request == adapter


def test_routed_offline_broadcast_and_prompt_selection():
    from vllm.entrypoints.common.offline import OfflineInferenceMixin
    from vllm.exceptions import VLLMValidationError

    mixin = OfflineInferenceMixin()
    adapter = make_lora_request(1)
    route = make_routing_request((adapter, make_lora_request(2)), (0.7, 0.3))
    assert mixin._lora_request_to_seq(route, 2) == [route, route]
    selections = [route, adapter, None]
    assert mixin._lora_request_to_seq(selections, 3) == selections
    with pytest.raises(VLLMValidationError, match="lengths"):
        mixin._lora_request_to_seq(selections, 2)
    with pytest.raises(VLLMValidationError, match="text-only"):
        mixin._resolve_mm_lora({"type": "multimodal"}, route)


def make_routing_request(
    lora_requests: tuple[LoRARequest, ...],
    lora_weights: tuple[float, ...],
) -> LoRARoutingRequest:
    return LoRARoutingRequest(
        routing_name="routed",
        routing_int_id=100,
        lora_requests=lora_requests,
        lora_weights=lora_weights,
    )


def set_tp_rank(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    tp_size: int,
) -> None:
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_rank",
        lambda: rank,
    )
    monkeypatch.setattr(
        "vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: rank,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: tp_size,
    )


def set_routed_mapping(
    lora_linear,
    route_mapping: LoRARouteMapping,
    lora_config: LoRAConfig,
) -> PunicaWrapperCPU:
    punica_wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=8,
        max_batches=4,
        device=torch.device("cpu"),
    )
    lora_linear.set_mapping(punica_wrapper)
    punica_wrapper.update_metadata(
        route_mapping,
        lora_index_to_id=[11, 12, None, None],
        max_loras=lora_config.max_loras,
        vocab_size=128,
    )
    return punica_wrapper


def make_weight(start: int, rows: int, cols: int) -> torch.Tensor:
    values = torch.arange(start, start + rows * cols, dtype=torch.float32)
    return ((values % 17) - 8).reshape(rows, cols) / 10


def build_column_gather_calls(lora_layers, x: torch.Tensor):
    route_mapping = lora_layers[0].punica_wrapper.lora_route_mapping
    token_lora_indices, token_lora_weights = route_mapping
    calls = []
    for slice_idx in range(lora_layers[0].n_slices):
        for lora_idx, token_mask, _ in lora_layers[0]._iter_lora_route_groups(
            token_lora_indices, token_lora_weights
        ):
            local_shrinks = []
            for lora_layer in lora_layers:
                lora_a = lora_layer.lora_a_stacked[slice_idx]
                local_shrinks.append(
                    x[token_mask].to(dtype=lora_a.dtype) @ lora_a[lora_idx, 0].T
                )
            calls.append((local_shrinks, torch.cat(local_shrinks, dim=-1)))
    return calls


def build_row_reduce_calls(lora_layers, x_shards: list[torch.Tensor]):
    route_mapping = lora_layers[0].punica_wrapper.lora_route_mapping
    token_lora_indices, token_lora_weights = route_mapping
    calls = []
    for lora_idx, token_mask, _ in lora_layers[0]._iter_lora_route_groups(
        token_lora_indices, token_lora_weights
    ):
        local_shrinks = []
        for lora_layer, x_shard in zip(lora_layers, x_shards):
            lora_a = lora_layer.lora_a_stacked[0]
            local_shrinks.append(
                x_shard[token_mask].to(dtype=lora_a.dtype) @ lora_a[lora_idx, 0].T
            )
        calls.append((local_shrinks, torch.stack(local_shrinks).sum(dim=0)))
    return calls


def apply_routed_delta(
    expected: torch.Tensor,
    x: torch.Tensor,
    lora_as: tuple[torch.Tensor, ...],
    lora_bs: tuple[torch.Tensor, ...],
    route_weights: tuple[tuple[float, ...], ...],
    delta_slice: slice | None = None,
) -> torch.Tensor:
    for token_idx in range(x.shape[0]):
        for route_idx, (lora_a, lora_b) in enumerate(zip(lora_as, lora_bs)):
            delta = x[token_idx] @ lora_a.T @ lora_b.T
            if delta_slice is not None:
                delta = delta[delta_slice]
            expected[token_idx] += route_weights[token_idx][route_idx] * delta
    return expected


def test_lora_routing_request_validates_adapters_and_weights():
    lora_1 = make_lora_request(1)
    lora_2 = make_lora_request(2)

    request = make_routing_request((lora_1, lora_2), (0.7, 0.3))

    assert request.top_k == 2
    assert request.lora_name == "routed"
    assert request.adapter_id == 100

    with pytest.raises(ValueError, match="cannot be empty"):
        make_routing_request((), ())

    with pytest.raises(ValueError, match="same length"):
        make_routing_request((lora_1, lora_2), (1.0,))

    with pytest.raises(ValueError, match="duplicate"):
        make_routing_request((lora_1, lora_1), (0.5, 0.5))

    with pytest.raises(ValueError, match="finite"):
        make_routing_request((lora_1,), (float("nan"),))

    with pytest.raises(ValueError, match="non-negative"):
        make_routing_request((lora_1,), (-1.0,))

    with pytest.raises(ValueError, match="positive"):
        make_routing_request((lora_1,), (0.0,))


def test_lora_request_helpers_return_concrete_adapters():
    lora_1 = make_lora_request(1)
    lora_2 = make_lora_request(2)
    routed_request = make_routing_request((lora_1, lora_2), (0.25, 0.75))

    assert iter_lora_requests(None) == ()
    assert iter_lora_requests(lora_1) == (lora_1,)
    assert iter_lora_requests(routed_request) == (lora_1, lora_2)
    assert iter_lora_int_ids(None) == ()
    assert iter_lora_int_ids(lora_1) == (1,)
    assert iter_lora_int_ids(routed_request) == (1, 2)


def test_lora_routing_utils_track_indexed_requests():
    lora_1 = make_lora_request(1)
    lora_2 = make_lora_request(2)
    lora_3 = make_lora_request(3)
    routed_request = make_routing_request((lora_1, lora_2), (0.25, 0.75))
    request_lora_mapping = np.zeros(3, dtype=np.int64)
    request_lora_requests: dict[int, LoRARequestLike] = {}
    lora_id_to_request_ids: dict[int, set[str]] = {}
    lora_id_to_lora_request: dict[int, LoRARequest] = {}

    add_lora_request(
        request_lora_mapping,
        request_lora_requests,
        lora_id_to_request_ids,
        lora_id_to_lora_request,
        req_index=0,
        req_id="req_0",
        lora_request=routed_request,
    )
    add_lora_request(
        request_lora_mapping,
        request_lora_requests,
        lora_id_to_request_ids,
        lora_id_to_lora_request,
        req_index=1,
        req_id="req_1",
        lora_request=lora_3,
    )

    assert tuple(request_lora_mapping) == (0, 3, 0)
    assert request_lora_requests == {0: routed_request, 1: lora_3}
    assert lora_id_to_request_ids == {
        1: {"req_0"},
        2: {"req_0"},
        3: {"req_1"},
    }
    assert lora_id_to_lora_request == {1: lora_1, 2: lora_2, 3: lora_3}

    remove_lora_request(
        request_lora_mapping,
        request_lora_requests,
        lora_id_to_request_ids,
        lora_id_to_lora_request,
        req_index=0,
        req_id="req_0",
    )

    assert tuple(request_lora_mapping) == (0, 3, 0)
    assert request_lora_requests == {1: lora_3}
    assert lora_id_to_request_ids == {3: {"req_1"}}
    assert lora_id_to_lora_request == {3: lora_3}


def test_convert_route_mapping_rejects_unloaded_lora_id():
    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12),),
        token_lora_weights=((0.7, 0.3),),
        prompt_mapping=(0,),
    )

    with pytest.raises(ValueError, match=r"unloaded LoRA ids: \[12\]"):
        convert_route_mapping(
            route_mapping,
            lora_index_to_id=[11, None],
            device=torch.device("cpu"),
        )


def test_convert_route_mapping_groups_tokens_by_route_plan():
    """Tokens sharing every route's slot form one plan; idle tokens are dropped."""
    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (13, 0), (11, 12), (0, 0), (13, 0)),
        token_lora_weights=(
            (0.6, 0.4),
            (1.0, 0.0),
            (0.5, 0.5),
            (0.0, 0.0),
            (1.0, 0.0),
        ),
        prompt_mapping=(0,),
    )

    tensors = convert_route_mapping(
        route_mapping,
        lora_index_to_id=[11, 12, 13, None],
        device=torch.device("cpu"),
    )

    assert tensors.token_lora_indices.tolist() == [
        [0, 1],
        [2, -1],
        [0, 1],
        [-1, -1],
        [2, -1],
    ]
    assert tensors.token_lora_weights.dtype == torch.float32
    assert tensors.plan_lora_indices.tolist() == [[0, 1], [2, -1]]
    assert tensors.plan_token_indices.tolist() == [0, 2, 1, 4]
    assert tensors.plan_token_start_loc.tolist() == [0, 2, 4]


def test_route_buffers_preserve_addresses_and_clear_removed_routes():
    """A smaller batch must not replay stale routes, plans, or coefficients."""
    buffers = RoutedLoRATensors.make(8, 4, 3, "cpu")
    pointers = tuple(tensor.data_ptr() for tensor in buffers)
    for ids, weights in (
        (((11, 12), (12, 11), (11, 12)), ((0.7, 0.3),) * 3),
        (((12,),), ((1.0,),)),
        (((0,),), ((0.0,),)),
    ):
        source = convert_route_mapping(
            LoRARouteMapping(ids, weights, (0,)), [11, 12], "cpu"
        )
        buffers.copy_(source)
        tokens, routes = source.token_lora_indices.shape
        plans = source.plan_lora_indices.size(0)
        active = source.plan_token_indices.numel()
        assert tuple(tensor.data_ptr() for tensor in buffers) == pointers
        torch.testing.assert_close(
            buffers.token_lora_indices[:tokens, :routes], source.token_lora_indices
        )
        torch.testing.assert_close(
            buffers.token_lora_weights[:tokens, :routes], source.token_lora_weights
        )
        assert (buffers.token_lora_indices[tokens:] == -1).all()
        assert (buffers.token_lora_indices[:, routes:] == -1).all()
        assert (buffers.token_lora_weights[tokens:] == 0).all()
        assert (buffers.token_lora_weights[:, routes:] == 0).all()
        torch.testing.assert_close(
            buffers.plan_lora_indices[:plans, :routes], source.plan_lora_indices
        )
        assert (buffers.plan_lora_indices[plans:] == -1).all()
        torch.testing.assert_close(
            buffers.plan_token_indices[:active], source.plan_token_indices
        )
        torch.testing.assert_close(
            buffers.plan_token_start_loc[: plans + 1], source.plan_token_start_loc
        )
        assert (buffers.plan_token_start_loc[plans + 1 :] == active).all()


def test_scalar_lora_state_does_not_build_route_mapping():
    lora_1 = make_lora_request(1)
    lora_2 = make_lora_request(2)
    state = LoraState(max_num_reqs=3)
    state.add_request("req_0", 0, lora_1)
    state.add_request("req_1", 1, None)
    state.add_request("req_2", 2, lora_2)

    prompt_mapping, token_mapping, active_loras, route_mapping = state.make_lora_inputs(
        req_ids=["req_0", "req_1", "req_2"],
        idx_mapping=np.array([0, 1, 2]),
        num_scheduled_tokens=np.array([2, 0, 1], dtype=np.int32),
        num_logits=np.array([1, 1, 1], dtype=np.int32),
    )

    assert prompt_mapping == (1, 0, 2)
    assert token_mapping == (1, 1, 2)
    assert {request.lora_int_id for request in active_loras} == {1, 2}
    assert route_mapping is None


@pytest.mark.parametrize("num_logits", [(1, 1, 1), (2, 1, 1)])
def test_lora_state_builds_flattened_route_mapping_for_mixed_batch(num_logits):
    lora_1 = make_lora_request(1)
    lora_2 = make_lora_request(2)
    lora_3 = make_lora_request(3)
    routed_request = make_routing_request((lora_1, lora_2), (0.6, 0.4))
    state = LoraState(max_num_reqs=3)
    state.add_request("req_0", 0, routed_request)
    state.add_request("req_1", 1, lora_3)
    state.add_request("req_2", 2, None)

    prompt_mapping, token_mapping, active_loras, route_mapping = state.make_lora_inputs(
        req_ids=["req_0", "req_1", "req_2"],
        idx_mapping=np.array([0, 1, 2]),
        num_scheduled_tokens=np.array([2, 1, 1], dtype=np.int32),
        num_logits=np.array(num_logits, dtype=np.int32),
    )

    assert prompt_mapping == (0,) * sum(num_logits)
    assert token_mapping == (0, 0, 0, 0)
    assert {request.lora_int_id for request in active_loras} == {1, 2, 3}
    assert route_mapping is not None
    assert route_mapping.prompt_mapping == prompt_mapping
    assert route_mapping.token_lora_ids == ((1, 2), (1, 2), (3, 0), (0, 0))
    assert route_mapping.token_lora_weights == (
        (0.6, 0.4),
        (0.6, 0.4),
        (1.0, 0.0),
        (0.0, 0.0),
    )


@torch.inference_mode()
@pytest.mark.parametrize("fully_sharded_loras", [False, True])
def test_routed_lora_reference_linear_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
    fully_sharded_loras: bool,
):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    device = torch.device("cpu")
    lora_config = LoRAConfig(
        max_loras=4,
        max_lora_rank=8,
        fully_sharded_loras=fully_sharded_loras,
        lora_dtype=torch.float32,
    )
    linear = ReplicatedLinear(
        4,
        3,
        bias=False,
        params_dtype=torch.float32,
        prefix="routed_lora_test",
        disable_tp=True,
    )
    linear.weight.data = torch.tensor(
        [
            [0.2, 0.1, -0.3, 0.4],
            [0.5, -0.2, 0.6, 0.1],
            [-0.4, 0.3, 0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    linear.quant_method.process_weights_after_loading(linear)
    lora_linear = ReplicatedLinearWithLoRA(linear)
    lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
    punica_wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=8,
        max_batches=4,
        device=device,
    )
    lora_linear.set_mapping(punica_wrapper)

    lora_1_a = torch.tensor(
        [[0.1, 0.0, 0.2, -0.1], [0.0, 0.3, -0.2, 0.4]],
        dtype=torch.float32,
    )
    lora_1_b = torch.tensor(
        [[0.2, -0.1], [0.4, 0.3], [-0.3, 0.1]],
        dtype=torch.float32,
    )
    lora_2_a = torch.tensor(
        [[-0.2, 0.5, 0.1, 0.0], [0.3, -0.1, 0.0, 0.2]],
        dtype=torch.float32,
    )
    lora_2_b = torch.tensor(
        [[0.1, 0.2], [-0.2, 0.5], [0.3, -0.4]],
        dtype=torch.float32,
    )
    lora_linear.set_lora(0, lora_1_a, lora_1_b)
    lora_linear.set_lora(1, lora_2_a, lora_2_b)

    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (11, 12), (12, 0)),
        token_lora_weights=((0.7, 0.3), (0.25, 0.75), (1.0, 0.0)),
        prompt_mapping=(0,),
        is_prefill=True,
    )
    punica_wrapper.update_metadata(
        route_mapping,
        lora_index_to_id=[11, 12, None, None],
        max_loras=lora_config.max_loras,
        vocab_size=128,
    )

    x = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0],
            [-1.0, 1.5, 0.25, -0.75],
            [0.2, -0.4, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )

    result = lora_linear(x)[0]
    base = linear(x)[0]
    lora_1_delta = x @ lora_1_a.T @ lora_1_b.T
    lora_2_delta = x @ lora_2_a.T @ lora_2_b.T
    expected = base.clone()
    expected[0] += 0.7 * lora_1_delta[0] + 0.3 * lora_2_delta[0]
    expected[1] += 0.25 * lora_1_delta[1] + 0.75 * lora_2_delta[1]
    expected[2] += lora_2_delta[2]

    torch.testing.assert_close(result, expected)


@torch.inference_mode()
def test_routed_lora_reference_quantized_linear_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    device = torch.device("cpu")
    lora_config = LoRAConfig(
        max_loras=4,
        max_lora_rank=8,
        lora_dtype=torch.float32,
    )
    linear = ReplicatedLinear(
        4,
        2,
        bias=False,
        params_dtype=torch.float32,
        quant_config=FakeQuantConfig(num_bits=4),
        prefix="routed_lora_quantized_test",
        disable_tp=True,
    )
    assert isinstance(linear.quant_method, FakeQuantLinearMethod)
    linear.weight.data = torch.tensor(
        [
            [0.2, -0.3, 0.5, 0.1],
            [-0.4, 0.6, 0.2, -0.2],
        ],
        dtype=torch.float32,
    )
    linear.quant_method.process_weights_after_loading(linear)
    lora_linear = ReplicatedLinearWithLoRA(linear)
    lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
    punica_wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=8,
        max_batches=4,
        device=device,
    )
    lora_linear.set_mapping(punica_wrapper)

    lora_1_a = torch.tensor(
        [[0.1, -0.2, 0.3, 0.0], [0.4, 0.1, -0.1, 0.2]],
        dtype=torch.float32,
    )
    lora_1_b = torch.tensor(
        [[0.2, -0.4], [0.5, 0.1]],
        dtype=torch.float32,
    )
    lora_2_a = torch.tensor(
        [[-0.3, 0.2, 0.0, 0.5], [0.1, 0.4, -0.2, 0.3]],
        dtype=torch.float32,
    )
    lora_2_b = torch.tensor(
        [[-0.1, 0.3], [0.4, -0.2]],
        dtype=torch.float32,
    )
    lora_linear.set_lora(0, lora_1_a, lora_1_b)
    lora_linear.set_lora(1, lora_2_a, lora_2_b)

    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (12, 0)),
        token_lora_weights=((0.4, 0.6), (1.0, 0.0)),
        prompt_mapping=(0,),
        is_prefill=True,
    )
    punica_wrapper.update_metadata(
        route_mapping,
        lora_index_to_id=[11, 12, None, None],
        max_loras=lora_config.max_loras,
        vocab_size=128,
    )

    x = torch.tensor(
        [
            [1.25, -0.5, 0.2, 2.0],
            [-0.7, 1.1, -0.3, 0.5],
        ],
        dtype=torch.float32,
    )

    result = lora_linear(x)[0]
    quantized_base = F.linear(fake_quantize_activation(x, 4), linear.weight)
    unquantized_base = F.linear(x, linear.weight)
    assert not torch.allclose(quantized_base, unquantized_base)

    lora_1_delta = x @ lora_1_a.T @ lora_1_b.T
    lora_2_delta = x @ lora_2_a.T @ lora_2_b.T
    expected = quantized_base.clone()
    expected[0] += 0.4 * lora_1_delta[0] + 0.6 * lora_2_delta[0]
    expected[1] += lora_2_delta[1]

    torch.testing.assert_close(result, expected)


@torch.inference_mode()
def test_routed_lora_reference_fully_sharded_column_parallel_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    tp_size = 2
    lora_config = LoRAConfig(
        max_loras=4,
        max_lora_rank=8,
        fully_sharded_loras=True,
        lora_dtype=torch.float32,
    )
    full_weight = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.4],
            [-0.4, 0.5, 0.1, -0.2],
            [0.6, 0.2, -0.5, 0.1],
            [0.3, -0.3, 0.4, 0.2],
        ],
        dtype=torch.float32,
    )
    lora_1_a = make_weight(20, 8, 4)
    lora_1_b = make_weight(60, 4, 8)
    lora_2_a = make_weight(100, 8, 4)
    lora_2_b = make_weight(140, 4, 8)
    route_weights = ((0.7, 0.3), (0.25, 0.75), (0.0, 1.0))
    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (11, 12), (0, 12)),
        token_lora_weights=route_weights,
        prompt_mapping=(0,),
        is_prefill=True,
    )
    x = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0],
            [-1.0, 1.5, 0.25, -0.75],
            [0.2, -0.4, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )

    lora_layers = []
    for rank in range(tp_size):
        set_tp_rank(monkeypatch, rank, tp_size)
        linear = ColumnParallelLinear(
            4,
            4,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"routed_column_sharded_lora_test_{rank}",
            tp_rank=rank,
            tp_size=tp_size,
        )
        output_shard = full_weight.shape[0] // tp_size
        linear.weight.data = full_weight[
            rank * output_shard : (rank + 1) * output_shard
        ]
        linear.quant_method.process_weights_after_loading(linear)
        lora_linear = ColumnParallelLinearWithShardedLoRA(linear)
        lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
        lora_linear.set_lora(0, lora_1_a, lora_1_b)
        lora_linear.set_lora(1, lora_2_a, lora_2_b)
        set_routed_mapping(lora_linear, route_mapping, lora_config)
        lora_layers.append(lora_linear)

    gather_calls = build_column_gather_calls(lora_layers, x)
    state = {"rank": 0, "call_idx": 0}

    def fake_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        local_shrinks, gathered = gather_calls[state["call_idx"]]
        torch.testing.assert_close(tensor, local_shrinks[state["rank"]])
        state["call_idx"] += 1
        return gathered

    import vllm.lora.layers.column_parallel_linear as column_linear

    monkeypatch.setattr(
        column_linear, "tensor_model_parallel_all_gather", fake_all_gather
    )

    for rank, lora_layer in enumerate(lora_layers):
        state["rank"] = rank
        state["call_idx"] = 0
        result = lora_layer.apply(x)
        assert state["call_idx"] == len(gather_calls)

        output_shard = full_weight.shape[0] // tp_size
        output_slice = slice(rank * output_shard, (rank + 1) * output_shard)
        expected = x @ full_weight[output_slice].T
        expected = apply_routed_delta(
            expected,
            x,
            (lora_1_a, lora_2_a),
            (lora_1_b, lora_2_b),
            route_weights,
            delta_slice=output_slice,
        )
        torch.testing.assert_close(result, expected)


@torch.inference_mode()
def test_routed_lora_reference_fully_sharded_row_parallel_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    tp_size = 2
    lora_config = LoRAConfig(
        max_loras=4,
        max_lora_rank=8,
        fully_sharded_loras=True,
        lora_dtype=torch.float32,
    )
    full_weight = make_weight(1, 4, 4)
    lora_1_a = make_weight(20, 8, 4)
    lora_1_b = make_weight(60, 4, 8)
    lora_2_a = make_weight(100, 8, 4)
    lora_2_b = make_weight(140, 4, 8)
    route_weights = ((0.6, 0.4), (0.25, 0.75), (0.0, 1.0))
    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (11, 12), (0, 12)),
        token_lora_weights=route_weights,
        prompt_mapping=(0,),
        is_prefill=True,
    )
    x = make_weight(100, 3, 4)
    input_shard_size = x.shape[1] // tp_size
    output_shard_size = full_weight.shape[0] // tp_size
    x_shards = [
        x[:, rank * input_shard_size : (rank + 1) * input_shard_size]
        for rank in range(tp_size)
    ]

    lora_layers = []
    for rank in range(tp_size):
        set_tp_rank(monkeypatch, rank, tp_size)
        linear = RowParallelLinear(
            4,
            4,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            params_dtype=torch.float32,
            prefix=f"routed_row_sharded_lora_test_{rank}",
        )
        input_slice = slice(rank * input_shard_size, (rank + 1) * input_shard_size)
        linear.weight.data = full_weight[:, input_slice]
        linear.quant_method.process_weights_after_loading(linear)
        lora_linear = RowParallelLinearWithShardedLoRA(linear)
        lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
        lora_linear.set_lora(0, lora_1_a, lora_1_b)
        lora_linear.set_lora(1, lora_2_a, lora_2_b)
        set_routed_mapping(lora_linear, route_mapping, lora_config)
        lora_layers.append(lora_linear)

    reduce_calls = build_row_reduce_calls(lora_layers, x_shards)
    state = {"rank": 0, "call_idx": 0}

    def fake_all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        local_shrinks, reduced = reduce_calls[state["call_idx"]]
        torch.testing.assert_close(tensor, local_shrinks[state["rank"]])
        state["call_idx"] += 1
        return reduced

    import vllm.lora.layers.row_parallel_linear as row_linear

    monkeypatch.setattr(row_linear, "tensor_model_parallel_all_reduce", fake_all_reduce)

    routed_delta = apply_routed_delta(
        torch.zeros(x.shape[0], full_weight.shape[0], dtype=torch.float32),
        x,
        (lora_1_a, lora_2_a),
        (lora_1_b, lora_2_b),
        route_weights,
    )
    rank_outputs = []
    for rank, (lora_layer, x_shard) in enumerate(zip(lora_layers, x_shards)):
        state["rank"] = rank
        state["call_idx"] = 0
        result = lora_layer.apply(x_shard)
        assert state["call_idx"] == len(reduce_calls)

        input_slice = slice(rank * input_shard_size, (rank + 1) * input_shard_size)
        output_slice = slice(rank * output_shard_size, (rank + 1) * output_shard_size)
        expected = x_shard @ full_weight[:, input_slice].T
        expected[:, output_slice] += routed_delta[:, output_slice]
        torch.testing.assert_close(result, expected)
        rank_outputs.append(result)

    full_expected = x @ full_weight.T + routed_delta
    torch.testing.assert_close(torch.stack(rank_outputs).sum(dim=0), full_expected)


@torch.inference_mode()
@pytest.mark.parametrize("quantized_base", [False, True])
def test_routed_lora_reference_qkv_parallel_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
    quantized_base: bool,
):
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    device = torch.device("cpu")
    lora_config = LoRAConfig(
        max_loras=4,
        max_lora_rank=8,
        lora_dtype=torch.float32,
    )
    linear = QKVParallelLinear(
        hidden_size=4,
        head_size=2,
        total_num_heads=2,
        total_num_kv_heads=1,
        bias=False,
        params_dtype=torch.float32,
        quant_config=FakeQuantConfig(num_bits=4) if quantized_base else None,
        prefix="routed_qkv_lora_test",
        disable_tp=True,
    )
    if quantized_base:
        assert isinstance(linear.quant_method, FakeQuantLinearMethod)
    linear.weight.data = torch.tensor(
        [
            [0.2, -0.1, 0.3, 0.4],
            [-0.4, 0.5, 0.1, -0.2],
            [0.6, 0.2, -0.5, 0.1],
            [0.3, -0.3, 0.4, 0.2],
            [-0.2, 0.7, 0.1, 0.5],
            [0.4, -0.6, 0.2, -0.1],
            [0.5, 0.1, -0.4, 0.3],
            [-0.1, 0.2, 0.6, -0.5],
        ],
        dtype=torch.float32,
    )
    linear.quant_method.process_weights_after_loading(linear)
    lora_linear = MergedQKVParallelLinearWithLoRA(linear)
    lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
    punica_wrapper = PunicaWrapperCPU(
        max_num_batched_tokens=8,
        max_batches=4,
        device=device,
    )
    lora_linear.set_mapping(punica_wrapper)

    lora_1_a = [
        torch.tensor(
            [[0.1, 0.2, -0.1, 0.0], [0.3, -0.2, 0.4, 0.1]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[-0.2, 0.1, 0.3, 0.4], [0.0, 0.5, -0.1, 0.2]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.4, -0.1, 0.2, 0.3], [-0.3, 0.2, 0.1, 0.0]],
            dtype=torch.float32,
        ),
    ]
    lora_1_b = [
        torch.tensor(
            [[0.2, -0.1], [0.3, 0.4], [-0.2, 0.5], [0.1, 0.2]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.5, -0.3], [0.2, 0.1]], dtype=torch.float32),
        torch.tensor([[-0.4, 0.2], [0.3, 0.6]], dtype=torch.float32),
    ]
    lora_2_a = [
        torch.tensor(
            [[-0.1, 0.3, 0.2, 0.1], [0.5, 0.0, -0.2, 0.4]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.2, -0.4, 0.1, 0.5], [0.3, 0.1, 0.0, -0.2]],
            dtype=torch.float32,
        ),
        torch.tensor(
            [[0.0, 0.2, -0.3, 0.4], [0.1, -0.5, 0.3, 0.2]],
            dtype=torch.float32,
        ),
    ]
    lora_2_b = [
        torch.tensor(
            [[0.1, 0.3], [-0.5, 0.2], [0.4, -0.1], [0.2, 0.5]],
            dtype=torch.float32,
        ),
        torch.tensor([[0.3, 0.4], [-0.2, 0.6]], dtype=torch.float32),
        torch.tensor([[0.5, -0.1], [0.2, -0.3]], dtype=torch.float32),
    ]
    lora_linear.set_lora(0, lora_1_a, lora_1_b)
    lora_linear.set_lora(1, lora_2_a, lora_2_b)

    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (11, 12), (12, 0)),
        token_lora_weights=((0.7, 0.3), (0.25, 0.75), (1.0, 0.0)),
        prompt_mapping=(0,),
        is_prefill=True,
    )
    punica_wrapper.update_metadata(
        route_mapping,
        lora_index_to_id=[11, 12, None, None],
        max_loras=lora_config.max_loras,
        vocab_size=128,
    )

    x = torch.tensor(
        [
            [1.0, 0.5, -0.5, 2.0],
            [-1.0, 1.5, 0.25, -0.75],
            [0.2, -0.4, 0.6, 0.8],
        ],
        dtype=torch.float32,
    )

    result = lora_linear(x)[0]
    if quantized_base:
        expected = F.linear(fake_quantize_activation(x, 4), linear.weight)
        unquantized_base = F.linear(x, linear.weight)
        assert not torch.allclose(expected, unquantized_base)
    else:
        expected = linear(x)[0]
    expected = expected.clone()
    route_weights = ((0.7, 0.3), (0.25, 0.75), (0.0, 1.0))
    slot_loras = ((lora_1_a, lora_1_b), (lora_2_a, lora_2_b))
    offset = 0
    for slice_idx, output_slice in enumerate(lora_linear.output_slices):
        for token_idx in range(x.shape[0]):
            for route_idx, (lora_a, lora_b) in enumerate(slot_loras):
                delta = x[token_idx] @ lora_a[slice_idx].T @ lora_b[slice_idx].T
                expected[token_idx, offset : offset + output_slice] += (
                    route_weights[token_idx][route_idx] * delta
                )
        offset += output_slice

    torch.testing.assert_close(result, expected)


@torch.inference_mode()
@pytest.mark.parametrize(
    "routed_kernel_limit",
    [
        "supported",
        "cooperative_launch",
        "bf16_hardware",
        "capacity",
        "later_slice",
        "shared_memory",
        "dtype",
        "rank",
        "layout",
        "driver_error",
    ],
    indirect=True,
)
def test_routed_lora_kernel_qkv_parallel_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
    routed_kernel_limit,
):
    """Packed slices match an oracle on both fused and unsupported-config paths;
    preflighting a later slice must not apply an earlier update twice."""
    from vllm.lora.punica_wrapper.punica_gpu import PunicaWrapperGPU
    from vllm.platforms import current_platform

    if not current_platform.is_cuda():
        pytest.skip("Requires an NVIDIA GPU")
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.parameter.get_tensor_model_parallel_world_size",
        lambda: 1,
    )

    dtype = (
        torch.float32
        if routed_kernel_limit == "dtype"
        else torch.float16
        if routed_kernel_limit == "shared_memory"
        else torch.bfloat16
    )
    device = torch.device("cuda")
    generator = torch.Generator().manual_seed(0)
    hidden = 64
    rank = 256 if routed_kernel_limit == "rank" else 16
    lora_config = LoRAConfig(max_loras=4, max_lora_rank=rank, lora_dtype=dtype)
    linear = QKVParallelLinear(
        hidden_size=hidden,
        head_size=16,
        total_num_heads=4,
        total_num_kv_heads=2,
        bias=False,
        params_dtype=dtype,
        prefix="routed_qkv_kernel_test",
        disable_tp=True,
    )
    linear.weight.data = (
        torch.randn(linear.weight.shape, generator=generator) / hidden**0.5
    ).to(dtype)
    linear = linear.to(device)
    linear.quant_method.process_weights_after_loading(linear)
    lora_linear = MergedQKVParallelLinearWithLoRA(linear)
    lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
    punica_wrapper = PunicaWrapperGPU(
        max_num_batched_tokens=64,
        max_batches=8,
        device=device,
        lora_config=lora_config,
    )
    lora_linear.set_mapping(punica_wrapper)
    for slot in range(3):
        lora_linear.set_lora(
            slot,
            [torch.randn(rank, hidden, generator=generator) / hidden**0.5] * 3,
            [
                torch.randn(size, rank, generator=generator) / rank**0.5
                for size in lora_linear.output_slices
            ],
        )

    # Three requests: a two-adapter composition, a single adapter, no adapter.
    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12),) * 15 + ((13, 0),) * 15 + ((0, 0),) * 7,
        token_lora_weights=((0.7, 0.3),) * 15 + ((1.0, 0.0),) * 15 + ((0.0, 0.0),) * 7,
        prompt_mapping=(0, 0, 0),
        is_prefill=True,
    )
    punica_wrapper.update_metadata(
        route_mapping,
        lora_index_to_id=[11, 12, 13, None],
        max_loras=lora_config.max_loras,
        vocab_size=128,
    )
    if routed_kernel_limit == "layout":
        x = torch.randn(37, hidden * 2, generator=generator).to(device, dtype)[:, ::2]
    else:
        x = torch.randn(37, hidden, generator=generator).to(device, dtype)
    base = linear(x)[0]
    if routed_kernel_limit == "driver_error":
        with pytest.raises(RuntimeError, match="cuOccupancyMaxActiveBlocks"):
            lora_linear(x)
        return
    assert lora_linear._can_use_routed_lora_kernel(x, base) == (
        routed_kernel_limit == "supported"
    )

    result = lora_linear(x)[0]

    route_mapping_tensors = punica_wrapper.lora_route_mapping
    assert route_mapping_tensors is not None
    token_lora_indices, token_lora_weights = route_mapping_tensors
    expected = base.float()
    offset = 0
    for slice_idx, size in enumerate(lora_linear.output_slices):
        lora_a = lora_linear.lora_a_stacked[slice_idx][:, 0].float()
        lora_b = lora_linear.lora_b_stacked[slice_idx][:, 0].float()
        for route in range(2):
            for slot in range(3):
                rows = token_lora_indices[:, route] == slot
                delta = x[rows].float() @ lora_a[slot].T @ lora_b[slot].T
                expected[rows, offset : offset + size] += (
                    token_lora_weights[rows, route, None] * delta
                )
        offset += size
    torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=2e-2)
    assert torch.equal(result[30:], base[30:])


@torch.inference_mode()
def test_routed_lora_reference_fully_sharded_qkv_parallel_matches_oracle(
    monkeypatch: pytest.MonkeyPatch,
):
    tp_size = 2
    local_qkv_size = 2
    lora_config = LoRAConfig(
        max_loras=4,
        max_lora_rank=8,
        fully_sharded_loras=True,
        lora_dtype=torch.float32,
    )
    full_weight = make_weight(1, 12, 4)
    lora_1_a = [make_weight(60 + idx * 40, 8, 4) for idx in range(3)]
    lora_1_b = [make_weight(180 + idx * 40, 4, 8) for idx in range(3)]
    lora_2_a = [make_weight(300 + idx * 40, 8, 4) for idx in range(3)]
    lora_2_b = [make_weight(420 + idx * 40, 4, 8) for idx in range(3)]
    route_weights = ((0.7, 0.3), (0.25, 0.75), (0.0, 1.0))
    route_mapping = LoRARouteMapping(
        token_lora_ids=((11, 12), (11, 12), (0, 12)),
        token_lora_weights=route_weights,
        prompt_mapping=(0,),
        is_prefill=True,
    )
    x = make_weight(300, 3, 4)

    lora_layers = []
    for rank in range(tp_size):
        set_tp_rank(monkeypatch, rank, tp_size)
        linear = QKVParallelLinear(
            hidden_size=4,
            head_size=2,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            params_dtype=torch.float32,
            prefix=f"routed_qkv_sharded_lora_test_{rank}",
            disable_tp=False,
        )
        local_q = full_weight[rank * local_qkv_size : (rank + 1) * local_qkv_size]
        local_k = full_weight[
            4 + rank * local_qkv_size : 4 + (rank + 1) * local_qkv_size
        ]
        local_v = full_weight[
            8 + rank * local_qkv_size : 8 + (rank + 1) * local_qkv_size
        ]
        linear.weight.data = torch.cat([local_q, local_k, local_v], dim=0)
        linear.quant_method.process_weights_after_loading(linear)
        lora_linear = MergedQKVParallelLinearWithShardedLoRA(linear)
        lora_linear.create_lora_weights(lora_config.max_loras, lora_config)
        lora_linear.set_lora(0, lora_1_a, lora_1_b)
        lora_linear.set_lora(1, lora_2_a, lora_2_b)
        set_routed_mapping(lora_linear, route_mapping, lora_config)
        lora_layers.append(lora_linear)

    gather_calls = build_column_gather_calls(lora_layers, x)
    state = {"rank": 0, "call_idx": 0}

    def fake_all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        del dim
        local_shrinks, gathered = gather_calls[state["call_idx"]]
        torch.testing.assert_close(tensor, local_shrinks[state["rank"]])
        state["call_idx"] += 1
        return gathered

    import vllm.lora.layers.column_parallel_linear as column_linear

    monkeypatch.setattr(
        column_linear, "tensor_model_parallel_all_gather", fake_all_gather
    )

    slot_loras = ((lora_1_a, lora_1_b), (lora_2_a, lora_2_b))
    for rank, lora_layer in enumerate(lora_layers):
        state["rank"] = rank
        state["call_idx"] = 0
        result = lora_layer.apply(x)
        assert state["call_idx"] == len(gather_calls)

        local_rows = []
        for full_offset in (0, 4, 8):
            local_rows.append(
                full_weight[
                    full_offset + rank * local_qkv_size : full_offset
                    + (rank + 1) * local_qkv_size
                ]
            )
        expected = x @ torch.cat(local_rows, dim=0).T
        rank_delta_slice = slice(rank * local_qkv_size, (rank + 1) * local_qkv_size)
        local_offset = 0
        for slice_idx, output_slice in enumerate(lora_layer.output_slices):
            for token_idx in range(x.shape[0]):
                for route_idx, (lora_a, lora_b) in enumerate(slot_loras):
                    delta = x[token_idx] @ lora_a[slice_idx].T @ lora_b[slice_idx].T
                    expected[token_idx, local_offset : local_offset + output_slice] += (
                        route_weights[token_idx][route_idx] * delta[rank_delta_slice]
                    )
            local_offset += output_slice

        torch.testing.assert_close(result, expected)
