# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
from vllm.lora.punica_wrapper.utils import convert_route_mapping
from vllm.lora.request import (
    LoRARequest,
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
        return "custom_quant"

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
    request_lora_requests = {}
    lora_id_to_request_ids = {}
    lora_id_to_lora_request = {}

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
    )

    assert prompt_mapping == (1, 0, 2)
    assert token_mapping == (1, 1, 2)
    assert {request.lora_int_id for request in active_loras} == {1, 2}
    assert route_mapping is None


def test_lora_state_builds_flattened_route_mapping_for_mixed_batch():
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
    )

    assert prompt_mapping == (0, 0, 0)
    assert token_mapping == (0, 0, 0, 0)
    assert {request.lora_int_id for request in active_loras} == {1, 2, 3}
    assert route_mapping is not None
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
