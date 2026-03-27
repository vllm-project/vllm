# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch._inductor.pattern_matcher as pm
import torch.fx as fx
from torch._inductor.pattern_matcher import Match, PatternMatcherPass
from torch.fx.experimental.symbolic_shapes import statically_known_true

from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import get_tensor_model_parallel_world_size
from vllm.platforms import current_platform

from ..fx_utils import is_func

FP8_DTYPE = current_platform.fp8_dtype()
FLASHINFER_BMM_FP8_MIN_M = 64
FLASHINFER_VIEW_OP = torch.ops.aten.reshape.default


def _get_node_arg(node: fx.Node, name: str, index: int) -> object:
    return node.kwargs.get(name, node.args[index] if len(node.args) > index else None)


def _node_shape(node: fx.Node) -> list[object] | None:
    val = node.meta.get("val")
    if hasattr(val, "shape"):
        return list(val.shape)
    return None


def _node_first_dim(node: fx.Node) -> object | None:
    shape = _node_shape(node)
    if shape:
        return shape[0]
    return None


def _dim_is_statically_lt(dim: int | torch.SymInt, threshold: int) -> bool:
    if isinstance(dim, int):
        return dim < threshold
    try:
        return bool(statically_known_true(dim < threshold))
    except Exception:
        return False


def _passes_min_m(node: fx.Node) -> bool:
    gemm_m = _node_first_dim(node)
    if gemm_m is None or not isinstance(gemm_m, int | torch.SymInt):
        return True
    return not _dim_is_statically_lt(gemm_m, FLASHINFER_BMM_FP8_MIN_M)


def _single_user(node: fx.Node) -> fx.Node | None:
    users = list(node.users)
    if len(users) != 1:
        return None
    return users[0]


def _has_exact_qkv_split_user(node: fx.Node) -> bool:
    split_node = _single_user(node)
    if split_node is None or not is_func(
        split_node, torch.ops.aten.split_with_sizes.default
    ):
        return False

    split_input = _get_node_arg(split_node, "self", 0)
    split_sizes = _get_node_arg(split_node, "split_sizes", 1)
    dim = _get_node_arg(split_node, "dim", 2)

    if split_input is not node or dim not in (-1, 1):
        return False
    if not isinstance(split_sizes, list | tuple) or len(split_sizes) != 3:
        return False
    if any(not isinstance(size, int | torch.SymInt) for size in split_sizes):
        return False

    shape = _node_shape(node)
    if shape is not None and len(shape) == 2:
        last_dim = shape[1]
        if (
            isinstance(last_dim, int)
            and all(isinstance(size, int) for size in split_sizes)
            and sum(split_sizes) != last_dim
        ):
            return False

    return True


class _BaseFlashInferPattern:
    def __init__(self, dtype: torch.dtype, device: str | None) -> None:
        self.dtype = dtype
        self.device = device
        self.tp = get_tp_group()
        self.tp_size = get_tensor_model_parallel_world_size()

    def get_inputs(self) -> list[torch.Tensor]:
        input = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        weight = torch.empty([16, 16], device=self.device, dtype=FP8_DTYPE)
        scale_a = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        scale_b = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        return [input, weight, scale_a, scale_b]


class FlashInferBMMFP8ReduceScatterPattern(_BaseFlashInferPattern):
    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            bmm_result = torch.ops.vllm.bmm_fp8.default(
                input.unsqueeze(0),
                weight.unsqueeze(0),
                scale_a,
                scale_b,
                self.dtype,
                "auto",
            )
            mm_result = FLASHINFER_VIEW_OP(
                bmm_result,
                [input.shape[0], weight.shape[1]],
            )
            return torch.ops.vllm.reduce_scatter.default(
                mm_result,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            output_shape = [*input.shape[:-1], weight.shape[1]]
            return torch.ops.vllm.fused_flashinfer_scaled_matmul_reduce_scatter.default(
                input,
                weight,
                scale_a,
                scale_b,
                "sum",
                0,
                0,
                self.tp.device_group.group_name,
                output_shape,
                self.dtype,
            )

        def extra_check(match: Match) -> bool:
            rs_node = match.output_node()
            rs_input = _get_node_arg(rs_node, "tensor", 0)
            return isinstance(rs_input, fx.Node) and _passes_min_m(rs_input)

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=extra_check,
        )


class AllGatherFlashInferBMMFP8QKVPattern(_BaseFlashInferPattern):
    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            all_gather = torch.ops.vllm.all_gather.default(
                x,
                dim=0,
                world_size=self.tp_size,
                group_name=self.tp.unique_name,
            )
            bmm_result = torch.ops.vllm.bmm_fp8.default(
                all_gather.unsqueeze(0),
                weight.unsqueeze(0),
                scale_a,
                scale_b,
                self.dtype,
                "auto",
            )
            return FLASHINFER_VIEW_OP(
                bmm_result,
                [all_gather.shape[0], weight.shape[1]],
            )

        def replacement(
            x: torch.Tensor,
            weight: torch.Tensor,
            scale_a: torch.Tensor,
            scale_b: torch.Tensor,
        ) -> torch.Tensor:
            _, mm_output = (
                torch.ops.vllm.fused_all_gather_flashinfer_scaled_matmul.default(
                    x,
                    weight,
                    scale_a,
                    scale_b,
                    0,
                    self.tp.device_group.group_name,
                    self.dtype,
                )
            )
            return mm_output

        def extra_check(match: Match) -> bool:
            qkv_output = match.output_node()
            return _passes_min_m(qkv_output) and _has_exact_qkv_split_user(qkv_output)

        pm.register_replacement(
            pattern,
            replacement,
            self.get_inputs(),
            pm.fwd_only,
            pm_pass,
            extra_check=extra_check,
        )
