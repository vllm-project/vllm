# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator
from dataclasses import dataclass
from functools import reduce

import numpy as np
import torch
from torch.export import export

from vllm.logger import init_logger
from vllm.v1.outputs import MFUInfo

logger = init_logger(__name__)


@dataclass
class TensorMeta:
    shape: list
    dtype: torch.dtype


# optionally grab the module assiociated with a node
# useful for finding additional metadata / weights
def resolve_module(module, path: str):
    try:
        for part in path.split("."):
            module = module[int(part)] if part.isdigit() else getattr(module, part)
        return module
    except Exception:
        pass
    return None


def get_shape_dtype(node, model):
    default_shape: list[int] = []
    default_dtype = torch.int32
    default = TensorMeta(default_shape, default_dtype)
    # in some cases, gather relevant model weights
    if isinstance(node, str) and node.startswith("model."):
        weights = {}
        for k in model.state_dict():
            if node not in k:
                continue
            t = model.state_dict()[k]
            if isinstance(t, torch.Tensor):
                weights[k] = TensorMeta(t.shape, t.dtype)
        if len(weights) > 0:
            return weights, resolve_module(model, node)

    def from_tensor(t):
        return TensorMeta(t.shape, t.dtype)

    if not hasattr(node, "meta"):
        return default
    if node.meta is None:
        return default
    tensor_meta = node.meta["val"]
    if tensor_meta is None:
        return default
    if isinstance(tensor_meta, list):
        tensor_meta = tensor_meta[0]
    shape = tensor_meta.shape or default_shape
    dtype = tensor_meta.dtype or default_dtype
    return TensorMeta(shape, dtype)


def get_dtype_size(dtype):
    dtype_bytes = 4
    if dtype == torch.float16:
        dtype_bytes = 2
    elif dtype == torch.float64:
        dtype_bytes = 8
    elif dtype in [torch.int8, torch.uint8]:
        dtype_bytes = 1
    return dtype_bytes


def get_tensor_bytes(tensor_meta):
    return get_dtype_size(tensor_meta.dtype) * reduce(
        operator.mul, tensor_meta.shape, 1
    )


def mfu_linear(inputs, output):
    M, K = inputs[0].shape
    _, N = inputs[1].shape
    flops = M * N * K * 2
    read = get_tensor_bytes(inputs[0]) + get_tensor_bytes(inputs[1])
    write = get_tensor_bytes(output)
    return flops, read, write


def mfu_pointwise(inputs, output):
    flops = reduce(operator.mul, output.shape, 1)  # single operation per output
    read = reduce(operator.add, [get_tensor_bytes(i) for i in inputs], 0)
    write = get_tensor_bytes(output)
    return flops, read, write


def mfu_mean(inputs, output):
    input_elements = reduce(operator.mul, inputs[0].shape, 1)
    output_elements = reduce(operator.mul, output.shape, 1)

    elements_per_reduction = (
        input_elements // output_elements if output_elements > 0 else input_elements
    )

    flops = output_elements * elements_per_reduction
    read = get_tensor_bytes(inputs[0])
    write = get_tensor_bytes(output)
    return flops, read, write


def scaled_flop_mfu(func, scale):
    def new_func(inputs, output):
        f, r, w = func(inputs, output)
        return f * scale, r, w

    return new_func


# remove non-tensor meta inputs (additional inputs are dicts)
def no_additional(func):
    def new_func(inputs, output):
        pruned_inputs = [i for i in inputs if isinstance(i, TensorMeta)]
        return func(pruned_inputs, output)

    return new_func


def mfu_vllm_all_reduce(inputs, output):
    # non-reduced outer dim
    flops = reduce(operator.mul, inputs[0].shape[1:], 1)
    read = get_tensor_bytes(inputs[0])
    write = get_tensor_bytes(output)
    return flops, read, write


def mfu_layer_norm(inputs, output):
    N = reduce(operator.mul, inputs[0].shape, 1)
    flops = N * 8
    read = reduce(operator.add, [get_tensor_bytes(i) for i in inputs], 0)
    write = get_tensor_bytes(output)
    return flops, read, write


def mfu_vllm_unified_attention_with_output(inputs, output):
    query = inputs[0]  # Shape: [B, S, D] or [B, H, S, D]

    if len(query.shape) == 4:  # Multi-head format [B, H, S, D]
        B, H, S, D = query.shape
        total_dim = H * D
    else:  # Standard format [B, S, D]
        B, S, total_dim = query.shape
        H = 1
        D = total_dim

    # Attention FLOPs:
    # 1. Q @ K^T: B*H*S*S*D operations
    # 2. Softmax: B*H*S*S*3 operations (exp, sum, divide)
    # 3. Attn @ V: B*H*S*S*D operations
    # 4. Output projection: B*S*total_dim*total_dim operations
    attention_flops = B * H * S * S * (2 * D + 3)
    output_proj_flops = B * S * total_dim * total_dim * 2

    flops = attention_flops + output_proj_flops

    read = reduce(operator.add, [get_tensor_bytes(i) for i in inputs], 0)
    write = get_tensor_bytes(output)
    return flops, read, write


def mfu_embedding(inputs, output):
    indices = inputs[0]
    embedding_table = inputs[1]

    num_lookups = reduce(operator.mul, indices.shape[:-1], 1)  # skip inner dim
    flops = num_lookups

    embedding_dim = embedding_table.shape[-1]
    read = get_tensor_bytes(indices) + (
        num_lookups * embedding_dim * get_dtype_size(embedding_table.dtype)
    )
    write = get_tensor_bytes(output)
    return flops, read, write


def mfu_nop(inputs, output):
    return 0, 0, 0


def mfu_vllm_moe_forward(inputs, output):
    bs, hidden = inputs[0].shape
    weights, module = inputs[-1]

    for k in weights:
        if "w13_weight" in k:
            expert_w13 = weights[k]
        if "w2_weight" in k:
            expert_w2 = weights[k]
    top_k = module.top_k
    experts, combined_intermediate, expert_hidden = expert_w13.shape
    quant_factor = hidden / expert_hidden  # 2 if mxe4, 1 if fp8
    intermediate = combined_intermediate // 2
    experts_, output_hidden, expert_intermediate = expert_w2.shape
    assert intermediate == expert_intermediate * quant_factor
    assert output_hidden == hidden
    assert experts == experts_

    # upscale
    flops = bs * top_k * hidden * intermediate * 2 * 2  # fma, swiglu
    # downscale
    flops += bs * top_k * intermediate * output_hidden * 2  # fma

    # input
    read_bytes = get_tensor_bytes(inputs[0])
    # weights (quantized to mxe4 or not, same bytes)
    read_bytes += get_tensor_bytes(expert_w13)
    read_bytes += get_tensor_bytes(expert_w2)

    write_bytes = get_tensor_bytes(output)

    return flops, read_bytes, write_bytes


mapping = {
    "aten.linear.default": mfu_linear,
    "vllm.all_reduce.default": mfu_vllm_all_reduce,
    "vllm.moe_forward.default": mfu_vllm_moe_forward,
    "aten.layer_norm.default": mfu_layer_norm,
    "vllm.unified_attention_with_output.default": (
        mfu_vllm_unified_attention_with_output
    ),
    "aten.embedding.default": mfu_embedding,
    "aten.mean.dim": mfu_mean,
    "aten.add.Tensor": mfu_pointwise,
    "aten.sub.Tensor": mfu_pointwise,
    "aten.mul.Tensor": mfu_pointwise,
    "aten.ge.Scalar": mfu_pointwise,
    "aten.lt.Scalar": mfu_pointwise,
    "aten.__and__.Tensor": mfu_pointwise,
    "aten.__or__.Tensor": mfu_pointwise,
    "aten.bitwise_not.default": mfu_pointwise,
    "aten.to.dtype": mfu_pointwise,
    "aten.div.Tensor": mfu_pointwise,
    "aten.pow.Tensor_Scalar": scaled_flop_mfu(mfu_pointwise, 8),  # approx
    "aten.tanh.default": scaled_flop_mfu(mfu_pointwise, 5),
    "aten.rsqrt.default": scaled_flop_mfu(mfu_pointwise, 10),  # approx
    "aten.masked_fill_.Scalar": scaled_flop_mfu(mfu_pointwise, 2),
    "aten.cat.default": mfu_nop,
    "<built-in function getitem>": mfu_nop,
    "aten.zeros.default": mfu_nop,
    "aten.chunk.default": mfu_nop,
    "aten.pad.default": mfu_nop,
    "aten.split_with_sizes.default": mfu_nop,
    "aten.flatten.default": mfu_nop,
    "aten.flatten.using_ints": mfu_nop,
    "aten.index_select.default": mfu_nop,
    "aten.contiguous.default": mfu_nop,
    "aten.reshape.default": mfu_nop,
    "aten.slice.default": mfu_nop,
    "aten.slice.Tensor": mfu_nop,
    "aten.alias.default": mfu_nop,
    "aten.view.default": mfu_nop,
    "aten.unsqueeze.default": mfu_nop,
    "aten._assert_tensor_metadata.default": mfu_nop,
}


def get_additional_inputs(nodes, model):
    additional = {}
    for node in nodes:
        sd_keys = model.state_dict().keys()
        if not isinstance(node, str):
            continue
        for k in sd_keys:
            if node in k:
                t = model.state_dict()[k]
                additional[k] = TensorMeta(t.shape, t.dtype)
    if len(additional) == 0:
        return None
    return additional


def analyze_node(node, model):
    if node.op == "call_function":
        output = get_shape_dtype(node, model)
        inputs = [get_shape_dtype(arg, model) for arg in node.args]
        op_name = str(node.target)
        if op_name in mapping:
            return mapping[op_name](inputs, output)
        # assume pointwise
        f, r, w = mfu_pointwise(inputs, output)
        log = f"""Assuming {op_name} is pointwise, \
with {f} flops, {r} read bytes, {w} written bytes"""
        logger.warning(log)
        return f, r, w
    return 0, 0, 0


def analyze_model_mfu_fast(model, parameter_count, args, kwargs):
    if "input_ids" not in kwargs:
        return MFUInfo()
    assert parameter_count > 0, "Please specify mfu_analysis_active_parameters"
    mfu_flops = kwargs["input_ids"].numel() * parameter_count * 2
    return MFUInfo(mfu_flops)


def analyze_model_mfu(model, args, kwargs):
    ep = export(model, args=args, kwargs=kwargs)

    mfu_info = MFUInfo()
    for node in ep.graph.nodes:
        f, r, w = analyze_node(node, model)
        mfu_info.flops += f
        mfu_info.read_bytes += r
        mfu_info.write_bytes += w
    return mfu_info


class MFUAnalysisLogging:
    """Aggregate and log MFU metrics.

    LoggingStatLogger aggregates per-iteration metrics over a set
    time interval using observe() and then logs them using log()
    before resetting to zero.
    """

    def __init__(self):
        self.reset()
        self.flops: list[float] = []
        self.latency_s: list[float] = []
        self.read_bytes: list[float] = []
        self.write_bytes: list[float] = []

        self.achieved_flops: list[float] = []
        self.achieved_read: list[float] = []
        self.achieved_write: list[float] = []

    def reset(self):
        self.flops = []
        self.latency_s = []
        self.read_bytes = []
        self.write_bytes = []

        self.achieved_flops = []
        self.achieved_read = []
        self.achieved_write = []

    def observe(self, mfu_info: MFUInfo):
        self.flops.append(mfu_info.flops)
        self.read_bytes.append(mfu_info.read_bytes)
        self.write_bytes.append(mfu_info.write_bytes)
        self.latency_s.append(mfu_info.latency_s)

    def format_units(self, value, base_unit):
        if base_unit == "FLOPS" or base_unit == "B/s":
            if value >= 1e12:
                return f"{value / 1e12:.2f} T{base_unit}"
            else:
                return f"{value / 1e9:.2f} G{base_unit}"

    def log(self, log_fn=logger.info):
        flops = np.array(self.flops)
        latency_s = np.array(self.latency_s)
        read_bytes = np.array(self.read_bytes)
        write_bytes = np.array(self.write_bytes)

        flops_rate = flops / latency_s  # FLOPS per second
        read_rate = read_bytes / latency_s  # bytes read per second
        write_rate = write_bytes / latency_s  # bytes written per second

        flops_stats = {
            "peak": np.max(flops_rate),
            "mean": np.mean(flops_rate),
            "min": np.min(flops_rate),
        }

        read_stats = {
            "peak": np.max(read_rate),
            "mean": np.mean(read_rate),
            "min": np.min(read_rate),
        }

        write_stats = {
            "peak": np.max(write_rate),
            "mean": np.mean(write_rate),
            "min": np.min(write_rate),
        }
        avg_total_flops = np.mean(flops)

        message = (
            "Avg total flop count: "
            f"{self.format_units(avg_total_flops, 'FLOPS')} | "
            "FLOPS/s - Peak: "
            f"{self.format_units(flops_stats['peak'], 'FLOPS')}, "
            f"Mean: {self.format_units(flops_stats['mean'], 'FLOPS')}, "
            f"Min: {self.format_units(flops_stats['min'], 'FLOPS')} | "
            f"Read - Peak: {self.format_units(read_stats['peak'], 'B/s')}, "
            f"Mean: {self.format_units(read_stats['mean'], 'B/s')}, "
            f"Min: {self.format_units(read_stats['min'], 'B/s')} | "
            f"Write - Peak: {self.format_units(write_stats['peak'], 'B/s')}, "
            f"Mean: {self.format_units(write_stats['mean'], 'B/s')}, "
            f"Min: {self.format_units(write_stats['min'], 'B/s')}"
        )

        log_fn(message)
