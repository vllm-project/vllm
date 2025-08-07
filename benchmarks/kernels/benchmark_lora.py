# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import copy
import json
import pickle
import time
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import ArgPool, Bench, CudaGraphBenchParams
from weight_shapes import WEIGHT_SHAPES

from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import LoRAKernelMeta, lora_expand, lora_shrink
    from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT

from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_TP_SIZES = [1]
DEFAULT_BATCH_SIZES = [
    1,
    16,
    32,
    64,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    2048,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
]
DEFAULT_HIDDEN_SIZES = [1024, 2048, 4096, 8192, 16384]
DEFAULT_LORA_RANKS = [16]
DEFAULT_NUM_LORAS = [1, 2, 3, 4]
DEFAULT_SORT_BY_LORA_IDS = [False, True]
DEFAULT_SEQ_LENGTHS = [1]
DEFAULT_EXPAND_FN_ADD_INPUTS = [True, False]


# Utilities
def dtype_to_str(dtype: torch.dtype):
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    raise ValueError(f"Unsupported dtype {dtype}")


def make_rand_lora_weight_tensor(
    k: int, n: int, num_loras: int, dtype: torch.dtype, device: str = "cuda"
) -> torch.Tensor:
    # LoRA weights column major
    return torch.rand((num_loras, n, k), dtype=dtype).to(device)


def make_rand_tensors(
    a_shape: tuple[int],
    b_shape: tuple[int],
    c_shape: tuple[int],
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    num_slices: int,
    device: str = "cuda",
) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
    """
    Make LoRA input/output matrices.
    """
    A = torch.rand(a_shape, dtype=a_dtype).to(device)

    # LoRA weights column major
    Bs = [torch.rand(b_shape, dtype=b_dtype).to(device) for _ in range(num_slices)]

    C = torch.zeros(c_shape, dtype=c_dtype).to(device)
    return A, Bs, C


def make_prompt_lora_mapping(
    num_prompts: int, num_active_loras: int, sort_by_lora_id: bool, device: str
) -> torch.Tensor:
    """
    All prompts are mapped to a LoRA ID in range [0, num_active_loras).
    where 0 refers to first lora, 1 refers to second lora and so on.
    """
    assert num_active_loras > 0

    if not sort_by_lora_id:
        return torch.randint(0, num_active_loras, (num_prompts,), dtype=torch.long)

    # Divide LoRAs equally and in order.
    part_size = num_prompts // num_active_loras
    part_size = max(part_size, 1)

    lora_id = 0
    prompt_lora_mapping = []
    while len(prompt_lora_mapping) < num_prompts:
        prompt_lora_mapping.extend([lora_id] * part_size)
        lora_id = lora_id + 1 if lora_id + 1 < num_active_loras else lora_id
    return torch.tensor(
        prompt_lora_mapping[:num_prompts], dtype=torch.long, device=device
    )


def make_token_lora_mapping(
    num_tokens: int,
    num_prompts: int,
    prompt_lora_mapping: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    device: str,
):
    """
    Make token_lora_mapping from prompt_lora_mapping and seq_lens_tensor
    """
    assert prompt_lora_mapping.shape[0] == num_prompts

    # token to lora index mapping
    token_lora_mapping = [0] * num_tokens
    current_offset = 0
    for b_id in range(num_prompts):
        lora_index = prompt_lora_mapping[b_id].item()
        s = current_offset
        e = s + seq_len_tensor[b_id].item()
        token_lora_mapping[s:e] = [lora_index] * (e - s)
        current_offset += seq_len_tensor[b_id].item()

    return torch.tensor(token_lora_mapping, dtype=torch.long, device=device)


def ref_group_gemm(
    ref_out: torch.Tensor,
    input: torch.Tensor,
    lora_weights: list[torch.Tensor],
    seq_lens_cpu: torch.Tensor,
    prompt_lora_mapping_cpu: torch.Tensor,
    scaling: float,
    add_inputs: Optional[bool],
):
    """
    Torch group gemm reference implementation to test correctness of
    benchmarking operations.
    """
    batches = seq_lens_cpu.size(0)
    out_list = []
    current_offset = 0
    for lora_index, b_length in zip(range(batches), seq_lens_cpu):
        x = input[current_offset : b_length + current_offset, :]
        current_offset += b_length
        w = lora_weights[prompt_lora_mapping_cpu[lora_index]]
        result = torch.nn.functional.linear(x, w)
        result *= scaling
        out_list.append(result)

    cat_result = torch.cat(out_list, dim=0)

    if add_inputs:
        ref_out += cat_result
    else:
        ref_out.copy_(cat_result)


class OpType(Enum):
    """
    LoRA Ops to benchmark and its properties.
    """

    LORA_SHRINK = auto()
    LORA_EXPAND = auto()

    @staticmethod
    def from_str(s: str) -> "OpType":
        if s.lower() == "lora_shrink":
            return OpType.LORA_SHRINK
        if s.lower() == "lora_expand":
            return OpType.LORA_EXPAND
        raise ValueError(f"Unrecognized str {s} to convert to OpType")

    def is_shrink_fn(self) -> bool:
        return self in [OpType.LORA_SHRINK]

    def is_expand_fn(self) -> bool:
        return self in [OpType.LORA_EXPAND]

    def num_slices(self) -> list[int]:
        return [1, 2, 3]

    def mkn(
        self, batch_size: int, seq_length: int, hidden_size: int, lora_rank: int
    ) -> tuple[int, int, int]:
        num_tokens = batch_size * seq_length
        if self.is_shrink_fn():
            m = num_tokens
            k = hidden_size
            n = lora_rank
        else:
            assert self.is_expand_fn()
            m = num_tokens
            k = lora_rank
            n = hidden_size
        return m, k, n

    def matmul_dtypes(
        self, op_dtype: torch.dtype
    ) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        """
        return a type, b type and c type for A x B = C
        """
        if self.is_shrink_fn():
            return op_dtype, op_dtype, torch.float32
        else:
            assert self.is_expand_fn()
            return torch.float32, op_dtype, op_dtype

    def matmul_shapes(
        self,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        lora_rank: int,
        num_loras: int,
        num_slices: int,
    ) -> tuple[tuple[int], tuple[int], tuple[int]]:
        """
        Given num_slices, return the shapes of the A, B, and C matrices
        in A x B = C, for the op_type
        """
        m, k, n = self.mkn(batch_size, seq_length, hidden_size, lora_rank)

        b_shape = (num_loras, n, k)  # col-major
        if self in [OpType.LORA_SHRINK]:
            # LoRA shrink kernels support num_slices inherently in the kernel.
            return ((m, k), b_shape, (num_slices, m, n))
        if self in [OpType.LORA_EXPAND]:
            # LoRA expand kernels support num_slices inherently in the kernel
            return ((num_slices, m, k), b_shape, (m, n * num_slices))
        raise ValueError(f"Unrecognized op_type {self}")

    def bench_fn(self) -> Callable:
        if self == OpType.LORA_SHRINK:
            return lora_shrink
        if self == OpType.LORA_EXPAND:
            return lora_expand

        raise ValueError(f"Unrecognized optype {self}")

    def run_ref_group_gemm(
        self,
        output: torch.Tensor,
        input: torch.Tensor,
        lora_weights: list[torch.Tensor],
        **kwargs,
    ) -> Callable:
        """Each benchmark operation expects the input, lora_weights and outputs
        in a slightly different format. Refer to self.matmul_shapes().
        run_ref_group_gemm accounts for those differences in executing a
        reference group gemm for correctness testing.
        """
        w_dtype = lora_weights[0].dtype
        num_slices = len(lora_weights)
        if self in [OpType.LORA_SHRINK]:
            for slice_idx in range(num_slices):
                ref_group_gemm(
                    ref_out=output[slice_idx, :],
                    input=input,
                    lora_weights=lora_weights[slice_idx],
                    **kwargs,
                )
        elif self in [OpType.LORA_EXPAND]:
            hidden_size = lora_weights[0].shape[1]
            for slice_idx in range(num_slices):
                slice_offset = slice_idx * hidden_size
                ref_group_gemm(
                    ref_out=output[:, slice_offset : slice_offset + hidden_size],
                    input=input[slice_idx].clone().to(dtype=w_dtype),
                    lora_weights=lora_weights[slice_idx],
                    **kwargs,
                )
        else:
            raise ValueError(f"Unrecognized optype {self}")


@dataclass
class BenchmarkContext:
    """
    LoRA benchmark context
    """

    batch_size: int
    hidden_size: int
    num_loras: int
    num_active_loras: int
    lora_rank: int
    sort_by_lora_id: bool
    dtype: torch.dtype
    seq_length: Optional[int] = None
    num_slices: Optional[int] = None  # num_slices for slice based ops

    def with_seq_length(self, seq_length: int) -> "BenchmarkContext":
        ctx = copy.copy(self)
        ctx.seq_length = seq_length
        return ctx

    def with_num_slices(self, num_slices: int) -> "BenchmarkContext":
        ctx = copy.copy(self)
        ctx.num_slices = num_slices
        return ctx

    def bench_label(self) -> str:
        return f"lora-{self.dtype}"

    def bench_sublabel(self, op_type: OpType) -> str:
        m, k, n = op_type.mkn(
            self.batch_size, self.seq_length, self.hidden_size, self.lora_rank
        )
        desc = {
            "bs": self.batch_size,
            "sl": self.seq_length,
            "m": m,
            "k": k,
            "n": n,
            "num_loras": self.num_loras,
            "sort_by_lora": self.sort_by_lora_id,
            "num_slices": self.num_slices,
        }
        return json.dumps(desc)


@dataclass
class BenchmarkTensors:
    """
    Input/Output tensors used for benchmarks
    """

    # matmul tensors
    input: torch.Tensor
    lora_weights_lst: list[torch.Tensor]
    output: torch.Tensor
    # LoRA kernel metadata
    lora_kernel_meta: LoRAKernelMeta
    # Metadata tensors used in testing correctness
    seq_lens: torch.Tensor
    prompt_lora_mapping: torch.Tensor

    def io_types(self) -> str:
        return (
            f"{dtype_to_str(self.input.dtype)}x"
            f"{dtype_to_str(self.lora_weights_lst[0].dtype)}=>"
            f"{dtype_to_str(self.output.dtype)}"
        )

    @staticmethod
    def make(
        ctx: BenchmarkContext, op_type: OpType, device: str = "cuda"
    ) -> "BenchmarkTensors":
        # Make input / output matmul tensors.
        a_shape, b_shape, c_shape = op_type.matmul_shapes(
            ctx.batch_size,
            ctx.seq_length,
            ctx.hidden_size,
            ctx.lora_rank,
            ctx.num_loras,
            ctx.num_slices,
        )
        a_type, b_type, c_type = op_type.matmul_dtypes(ctx.dtype)
        input_tensor, lora_weights, output_tensor = make_rand_tensors(
            a_shape, b_shape, c_shape, a_type, b_type, c_type, num_slices=ctx.num_slices
        )

        # Make metadata tensors.
        # Keep the metadata tensors in the CPU for further processing if needed.
        # The tensors get moved to the GPU before benchmarking.
        assert ctx.num_active_loras <= ctx.num_loras
        total_tokens = ctx.batch_size * ctx.seq_length

        # Make metadata tensors involved in correctness testing.
        # Prepare seq lens tensor
        seq_len_tensor = torch.randint(
            ctx.seq_length, ctx.seq_length + 1, (ctx.batch_size,)
        )
        assert total_tokens == seq_len_tensor.sum()
        # Prepare prompt lora indices tensor
        prompt_lora_indices_tensor = make_prompt_lora_mapping(
            ctx.batch_size, ctx.num_active_loras, ctx.sort_by_lora_id, "cpu"
        )

        # Make LoRAKernelMeta
        token_lora_indices_tensor = make_token_lora_mapping(
            total_tokens,
            ctx.batch_size,
            prompt_lora_indices_tensor,
            seq_len_tensor,
            "cpu",
        )
        lora_kernel_meta = LoRAKernelMeta.make(
            max_loras=ctx.num_loras,
            max_num_tokens=token_lora_indices_tensor.size(0),
            device="cpu",
        )
        lora_kernel_meta.prepare_tensors(token_lora_mapping=token_lora_indices_tensor)

        return BenchmarkTensors(
            input_tensor,
            lora_weights,
            output_tensor,
            lora_kernel_meta,
            seq_len_tensor,
            prompt_lora_indices_tensor,
        )

    def sanity_check(self) -> None:
        """
        Fails asserts when non-conformality is detected.
        """
        num_tokens = self.input.shape[-2]
        # check metadata tensors
        assert torch.sum(self.seq_lens) == num_tokens
        num_seqs = self.seq_lens.shape[0]
        # assert self.seq_start_loc.shape[0] == num_seqs
        assert self.prompt_lora_mapping.shape[0] == num_seqs
        assert self.lora_kernel_meta.token_lora_mapping.shape[0] == num_tokens

    def to_device(self, device: str):
        """
        Transfer tensors to device if the tensors aren't already on the device
        """

        def to_device(tensor: torch.Tensor):
            if tensor.device != device:
                tensor = tensor.to(device=device)
            return tensor

        self.input = to_device(self.input)
        self.output = to_device(self.output)
        self.seq_lens = to_device(self.seq_lens)
        self.prompt_lora_mapping = to_device(self.prompt_lora_mapping)
        for i in range(len(self.lora_weights_lst)):
            self.lora_weights_lst[i] = to_device(self.lora_weights_lst[i])

        # LoRA meta
        for field_name in LoRAKernelMeta.__dataclass_fields__:
            field = getattr(self.lora_kernel_meta, field_name)
            assert isinstance(field, torch.Tensor)
            setattr(self.lora_kernel_meta, field_name, to_device(field))

    def metadata(self) -> tuple[int, int, int]:
        """
        Return num_seqs, num_tokens and max_seq_len
        """
        num_seqs = self.seq_lens.shape[0]
        num_tokens = self.lora_kernel_meta.token_lora_mapping.shape[0]
        max_seq_len = torch.max(self.seq_lens).item()
        num_slices = len(self.lora_weights_lst)
        return num_seqs, num_tokens, max_seq_len, num_slices

    def as_lora_shrink_kwargs(self) -> dict[str, Any]:
        self.sanity_check()
        self.to_device(self.input.device)

        _, num_tokens, _, num_slices = self.metadata()

        # Sanity check matrix shapes.
        i_shape, lw_shape, o_shape = (
            self.input.shape,
            self.lora_weights_lst[0].shape,
            self.output.shape,
        )
        # Expected input shape [num_tokens, hidden_size]
        assert len(i_shape) == 2
        assert i_shape[0] == num_tokens
        hidden_size = i_shape[1]
        # Expected lora weight shape [num_loras, lora_rank, hidden_size]
        assert len(lw_shape) == 3
        assert lw_shape[2] == hidden_size
        lora_rank = lw_shape[1]
        # Expected output shape [num_slices, num_tokens, lora_rank]
        assert len(o_shape) == 3
        assert o_shape == (num_slices, num_tokens, lora_rank)

        return {
            "inputs": self.input,
            "lora_a_weights": self.lora_weights_lst,
            "output_tensor": self.output,
            "token_lora_mapping": self.lora_kernel_meta.token_lora_mapping,
            "token_indices_sorted_by_lora_ids": (
                self.lora_kernel_meta.token_indices_sorted_by_lora_ids
            ),
            "num_tokens_per_lora": self.lora_kernel_meta.num_tokens_per_lora,
            "lora_token_start_loc": self.lora_kernel_meta.lora_token_start_loc,
            "lora_ids": self.lora_kernel_meta.active_lora_ids,
            "scaling": 1.0,
        }

    def as_lora_expand_kwargs(self, add_inputs: bool) -> dict[str, Any]:
        self.sanity_check()
        self.to_device(self.input.device)

        _, num_tokens, _, num_slices = self.metadata()

        # Sanity check matrix shapes.
        i_shape, lw_shape, o_shape = (
            self.input.shape,
            self.lora_weights_lst[0].shape,
            self.output.shape,
        )
        # Expected input shape : [num_slices, num_tokens, lora_rank]
        assert len(i_shape) == 3
        assert i_shape[0] == num_slices
        assert i_shape[1] == num_tokens
        lora_rank = i_shape[2]
        # Expected lora weight shape : [num_lora, hidden_size, lora_rank]
        assert len(lw_shape) == 3
        assert lw_shape[2] == lora_rank
        hidden_size = lw_shape[1]
        # Expected output shape : [num_tokens, hidden_size * num_slices]
        assert len(o_shape) == 2
        assert o_shape == (num_tokens, hidden_size * num_slices)

        return {
            "inputs": self.input,
            "lora_b_weights": self.lora_weights_lst,
            "output_tensor": self.output,
            "token_lora_mapping": self.lora_kernel_meta.token_lora_mapping,
            "token_indices_sorted_by_lora_ids": (
                self.lora_kernel_meta.token_indices_sorted_by_lora_ids
            ),
            "num_tokens_per_lora": self.lora_kernel_meta.num_tokens_per_lora,
            "lora_token_start_loc": self.lora_kernel_meta.lora_token_start_loc,
            "lora_ids": self.lora_kernel_meta.active_lora_ids,
            "offset_start": 0,
            "add_inputs": add_inputs,
        }

    def bench_fn_kwargs(
        self, op_type: OpType, add_inputs: Optional[bool] = None
    ) -> dict[str, Any]:
        if op_type.is_shrink_fn():
            assert add_inputs is None
        else:
            assert add_inputs is not None

        if op_type == OpType.LORA_SHRINK:
            return self.as_lora_shrink_kwargs()
        if op_type == OpType.LORA_EXPAND:
            return self.as_lora_expand_kwargs(add_inputs)
        raise ValueError(f"Unrecognized optype {self}")

    def test_correctness(
        self, op_type: OpType, expand_fn_add_inputs: Optional[bool]
    ) -> bool:
        """
        Test correctness of op_type implementation against a grouped gemm
        reference implementation.
        """
        seq_lens_cpu = self.seq_lens.to(device="cpu")
        prompt_lora_mapping_cpu = self.prompt_lora_mapping.to(device="cpu")
        ref_output = self.output.clone()

        self.output.zero_()
        op_type.bench_fn()(**self.bench_fn_kwargs(op_type, expand_fn_add_inputs))

        op_type.run_ref_group_gemm(
            ref_output,
            self.input,
            self.lora_weights_lst,
            seq_lens_cpu=seq_lens_cpu,
            prompt_lora_mapping_cpu=prompt_lora_mapping_cpu,
            scaling=1.0,
            add_inputs=expand_fn_add_inputs,
        )

        rtol, atol = {
            torch.float16: (6e-2, 6e-2),
            torch.bfloat16: (6e-2, 6e-2),
            torch.float32: (1e-2, 1e-2),
        }[self.output.dtype]

        return torch.allclose(ref_output, self.output, rtol=rtol, atol=atol)


def bench_optype(
    ctx: BenchmarkContext,
    arg_pool_size: int,
    op_type: OpType,
    cuda_graph_nops: Optional[int] = None,
    expand_fn_add_inputs: Optional[bool] = None,
    test_correctness: bool = False,
) -> TMeasurement:
    assert arg_pool_size >= 1
    if op_type.is_shrink_fn():
        assert expand_fn_add_inputs is None
    else:
        assert expand_fn_add_inputs is not None

    # BenchmarkContext -> BenchmarkTensors
    bench_tensors: list[BenchmarkTensors] = [
        BenchmarkTensors.make(ctx, op_type) for _ in range(arg_pool_size)
    ]
    for bt in bench_tensors:
        bt.sanity_check()

    # Test correctness of our implementation.
    if test_correctness:
        assert all(
            [bt.test_correctness(op_type, expand_fn_add_inputs) for bt in bench_tensors]
        )

    # BenchmarkTensors -> dict (kwargs)
    kwargs_list = [
        bt.bench_fn_kwargs(op_type, add_inputs=expand_fn_add_inputs)
        for bt in bench_tensors
    ]

    # Clear LoRA optimization hash-maps.
    _LORA_A_PTR_DICT.clear()
    _LORA_B_PTR_DICT.clear()
    # Run bench function so that _LORA_A_PTR_DICT and _LORA_B_PTR_DICT are setup
    for kwargs in kwargs_list:
        op_type.bench_fn()(**kwargs)
    torch.cuda.synchronize()

    # Merge into a single kwargs and qualify arguments as ArgPool
    kwargs = {k: ArgPool([]) for k in kwargs_list[0]}
    for _kwargs in kwargs_list:
        for k, v in _kwargs.items():
            kwargs[k].values.append(v)

    describe_args = (
        f"add_inputs={expand_fn_add_inputs}" if expand_fn_add_inputs is not None else ""
    )
    description = f"{op_type.name}({describe_args}) ({bench_tensors[0].io_types()})"

    cuda_graph_params = None
    if cuda_graph_nops:
        cuda_graph_params = CudaGraphBenchParams(cuda_graph_nops)
    timer = None
    with Bench(
        cuda_graph_params,
        ctx.bench_label(),
        ctx.bench_sublabel(op_type),
        description,
        op_type.bench_fn(),
        **kwargs,
    ) as bench:
        timer = bench.run()
    return timer


def bench_torch_mm(
    ctx: BenchmarkContext,
    arg_pool_size: int,
    op_type: OpType,
    cuda_graph_nops: Optional[int] = None,
) -> TMeasurement:
    """
    Benchmark basic torch.mm as a roofline.

    When all the input tokens have the same LoRA ID, the LoRA kernels are just
    a matmul. This torch.mm benchmark serves as a roofline for that case.

    input op_type is used in determining the m, k, n dimensions for the matmul.
    """

    batch_size, hidden_size, lora_rank, seq_length, dtype = (
        ctx.batch_size,
        ctx.hidden_size,
        ctx.lora_rank,
        ctx.seq_length,
        ctx.dtype,
    )

    m, k, n = op_type.mkn(batch_size, seq_length, hidden_size, lora_rank)
    # For a fairer comparison.
    n = n * ctx.num_slices

    # Get matmul input and output tensors for A x B = C
    As, Bs, Cs = [], [], []
    for _ in range(arg_pool_size):
        As.append(torch.rand((m, k), dtype=dtype).to("cuda"))
        Bs.append(torch.rand((n, k), dtype=dtype).to("cuda").t())
        Cs.append(torch.rand((m, n), dtype=dtype).to("cuda"))

    # Make torch.mm kwargs
    mm_kwargs = {"input": ArgPool(As), "mat2": ArgPool(Bs), "out": ArgPool(Cs)}

    description = (
        f"single-lora roofline using torch.mm ({dtype_to_str(dtype)}"
        f"x{dtype_to_str(dtype)}"
        f"=>{dtype_to_str(dtype)})"
    )
    cuda_graph_params = None
    if cuda_graph_nops:
        cuda_graph_params = CudaGraphBenchParams(cuda_graph_nops)
    with Bench(
        cuda_graph_params,
        ctx.bench_label(),
        ctx.bench_sublabel(op_type),
        description,
        torch.mm,
        **mm_kwargs,
    ) as bench:
        return bench.run()


# runner
def use_cuda_graph_recommendation() -> str:
    return """
            Triton kernels have a significant launch overhead with
            launched directly via python. This overhead is more noticeable
            for small the problem sizes. For these cases, it is recommended
            to use the script with `--cuda-graph-nops N` to benchmark N
            consecutive invocations of the benchmarking operations from 
            inside a CUDA Graph. Note that the returned measurement is for N 
            invocations of the operation.
            """


def print_timers(timers: list[TMeasurement], args: Optional[argparse.Namespace] = None):
    compare = TBenchmark.Compare(timers)
    compare.print()

    if args and args.cuda_graph_nops:
        print(
            f"Note : The timings reported above is for {args.cuda_graph_nops} "
            "consecutive invocations of the benchmarking functions. "
            f"Please divide by {args.cuda_graph_nops} for single invocation "
            "timings."
        )

    print(
        "Note on Comparison with torch.mm : The torch.mm numbers are "
        "benchmark numbers of a simple matmul emulating the single lora "
        "case. It is provided as a roofline for comparing our LoRA Kernel "
        "implementations. It is expected that the LoRA kernels will be "
        "slower than torch.mm in cases where num_loras is big. But for "
        "small num_loras the goal should be to match the torch.mm numbers."
    )


def run(args: argparse.Namespace, bench_ctxs: list[BenchmarkContext]):
    if args.cuda_graph_nops is not None:
        assert args.cuda_graph_nops > 0
        print(f"Benchmarking {args.cuda_graph_nops} invocations inside a CUDA Graph")
    else:
        print(f"CUDA Graphs not enabled.\n{use_cuda_graph_recommendation()}")

    timers = []
    for bench_ctx in bench_ctxs:
        for seq_len in args.seq_lengths:
            bench_ops: list[OpType] = args.op_types
            seq_len_timers = []
            for bench_op in bench_ops:
                for num_slices in bench_op.num_slices():
                    _ctx = bench_ctx.with_seq_length(seq_len).with_num_slices(
                        num_slices
                    )
                    # Benchmark torch.mm as a roofline
                    seq_len_timers.append(
                        bench_torch_mm(
                            _ctx, args.arg_pool_size, bench_op, args.cuda_graph_nops
                        )
                    )

                    # Benchmark bench_op
                    expand_fn_add_inputs = (
                        [None] if bench_op.is_shrink_fn() else args.expand_fn_add_inputs
                    )
                    for add_input_arg in expand_fn_add_inputs:
                        seq_len_timers.append(
                            bench_optype(
                                _ctx,
                                args.arg_pool_size,
                                bench_op,
                                args.cuda_graph_nops,
                                add_input_arg,
                                args.test_correctness,
                            )
                        )

            print_timers(seq_len_timers)
            timers.extend(seq_len_timers)

    # Result stdout dump
    print("== All Results ====")
    print_timers(timers, args)

    if args.output_directory:
        # Result file dump
        od = Path(args.output_directory)
        if not od.exists():
            od.mkdir()

        timestamp = int(time.time())
        pkl_file = od / f"lora_bench-{timestamp}.pkl"
        print(f"Writing benchmarks to {pkl_file}")
        with open(pkl_file, "wb") as f:
            pickle.dump(timers, f)


def as_benchmark_contexts(
    hidden_sizes: list[int], lora_ranks: list[int], args: argparse.Namespace
) -> list[BenchmarkContext]:
    ctxs: list[BenchmarkContext] = []
    for batch_size, hidden_size, lora_rank, num_loras, sort_by_lora_id in product(  # noqa
        args.batch_sizes,
        list(hidden_sizes),
        lora_ranks,
        args.num_loras,
        args.sort_by_lora_id,
    ):
        ctxs.append(
            BenchmarkContext(
                batch_size=batch_size,
                hidden_size=hidden_size,
                lora_rank=lora_rank,
                num_loras=num_loras,
                num_active_loras=args.num_active_loras
                if args.num_active_loras
                else num_loras,
                # To be filled based on the OpType to benchmark
                seq_length=None,
                sort_by_lora_id=sort_by_lora_id,
                dtype=args.dtype,
                # To be filled based on the OpType to benchmark
                num_slices=None,
            )
        )

    return ctxs


def run_list_bench(args: argparse.Namespace):
    print(args)

    print(
        "List bench :\n"
        f"  Hidden Sizes {args.hidden_sizes}"
        f"  LoRA Ranks {args.lora_ranks}"
    )

    # Get all benchmarking contexts
    bench_contexts: list[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=args.hidden_sizes, lora_ranks=args.lora_ranks, args=args
    )

    run(args, bench_contexts)


def run_range_bench(args: argparse.Namespace):
    print(args)

    hidden_sizes = list(
        range(
            args.hidden_sizes_start,
            args.hidden_sizes_end + 1,
            args.hidden_sizes_increment,
        )
    )
    lora_ranks = list(
        range(args.lora_ranks_start, args.lora_ranks_end + 1, args.lora_ranks_increment)
    )

    print(f"Range bench :\n Hidden Sizes {hidden_sizes} LoRA Ranks {lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: list[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=hidden_sizes, lora_ranks=lora_ranks, args=args
    )

    run(args, bench_contexts)


def run_model_bench(args: argparse.Namespace):
    print(args)

    def hidden_sizes_from_model(model: str, tp_size: int) -> set[int]:
        hidden_sizes = set()
        for KN, tp_split_dim in WEIGHT_SHAPES[model]:
            KN[tp_split_dim] = KN[tp_split_dim] // tp_size
            hidden_sizes.add(KN[1])
        return hidden_sizes

    # Get all hidden sizes
    hidden_sizes: set[int] = set()
    for model_name, tp_size in product(args.models, args.tp_sizes):
        hidden_sizes = hidden_sizes.union(hidden_sizes_from_model(model_name, tp_size))

    print(f"Model bench :\n Hidden Sizes {hidden_sizes} LoRA Ranks {args.lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: list[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=hidden_sizes, lora_ranks=args.lora_ranks, args=args
    )

    run(args, bench_contexts)


if __name__ == "__main__":

    def to_torch_dtype(dt):
        if dt == "torch.float16":
            return torch.float16
        if dt == "torch.bfloat16":
            return torch.bfloat16
        raise ValueError("unsupported dtype")

    def get_bool(s: str) -> bool:
        return s.lower() in ["true", "1"]

    def add_common_command_args(p: argparse.ArgumentParser):
        p.add_argument(
            "--dtype",
            type=to_torch_dtype,
            required=True,
            help="Available options are ['torch.float16', 'torch.bfloat16']",
        )

        p.add_argument(
            "--arg-pool-size",
            type=int,
            default=32,
            help="Run profiles with a pool of input/output/meta tensors instead"
            "of simply reusing the same tensors for all runs. A bigger arg-pool"
            "mitigates hardware caching effects during benchmarking.",
        )

        p.add_argument(
            "--cuda-graph-nops",
            type=int,
            help=(
                "when set profiling is done using cudagraph, "
                "with the given number of operations in a graph."
                "Note that the measurement returned is the time "
                "taken for N consecutive executions of the benchmarking "
                "functions, where N is the value of this argument."
            ),
        )
        p.add_argument("--num-loras", nargs="+", type=int, default=DEFAULT_NUM_LORAS)
        p.add_argument(
            "--num-active-loras",
            type=int,
            default=None,
            help="Active LoRAs. When None, all LoRAs are active",
        )
        p.add_argument(
            "--sort-by-lora-id",
            nargs="+",
            type=get_bool,
            default=DEFAULT_SORT_BY_LORA_IDS,
        )
        p.add_argument(
            "--op-types", nargs="+", type=OpType.from_str, default=list(OpType)
        )
        p.add_argument(
            "--seq-lengths", nargs="+", type=int, default=DEFAULT_SEQ_LENGTHS
        )
        p.add_argument(
            "--batch-sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES
        )
        p.add_argument(
            "--expand-fn-add-inputs",
            nargs="+",
            type=get_bool,
            default=DEFAULT_EXPAND_FN_ADD_INPUTS,
        )
        p.add_argument(
            "-o",
            "--output-directory",
            type=str,
            help=(
                "Output directory to store a the list of benchmarking"
                "TMeasurement objects as a pickle file"
            ),
        )

        p.add_argument(
            "--test-correctness",
            action="store_true",
            help=(
                "When enabled, the benchmarking functions are tested"
                "for correctness before the actual benchmarking"
            ),
        )

    parser = FlexibleArgumentParser(
        description=f"""
Benchmark LoRA kernels:
    {use_cuda_graph_recommendation()}

    list_bench example:
        python3 benchmarks/kernels/benchmark_lora.py list_bench --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16 --hidden-sizes 2048 --lora-ranks 16 --num-loras 1 4 --op-types lora_shrink lora_expand --seq-lengths 1 16 --sort-by-lora-id 1 --cuda-graph-nops 32

    model_bench example:
        python3 benchmarks/kernels/benchmark_lora.py model_bench --models meta-llama/Llama-3-8b  --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16  --lora-ranks 16 --num-loras 1 4 --op-types lora_shrink lora_expand --seq-lengths 1 16 --sort-by-lora-id 1 --cuda-graph-nops 32 

    range_bench example:
        python3 benchmarks/kernels/benchmark_lora.py range_bench  --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16   --num-loras 1 4 --op-types lora_shrink lora_expand --seq-lengths 1 16 --sort-by-lora-id 1 --cuda-graph-nops 32 --hidden-sizes-start 1024 --hidden-sizes-end 4096 --hidden-sizes-increment 1024 --lora-ranks-start 8 --lora-ranks-end 24 --lora-ranks-increment 8 
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    list_parser = subparsers.add_parser("list_bench")
    list_parser.add_argument(
        "--hidden-sizes", nargs="+", type=int, default=DEFAULT_HIDDEN_SIZES
    )
    list_parser.add_argument(
        "--lora-ranks", nargs="+", type=int, default=DEFAULT_LORA_RANKS
    )
    add_common_command_args(list_parser)
    list_parser.set_defaults(func=run_list_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--hidden-sizes-start", type=int, required=True)
    range_parser.add_argument("--hidden-sizes-end", type=int, required=True)
    range_parser.add_argument("--hidden-sizes-increment", type=int, required=True)
    range_parser.add_argument("--lora-ranks-start", type=int, required=True)
    range_parser.add_argument("--lora-ranks-end", type=int, required=True)
    range_parser.add_argument("--lora-ranks-increment", type=int, required=True)
    add_common_command_args(range_parser)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=DEFAULT_MODELS,
        choices=WEIGHT_SHAPES.keys(),
    )
    model_parser.add_argument(
        "--tp-sizes", nargs="+", type=int, default=DEFAULT_TP_SIZES
    )
    model_parser.add_argument(
        "--lora-ranks", nargs="+", type=int, default=DEFAULT_LORA_RANKS
    )
    add_common_command_args(model_parser)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
