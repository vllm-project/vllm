import argparse
import copy
import json
import pickle
import time
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.utils.benchmark as TBenchmark
from torch.utils.benchmark import Measurement as TMeasurement
from utils import ArgPool, Bench, CudaGraphBenchParams
from weight_shapes import WEIGHT_SHAPES

from vllm.lora.ops.bgmv_expand import bgmv_expand
from vllm.lora.ops.bgmv_expand_slice import bgmv_expand_slice
from vllm.lora.ops.bgmv_shrink import bgmv_shrink
from vllm.lora.ops.sgmv_expand import sgmv_expand
from vllm.lora.ops.sgmv_expand_slice import sgmv_expand_slice
from vllm.lora.ops.sgmv_shrink import sgmv_shrink
from vllm.utils import FlexibleArgumentParser

DEFAULT_MODELS = list(WEIGHT_SHAPES.keys())
DEFAULT_TP_SIZES = [1]
DEFAULT_BATCH_SIZES = [
    1, 16, 32, 64, 128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024,
    2048, 3072, 4096, 5120, 6144, 7168, 8192
]
DEFAULT_HIDDEN_SIZES = [1024, 2048, 4096, 8192, 16384]
DEFAULT_LORA_RANKS = [16]
DEFAULT_NUM_LORAS = [1, 2, 3, 4]
DEFAULT_SORT_BY_LORA_IDS = [False, True]
DEFAULT_SEQ_LENGTHS = [1]
DEFAULT_EXPAND_FN_ADD_INPUTS = [True, False]


## Utilities
def dtype_to_str(dtype: torch.dtype):
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "f32"
    raise ValueError(f"Unsupported dtype {dtype}")


def make_rand_lora_weight_tensor(k: int,
                                 n: int,
                                 num_loras: int,
                                 dtype: torch.dtype,
                                 device: str = "cuda") -> torch.Tensor:

    # LoRA weights column major
    return torch.rand((num_loras, n, k), dtype=dtype).to(device)


def make_rand_tensors(
    m: int,
    k: int,
    n: int,
    num_loras: int,
    num_slices: Optional[int],
    a_dtype: torch.dtype,
    b_dtype: torch.dtype,
    c_dtype: torch.dtype,
    device: str = "cuda",
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    # Make input / output tensors
    # Input matrix A of shape {m, k}
    # num_slices Input matrices B of shape {k, n}
    # Output matrix C of shape {m, n * num_slices}
    num_slices = num_slices if num_slices is not None else 1

    A = torch.rand((m, k), dtype=a_dtype).to(device)

    # LoRA weights column major
    Bs = [
        make_rand_lora_weight_tensor(k, n, num_loras, b_dtype, device)
        for _ in range(num_slices)
    ]

    C = torch.zeros((m, n * num_slices), dtype=c_dtype).to(device)

    return A, Bs, C


def make_prompt_lora_mapping(num_prompts: int, num_active_loras: int,
                             sort_by_lora_id: bool,
                             device: str) -> torch.Tensor:
    """
    All prompts are mapped to a Lora ID in range [0, num_active_loras).
    where 0 refers to first lora, 1 refers to second lora and so on.
    """
    assert num_active_loras > 0

    if not sort_by_lora_id:
        return torch.randint(0,
                             num_active_loras, (num_prompts, ),
                             dtype=torch.long)

    # Divide LoRAs equally and in order.
    part_size = num_prompts // num_active_loras
    part_size = max(part_size, 1)

    lora_id = 0
    prompt_lora_mapping = []
    while len(prompt_lora_mapping) < num_prompts:
        prompt_lora_mapping.extend([lora_id] * part_size)
        lora_id = lora_id + 1 if lora_id + 1 < num_active_loras else lora_id
    return torch.tensor(prompt_lora_mapping[:num_prompts],
                        dtype=torch.long,
                        device=device)


def make_token_lora_mapping(num_tokens: int, num_prompts: int,
                            prompt_lora_mapping: torch.Tensor,
                            seq_len_tensor: torch.Tensor, device: str):
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


## LoRA Ops to Benchmark and its properties
class OpType(Enum):
    SGMV_SHRINK = auto()
    BGMV_SHRINK = auto()
    SGMV_EXPAND = auto()
    BGMV_EXPAND = auto()
    SGMV_EXPAND_SLICE = auto()
    BGMV_EXPAND_SLICE = auto()

    @staticmethod
    def from_str(s: str) -> "OpType":
        if s.lower() == 'sgmv_shrink':
            return OpType.SGMV_SHRINK
        if s.lower() == 'sgmv_expand':
            return OpType.SGMV_EXPAND
        if s.lower() == 'bgmv_shrink':
            return OpType.BGMV_SHRINK
        if s.lower() == 'bgmv_expand':
            return OpType.BGMV_EXPAND
        if s.lower() == "sgmv_expand_slice":
            return OpType.SGMV_EXPAND_SLICE
        if s.lower() == "bgmv_expand_slice":
            return OpType.BGMV_EXPAND_SLICE
        raise ValueError(f"Unrecognized str {s} to convert to OpType")

    def is_shrink_fn(self) -> bool:
        return self in [OpType.SGMV_SHRINK, OpType.BGMV_SHRINK]

    def is_expand_fn(self) -> bool:
        return self in [OpType.SGMV_EXPAND, OpType.BGMV_EXPAND]

    def is_expand_slice_fn(self) -> bool:
        return self in [OpType.SGMV_EXPAND_SLICE, OpType.BGMV_EXPAND_SLICE]

    def is_prefill_op(self) -> bool:
        return self in [
            OpType.SGMV_SHRINK, OpType.SGMV_EXPAND, OpType.SGMV_EXPAND_SLICE
        ]

    def is_decode_op(self) -> bool:
        return self in [
            OpType.BGMV_SHRINK, OpType.BGMV_EXPAND, OpType.BGMV_EXPAND_SLICE
        ]

    def num_slices(self) -> List[int]:
        if self.is_expand_slice_fn():
            return [2, 3]
        return [1]

    def mkn(self, batch_size: int, seq_length: int, hidden_size: int,
            lora_rank: int) -> Tuple[int, int, int]:
        num_tokens = batch_size * seq_length
        if self.is_shrink_fn():
            m = num_tokens
            k = hidden_size
            n = lora_rank
        else:
            assert self.is_expand_fn() or self.is_expand_slice_fn()
            m = num_tokens
            k = lora_rank
            n = hidden_size
        return m, k, n

    def matmul_dtypes(
            self, op_dtype: torch.dtype
    ) -> Tuple[torch.dtype, torch.dtype, torch.dtype]:
        """
        return a type, b type and c type for A x B = C
        """
        if self.is_shrink_fn():
            return op_dtype, op_dtype, torch.float32
        else:
            assert self.is_expand_fn() or self.is_expand_slice_fn()
            return torch.float32, op_dtype, op_dtype

    def bench_fn(self) -> Callable:

        def emulate_sgmv_expand_slice(kwargs_list: List[Dict[str, Any]]):
            for x in kwargs_list:
                sgmv_expand_slice(**x)

        def emulate_bgmv_expand_slice(kwargs_list: List[Dict[str, Any]]):
            for x in kwargs_list:
                bgmv_expand_slice(**x)

        if self == OpType.SGMV_SHRINK:
            return sgmv_shrink
        if self == OpType.SGMV_EXPAND:
            return sgmv_expand
        if self == OpType.BGMV_SHRINK:
            return bgmv_shrink
        if self == OpType.BGMV_EXPAND:
            return bgmv_expand
        if self == OpType.SGMV_EXPAND_SLICE:
            return emulate_sgmv_expand_slice
        if self == OpType.BGMV_EXPAND_SLICE:
            return emulate_bgmv_expand_slice
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
    num_slices: Optional[int] = None  # num_slices for expand_slice kernels

    def with_seq_length(self, seq_length: int) -> "BenchmarkContext":
        ctx = copy.copy(self)
        ctx.seq_length = seq_length
        return ctx

    def with_num_slices(self, num_slices: Optional[int]) -> "BenchmarkContext":
        ctx = copy.copy(self)
        ctx.num_slices = num_slices
        return ctx

    def bench_label(self) -> str:
        return f"lora-{self.dtype}"

    def bench_sublabel(self, op_type: OpType) -> str:
        m, k, n = op_type.mkn(self.batch_size, self.seq_length,
                              self.hidden_size, self.lora_rank)
        desc = {
            'bs': self.batch_size,
            'sl': self.seq_length,
            'm': m,
            'k': k,
            'n': n,
            'num_loras': self.num_loras,
            'sort_by_lora': self.sort_by_lora_id,
            'num_slices': self.num_slices,
        }
        return json.dumps(desc)


@dataclass
class BenchmarkTensors:
    """
    Input/Output tensors used for benchmarks
    """
    # matmul tensors
    input: torch.Tensor
    lora_weights_lst: List[torch.Tensor]
    output: torch.Tensor
    # metadata tensors
    seq_lens: torch.Tensor
    seq_start_loc: torch.Tensor
    prompt_lora_mapping: torch.Tensor
    token_lora_mapping: torch.Tensor

    def io_types(self) -> str:
        return (f"{dtype_to_str(self.input.dtype)}x"
                f"{dtype_to_str(self.lora_weights_lst[0].dtype)}=>"
                f"{dtype_to_str(self.output.dtype)}")

    @staticmethod
    def make(ctx: BenchmarkContext,
             op_type: OpType,
             device: str = "cuda") -> "BenchmarkTensors":

        ## Make input / output matmul tensors
        a_type, b_type, c_type = op_type.matmul_dtypes(ctx.dtype)
        m, k, n = op_type.mkn(ctx.batch_size, ctx.seq_length, ctx.hidden_size,
                              ctx.lora_rank)
        input_tensor, lora_weights, output_tensor = \
            make_rand_tensors(m, k, n, ctx.num_loras,
                              num_slices = ctx.num_slices,
                              a_dtype = a_type,
                              b_dtype = b_type,
                              c_dtype = c_type)

        ## Make metadata tensors
        # Keep the metadata tensors in the CPU for further processing if needed.
        # The tensors get moved to the GPU before benchmarking.
        assert ctx.num_active_loras <= ctx.num_loras
        total_tokens = ctx.batch_size * ctx.seq_length

        # Prepare seq lens tensor
        seq_len_tensor = torch.randint(ctx.seq_length, ctx.seq_length + 1,
                                       (ctx.batch_size, ))
        # Prepare seq_start_loc tensor
        seq_start_loc_tensor = torch.cumsum(torch.tensor(
            [0] + seq_len_tensor[:-1].tolist(), dtype=torch.long),
                                            dim=0)
        assert total_tokens == seq_len_tensor.sum()
        # Prepare prompt lora indices tensor
        prompt_lora_indices_tensor = make_prompt_lora_mapping(
            ctx.batch_size, ctx.num_active_loras, ctx.sort_by_lora_id, "cpu")
        # Prepare token lora indices tensor
        token_lora_indices_tensor = make_token_lora_mapping(
            total_tokens, ctx.batch_size, prompt_lora_indices_tensor,
            seq_len_tensor, "cpu")

        return BenchmarkTensors(input_tensor, lora_weights, output_tensor,
                                seq_len_tensor, seq_start_loc_tensor,
                                prompt_lora_indices_tensor,
                                token_lora_indices_tensor)

    def sanity_check(self) -> None:
        """
        Fails asserts when non-conformality is detected.
        """
        # Check that the tensors have the right shapes
        m = self.input.shape[0]
        k = self.input.shape[1]
        n = self.output.shape[1]

        # check matmul tensors
        assert self.output.shape[0] == m
        assert len(self.lora_weights_lst) >= 1
        num_slices = len(self.lora_weights_lst)
        for w in self.lora_weights_lst:
            _, w_n, w_k = w.shape  # n, k flipped due to col-major ordering.
            assert (w_n, w_k) == (n, k) or (w_n * num_slices, w_k) == (n, k)
        # check metadata tensors
        assert torch.sum(self.seq_lens) == m
        num_seqs = self.seq_lens.shape[0]
        assert self.seq_start_loc.shape[0] == num_seqs
        assert self.prompt_lora_mapping.shape[0] == num_seqs
        assert self.token_lora_mapping.shape[0] == m

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
        self.seq_start_loc = to_device(self.seq_start_loc)
        self.prompt_lora_mapping = to_device(self.prompt_lora_mapping)
        self.token_lora_mapping = to_device(self.token_lora_mapping)
        for i in range(len(self.lora_weights_lst)):
            self.lora_weights_lst[i] = to_device(self.lora_weights_lst[i])

    def metadata(self) -> Tuple[int, int, int]:
        """
        Return num_seqs, num_tokens and max_seq_len
        """
        num_seqs = self.seq_lens.shape[0]
        num_tokens = self.input.shape[0]
        max_seq_len = torch.max(self.seq_lens).item()
        return num_seqs, num_tokens, max_seq_len

    def convert_to_sgmv_benchmark_tensors(self):
        """
        for sgmv punica kernels, when consecutive sequences have the
        same LoRA ID, we just merge them together.
        This happens in punica.py::compute_metadata
        """

        # Collapse seq_lens and seq_start_loc
        _, seq_lens = torch.unique_consecutive(self.token_lora_mapping,
                                               return_counts=True)
        cum_result = torch.cumsum(seq_lens, dim=0)
        seq_start_loc = torch.zeros_like(seq_lens)
        seq_start_loc[1:].copy_(cum_result[:-1])

        # Collapse prompt mapping
        prompt_lora_mapping = torch.unique_consecutive(
            self.prompt_lora_mapping)

        assert torch.sum(seq_lens) == torch.sum(self.seq_lens), \
         f"dont match - new {torch.sum(seq_lens)} vs {torch.sum(self.seq_lens)}"

        self.prompt_lora_mapping = prompt_lora_mapping.to(
            dtype=self.prompt_lora_mapping.dtype)
        self.seq_lens = seq_lens.to(dtype=self.seq_lens.dtype)
        self.seq_start_loc = seq_start_loc.to(dtype=self.seq_start_loc.dtype)

    ## Benchmark function args.
    def as_sgmv_shrink_kwargs(self) -> Dict[str, Any]:
        assert len(self.lora_weights_lst) == 1

        self.convert_to_sgmv_benchmark_tensors()
        self.sanity_check()
        self.to_device(self.input.device)

        num_seqs, num_tokens, max_seq_len = self.metadata()
        return {
            'inputs': self.input,
            'lora_a_weights': self.lora_weights_lst[0],
            'output_tensor': self.output,
            'b_seq_start_loc': self.seq_start_loc,
            'seq_len_tensor': self.seq_lens,
            'lora_indices_tensor': self.prompt_lora_mapping,
            'batches': num_seqs,
            'max_seq_length': max_seq_len,
            'token_nums': num_tokens,
            'scaling': 1.0,
        }

    def as_sgmv_expand_kwargs(self, add_inputs: bool) -> Dict[str, Any]:
        assert len(self.lora_weights_lst) == 1

        self.convert_to_sgmv_benchmark_tensors()
        self.sanity_check()
        self.to_device(self.input.device)

        num_seqs, num_tokens, max_seq_len = self.metadata()
        return {
            'inputs': self.input,
            'lora_b_weights': self.lora_weights_lst[0],
            'output_tensor': self.output,
            'b_seq_start_loc': self.seq_start_loc,
            'seq_len_tensor': self.seq_lens,
            'lora_indices_tensor': self.prompt_lora_mapping,
            'batches': num_seqs,
            'max_seq_length': max_seq_len,
            'token_nums': num_tokens,
            'add_inputs': add_inputs,
        }

    def as_bgmv_shrink_kwargs(self) -> Dict[str, Any]:
        assert len(self.lora_weights_lst) == 1
        self.to_device(self.input.device)
        return {
            'inputs': self.input,
            'lora_a_weights': self.lora_weights_lst[0],
            'output_tensor': self.output,
            'lora_indices_tensor': self.token_lora_mapping,
            'scaling': 1.0
        }

    def as_bgmv_expand_kwargs(self, add_inputs: bool):
        assert len(self.lora_weights_lst) == 1
        self.to_device(self.input.device)
        return {
            'inputs': self.input,
            'lora_b_weights': self.lora_weights_lst[0],
            'output_tensor': self.output,
            'lora_indices_tensor': self.token_lora_mapping,
            'add_inputs': add_inputs
        }

    def as_sgmv_expand_slice_kwargs(self, add_inputs: bool) -> Dict[str, Any]:
        assert len(self.lora_weights_lst) > 1
        self.convert_to_sgmv_benchmark_tensors()
        self.sanity_check()

        self.to_device(self.input.device)
        num_seqs, num_tokens, max_seq_len = self.metadata()

        num_slices = len(self.lora_weights_lst)
        slice_size = self.lora_weights_lst[0].shape[-2]  # n
        assert slice_size * num_slices == self.output.shape[-1]

        kwargs_list = []
        for i in range(num_slices):
            kwargs_list.append({
                'inputs': self.input,
                'lora_b_weights': self.lora_weights_lst[i],
                'output_tensor': self.output,
                'b_seq_start_loc': self.seq_start_loc,
                'seq_len_tensor': self.seq_lens,
                'lora_indices_tensor': self.prompt_lora_mapping,
                'batches': num_seqs,
                'max_seq_length': max_seq_len,
                'token_nums': num_tokens,
                'slice_offset': i * slice_size,
                'slice_size': slice_size,
                'add_inputs': add_inputs,
            })
        return {'kwargs_list': kwargs_list}

    def as_bgmv_expand_slice_kwargs(self, add_inputs: bool) -> Dict[str, Any]:
        assert len(self.lora_weights_lst) > 1
        num_slices = len(self.lora_weights_lst)
        slice_size = self.lora_weights_lst[0].shape[-2]  # n
        assert slice_size * num_slices == self.output.shape[-1]

        self.to_device(self.input.device)

        kwargs_list = []
        for i in range(num_slices):
            kwargs_list.append({
                'inputs': self.input,
                'lora_b_weights': self.lora_weights_lst[i],
                'output_tensor': self.output,
                'lora_indices_tensor': self.token_lora_mapping,
                'slice_offset': i * slice_size,
                'slice_size': slice_size,
                'add_inputs': add_inputs,
            })
        return {'kwargs_list': kwargs_list}

    def bench_fn_kwargs(self,
                        op_type: OpType,
                        add_inputs: Optional[bool] = None) -> Dict[str, Any]:
        if op_type.is_shrink_fn():
            assert add_inputs is None
        else:
            assert add_inputs is not None

        if op_type == OpType.SGMV_SHRINK:
            return self.as_sgmv_shrink_kwargs()
        if op_type == OpType.SGMV_EXPAND:
            return self.as_sgmv_expand_kwargs(add_inputs)
        if op_type == OpType.BGMV_SHRINK:
            return self.as_bgmv_shrink_kwargs()
        if op_type == OpType.BGMV_EXPAND:
            return self.as_bgmv_expand_kwargs(add_inputs)
        if op_type == OpType.SGMV_EXPAND_SLICE:
            return self.as_sgmv_expand_slice_kwargs(add_inputs)
        if op_type == OpType.BGMV_EXPAND_SLICE:
            return self.as_bgmv_expand_slice_kwargs(add_inputs)
        raise ValueError(f"Unrecognized optype {self}")


def bench_optype(ctx: BenchmarkContext,
                 arg_pool_size: int,
                 op_type: OpType,
                 with_cuda_graph: bool = False,
                 expand_fn_add_inputs: Optional[bool] = None) -> TMeasurement:

    assert arg_pool_size >= 1
    if op_type.is_shrink_fn():
        assert expand_fn_add_inputs is None
    else:
        assert expand_fn_add_inputs is not None

    # BenchmarkContext -> BenchmarkTensors
    bench_tensors : List[BenchmarkTensors] = \
        [BenchmarkTensors.make(ctx, op_type) for _ in range(arg_pool_size)]
    for bt in bench_tensors:
        bt.sanity_check()

    # BenchmarkTensors -> Dict (kwargs)
    kwargs_list = [
        bt.bench_fn_kwargs(op_type, add_inputs=expand_fn_add_inputs)
        for bt in bench_tensors
    ]

    # Merge into a single kwargs and quality arguments as ArgPool
    kwargs = {k: ArgPool([]) for k in kwargs_list[0]}
    for _kwargs in kwargs_list:
        for k, v in _kwargs.items():
            kwargs[k].values.append(v)

    describe_args = (f"add_inputs={expand_fn_add_inputs}"
                     if expand_fn_add_inputs is not None else "")
    description = (
        f"{op_type.name}({describe_args}) ({bench_tensors[0].io_types()})")
    cuda_graph_params = CudaGraphBenchParams(
        num_ops_in_cuda_graph=arg_pool_size) if with_cuda_graph else None
    with Bench(cuda_graph_params,
               ctx.bench_label(), ctx.bench_sublabel(op_type), description,
               op_type.bench_fn(), **kwargs) as bench:
        return bench.run()


def bench_torch_mm(ctx: BenchmarkContext,
                   arg_pool_size: int,
                   op_type: OpType,
                   with_cuda_graph: bool = False) -> TMeasurement:
    """
    Benchmark basic torch.mm as a roofline.
    input op_type is used in determining the m, k, n dimensions for the matmul.
    """

    batch_size, hidden_size, lora_rank, seq_length, dtype = (ctx.batch_size,
                                                             ctx.hidden_size,
                                                             ctx.lora_rank,
                                                             ctx.seq_length,
                                                             ctx.dtype)

    m, k, n = op_type.mkn(batch_size, seq_length, hidden_size, lora_rank)
    if op_type.is_expand_slice_fn():
        # For a fairer comparison.
        n = n * ctx.num_slices

    # Get matmul input and output tensors for A x B = C
    As, Bs, Cs = [], [], []
    for _ in range(arg_pool_size):
        As.append(torch.rand((m, k), dtype=dtype).to("cuda"))
        Bs.append(torch.rand((n, k), dtype=dtype).to("cuda").t())
        Cs.append(torch.rand((m, n), dtype=dtype).to("cuda"))

    # Make torch.mm kwargs
    mm_kwargs = {'input': ArgPool(As), 'mat2': ArgPool(Bs), 'out': ArgPool(Cs)}

    description = (f"torch.mm({dtype_to_str(dtype)}"
                   f"x{dtype_to_str(dtype)}"
                   f"=>{dtype_to_str(dtype)})")
    cuda_graph_params = CudaGraphBenchParams(
        num_ops_in_cuda_graph=arg_pool_size) if with_cuda_graph else None
    with Bench(cuda_graph_params, ctx.bench_label(),
               ctx.bench_sublabel(op_type), description, torch.mm,
               **mm_kwargs) as bench:
        return bench.run()


# runner
def print_timers(timers: List[TMeasurement]):
    compare = TBenchmark.Compare(timers)
    compare.print()


def run(args: argparse.Namespace, bench_ctxs: List[BenchmarkContext]):

    timers = []
    for bench_ctx in bench_ctxs:
        for seq_len in args.seq_lengths:
            bench_ops: List[OpType] = []
            if seq_len == 1:
                # bench all decode ops
                bench_ops = [op for op in args.op_types if op.is_decode_op()]
            else:
                # bench all prefill ops
                bench_ops = [op for op in args.op_types if op.is_prefill_op()]

            seq_len_timers = []
            for bench_op in bench_ops:
                for num_slices in bench_op.num_slices():
                    _ctx = bench_ctx.with_seq_length(seq_len).with_num_slices(
                        num_slices)
                    # Benchmark torch.mm as a roofline
                    seq_len_timers.append(
                        bench_torch_mm(_ctx, args.arg_pool_size, bench_op,
                                       args.with_cuda_graph))

                    # Benchmark bench_op
                    expand_fn_add_inputs = [
                        None
                    ] if bench_op.is_shrink_fn() else args.expand_fn_add_inputs
                    for add_input_arg in expand_fn_add_inputs:
                        seq_len_timers.append(
                            bench_optype(_ctx, args.arg_pool_size, bench_op,
                                         args.with_cuda_graph, add_input_arg))

            print_timers(seq_len_timers)
            timers.extend(seq_len_timers)

    # Result stdout dump
    print("== All Results ====")
    print_timers(timers)

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


def as_benchmark_contexts(hidden_sizes: List[int], lora_ranks: List[int],
                          args: argparse.Namespace) -> List[BenchmarkContext]:

    ctxs: List[BenchmarkContext] = []
    for batch_size, hidden_size, lora_rank, num_loras, sort_by_lora_id in product(  # noqa
            args.batch_sizes, list(hidden_sizes), lora_ranks, args.num_loras,
            args.sort_by_lora_id):
        ctxs.append(
            BenchmarkContext(
                batch_size=batch_size,
                hidden_size=hidden_size,
                lora_rank=lora_rank,
                num_loras=num_loras,
                num_active_loras=args.num_active_loras
                if args.num_active_loras else num_loras,
                # To be filled based on the OpType to benchmark
                seq_length=None,
                sort_by_lora_id=sort_by_lora_id,
                dtype=args.dtype,
                # To be filled based on the OpType to benchmark
                num_slices=None))

    return ctxs


def run_list_bench(args: argparse.Namespace):
    print(args)

    print("List bench :\n"
          f"  Hidden Sizes {args.hidden_sizes}"
          f"  LoRA Ranks {args.lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: List[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=args.hidden_sizes, lora_ranks=args.lora_ranks, args=args)

    run(args, bench_contexts)


def run_range_bench(args: argparse.Namespace):
    print(args)

    hidden_sizes = list(
        range(args.hidden_sizes_start, args.hidden_sizes_end + 1,
              args.hidden_sizes_increment))
    lora_ranks = list(
        range(args.lora_ranks_start, args.lora_ranks_end + 1,
              args.lora_ranks_increment))

    print("Range bench :\n"
          f" Hidden Sizes {hidden_sizes}"
          f" LoRA Ranks {lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: List[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=hidden_sizes, lora_ranks=lora_ranks, args=args)

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
        hidden_sizes = hidden_sizes.union(
            hidden_sizes_from_model(model_name, tp_size))

    print("Model bench :\n"
          f" Hidden Sizes {hidden_sizes}"
          f" LoRA Ranks {args.lora_ranks}")

    # Get all benchmarking contexts
    bench_contexts: List[BenchmarkContext] = as_benchmark_contexts(
        hidden_sizes=hidden_sizes, lora_ranks=args.lora_ranks, args=args)

    run(args, bench_contexts)


if __name__ == '__main__':

    def to_torch_dtype(dt):
        if dt == "torch.float16":
            return torch.float16
        if dt == "torch.bfloat16":
            return torch.bfloat16
        raise ValueError("unsupported dtype")

    def get_bool(s: str) -> bool:
        return s.lower() in ['true', '1']

    def add_common_command_args(p: argparse.ArgumentParser):
        p.add_argument(
            "--dtype",
            type=to_torch_dtype,
            required=True,
            help="Available options are ['torch.float16', 'torch.bfloat16']")

        p.add_argument(
            "--arg-pool-size",
            type=int,
            default=32,
            help="Run profiles with a pool of input/output/meta tensors instead"
            "of simply reusing the same tensors for all runs")

        p.add_argument("--with-cuda-graph",
                       action="store_true",
                       help="when set profiling is done using cudagraph")
        p.add_argument("--num-loras",
                       nargs="+",
                       type=int,
                       default=DEFAULT_NUM_LORAS)
        p.add_argument("--num-active-loras",
                       type=int,
                       default=None,
                       help="Active LoRAs. When None, all LoRAs are active")
        p.add_argument("--sort-by-lora-id",
                       nargs="+",
                       type=get_bool,
                       default=DEFAULT_SORT_BY_LORA_IDS)
        p.add_argument("--op-types",
                       nargs="+",
                       type=OpType.from_str,
                       default=list(OpType))
        p.add_argument('--seq-lengths',
                       nargs="+",
                       type=int,
                       default=DEFAULT_SEQ_LENGTHS)
        p.add_argument("--batch-sizes",
                       nargs="+",
                       type=int,
                       default=DEFAULT_BATCH_SIZES)
        p.add_argument("--expand-fn-add-inputs",
                       nargs="+",
                       type=get_bool,
                       default=DEFAULT_EXPAND_FN_ADD_INPUTS)
        p.add_argument('-o', '--output-directory', type=str)

    parser = FlexibleArgumentParser(
        description="""
Benchmark LoRA kernels:

    list_bench example:
        python3 benchmarks/kernels/benchmark_lora.py list_bench --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16 --hidden-sizes 2048 --lora-ranks 16 --num-loras 1 4 --op-types bgmv_shrink bgmv_expand sgmv_shrink sgmv_expand sgmv_expand_slice bgmv_expand_slice --seq-lengths 1 16 --sort-by-lora-id 1 --with-cuda-graph

    model_bench example:
        python3 benchmarks/kernels/benchmark_lora.py model_bench --models meta-llama/Llama-3-8b  --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16  --lora-ranks 16 --num-loras 1 4 --op-types bgmv_shrink bgmv_expand sgmv_shrink sgmv_expand sgmv_expand_slice bgmv_expand_slice --seq-lengths 1 16 --sort-by-lora-id 1 --with-cuda-graph 

    range_bench example:
        python3 benchmarks/kernels/benchmark_lora.py range_bench  --arg-pool-size 32 --batch-sizes 1 16 32 --dtype torch.float16   --num-loras 1 4 --op-types bgmv_shrink bgmv_expand sgmv_shrink sgmv_expand sgmv_expand_slice bgmv_expand_slice --seq-lengths 1 16 --sort-by-lora-id 1 --with-cuda-graph --hidden-sizes-start 1024 --hidden-sizes-end 4096 --hidden-sizes-increment 1024 --lora-ranks-start 8 --lora-ranks-end 24 --lora-ranks-increment 8 
            """,  # noqa: E501
        formatter_class=argparse.RawTextHelpFormatter)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    list_parser = subparsers.add_parser("list_bench")
    list_parser.add_argument("--hidden-sizes",
                             nargs="+",
                             type=int,
                             default=DEFAULT_HIDDEN_SIZES)
    list_parser.add_argument("--lora-ranks",
                             nargs="+",
                             type=int,
                             default=DEFAULT_LORA_RANKS)
    add_common_command_args(list_parser)
    list_parser.set_defaults(func=run_list_bench)

    range_parser = subparsers.add_parser("range_bench")
    range_parser.add_argument("--hidden-sizes-start", type=int, required=True)
    range_parser.add_argument("--hidden-sizes-end", type=int, required=True)
    range_parser.add_argument("--hidden-sizes-increment",
                              type=int,
                              required=True)
    range_parser.add_argument("--lora-ranks-start", type=int, required=True)
    range_parser.add_argument("--lora-ranks-end", type=int, required=True)
    range_parser.add_argument("--lora-ranks-increment",
                              type=int,
                              required=True)
    add_common_command_args(range_parser)
    range_parser.set_defaults(func=run_range_bench)

    model_parser = subparsers.add_parser("model_bench")
    model_parser.add_argument("--models",
                              nargs="+",
                              type=str,
                              default=DEFAULT_MODELS,
                              choices=WEIGHT_SHAPES.keys())
    model_parser.add_argument("--tp-sizes",
                              nargs="+",
                              type=int,
                              default=DEFAULT_TP_SIZES)
    model_parser.add_argument("--lora-ranks",
                              nargs="+",
                              type=int,
                              default=DEFAULT_LORA_RANKS)
    add_common_command_args(model_parser)
    model_parser.set_defaults(func=run_model_bench)

    args = parser.parse_args()
    args.func(args)
