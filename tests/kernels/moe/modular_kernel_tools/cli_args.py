# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

from .common import Config
from .mk_objects import (
    MK_ALL_PREPARE_FINALIZE_TYPES,
    MK_FUSED_EXPERT_TYPES,
    MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES,
)


def make_config_arg_parser(description: str):
    def to_pf_class_type(s: str) -> mk.FusedMoEPrepareAndFinalize:
        for pf in MK_ALL_PREPARE_FINALIZE_TYPES:
            if pf.__name__ == s:
                return pf
        raise ValueError(f"Cannot find a PrepareFinalize type that matches {s}")

    def to_experts_class_type(s: str) -> mk.FusedMoEModularExperts:
        for fe in MK_FUSED_EXPERT_TYPES:
            if fe.__name__ == s:
                return fe
        raise ValueError(f"Cannot find a FusedExperts type that matches {s}")

    def to_quant_torch_dtype(s: str) -> torch.dtype:
        if s == "torch.float8_e4m3fn":
            return torch.float8_e4m3fn
        raise ValueError(f"Unsupported quant type {s}")

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Number of ranks that participate in all2all",
    )
    parser.add_argument(
        "--pf-type",
        type=to_pf_class_type,
        required=True,
        help=(
            "Choose a PrepareFinalize Type : "
            f"{[x.__name__ for x in MK_ALL_PREPARE_FINALIZE_TYPES]}"
        ),
    )
    parser.add_argument(
        "--experts-type",
        type=to_experts_class_type,
        required=True,
        help=(
            f"Choose a FusedExpert type : {[x.__name__ for x in MK_FUSED_EXPERT_TYPES]}"
        ),
    )
    parser.add_argument(
        "-m",
        nargs="+",
        type=int,
        default=[64],
        help="num tokens per rank",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=7168,
        help="hidden-size",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1024,
        help="N dimension of the first fused-moe matmul",
    )
    parser.add_argument(
        "--num-experts", type=int, default=32, help="Global num experts"
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[4, 1], help="num topk")
    parser.add_argument(
        "--fused-moe-chunk-size",
        type=int,
        help="Fused moe chunk size used for the non-batched fused experts impl.",
    )

    # Quant args
    parser.add_argument(
        "--quant-dtype", type=to_quant_torch_dtype, help="Quant datatype"
    )
    parser.add_argument(
        "--per-token-quantized-activations",
        action="store_true",
        help=("The input activations must be per-token quantized"),
    )
    parser.add_argument(
        "--per-channel-quantized-weights",
        action="store_true",
        help="The weights must be per-channel quantized.",
    )
    parser.add_argument(
        "--block-shape", nargs="+", type=int, help="Quantization block shape"
    )

    # Torch trace profile generation args
    parser.add_argument(
        "--torch-trace-dir-path",
        type=str,
        default=None,
        help="Get torch trace for single execution",
    )

    return parser


def _validate_args(args: argparse.Namespace):
    if args.quant_dtype is not None:
        assert args.quant_dtype == torch.float8_e4m3fn
        if args.block_shape is not None:
            assert len(args.block_shape) == 2, (
                f"block shape must have 2 elements. got {args.block_shape}"
            )

    if args.experts_type in MK_SINGLE_GPU_PREPARE_FINALIZE_TYPES:
        assert args.world_size == 1, "Single GPU objects need world size set to 1"

    if args.torch_trace_dir_path is not None:
        from pathlib import Path

        assert Path(args.torch_trace_dir_path).is_dir(), (
            f"Please create {args.torch_trace_dir_path}"
        )


def make_config(args: argparse.Namespace) -> Config:
    _validate_args(args)

    quant_config = None
    if args.quant_dtype is not None:
        quant_config = FusedMoEQuantConfig.make(
            quant_dtype=args.quant_dtype,
            per_act_token_quant=args.per_token_quantized_activations,
            per_out_ch_quant=args.per_channel_quantized_weights,
            block_shape=args.block_shape,
        )

    return Config(
        Ms=args.m,
        K=args.k,
        N=args.n,
        E=args.num_experts,
        topks=args.topk,
        dtype=torch.bfloat16,  # hard-code
        quant_config=quant_config,
        prepare_finalize_type=args.pf_type,
        fused_experts_type=args.experts_type,
        fused_moe_chunk_size=args.fused_moe_chunk_size,
        world_size=args.world_size,
        torch_trace_dir_path=args.torch_trace_dir_path,
    )
