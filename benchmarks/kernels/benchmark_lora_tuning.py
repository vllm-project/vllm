# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse
import csv
import json
import os
from itertools import product
from pathlib import Path
from typing import Any

import benchmark_lora as _benchmark_lora_mod
import torch
from benchmark_lora import BenchmarkContext, OpType, bench_optype, dtype_to_str

from vllm.lora.ops.triton_ops import (  # noqa: F401
    fused_moe_lora_expand,
    fused_moe_lora_shrink,
    lora_expand,
    lora_shrink,
)  # noqa: F401
from vllm.lora.ops.triton_ops import (
    fused_moe_lora_op as _fused_moe_lora_mod,
)
from vllm.lora.ops.triton_ops import (
    lora_expand_op as _lora_expand_mod,
)
from vllm.lora.ops.triton_ops import (
    lora_shrink_op as _lora_shrink_mod,
)
from vllm.lora.ops.triton_ops import utils as _lora_utils_mod
from vllm.triton_utils import HAS_TRITON, triton

try:
    from transformers import AutoConfig

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - optional dependency
    AutoConfig = None  # type: ignore[assignment]
    _HAS_TRANSFORMERS = False

try:
    from safetensors.torch import load_file as _load_safetensors
except Exception:  # pragma: no cover - optional dependency
    _load_safetensors = None


_ORIGINAL_GET_LORA_OP_CONFIGS = _lora_utils_mod.get_lora_op_configs
_CURRENT_LORA_KERNEL_CONFIG: dict[str, Any] | None = None
_PATCHED = False


def _find_first_key(
    dct: dict[str, Any], candidates: list[str]
) -> tuple[Any | None, str | None]:
    for key in candidates:
        if key in dct:
            return dct[key], key
    return None, None


def _infer_lora_rank_from_config(lora_dir: str) -> tuple[int | None, str]:
    for fname in ("adapter_config.json", "adapter_config.bin", "config.json"):
        path = os.path.join(lora_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue

        val, key = _find_first_key(cfg, ["r", "lora_r", "lora_rank"])
        if isinstance(val, int):
            return val, f"{fname}:{key}"
    return None, ""


def _infer_lora_rank_from_safetensors(lora_dir: str) -> tuple[int | None, str]:
    if _load_safetensors is None:
        return None, ""

    safepath: str | None = None
    for fname in ("adapter_model.safetensors", "lora.safetensors"):
        path = os.path.join(lora_dir, fname)
        if os.path.isfile(path):
            safepath = path
            break

    if safepath is None:
        for fname in os.listdir(lora_dir):
            if fname.endswith(".safetensors"):
                safepath = os.path.join(lora_dir, fname)
                break

    if safepath is None:
        return None, ""

    try:
        tensors = _load_safetensors(safepath)
    except Exception:
        return None, ""

    preferred_keys = [k for k in tensors if "lora_" in k.lower()]
    keys_to_check = preferred_keys or list(tensors.keys())

    for name in keys_to_check:
        w = tensors[name]
        if w.ndim >= 2:
            rank = int(min(w.shape))
            return rank, f"{os.path.basename(safepath)}:{name}"

    return None, ""


def _maybe_auto_infer_dims(args: argparse.Namespace, is_fused_moe: bool) -> None:
    if is_fused_moe and getattr(args, "model_path", None) and _HAS_TRANSFORMERS:
        cfg = AutoConfig.from_pretrained(args.model_path)
        cfg_dict = cfg.to_dict()

        hidden_size, _ = _find_first_key(cfg_dict, ["hidden_size", "n_embd", "d_model"])
        num_experts, _ = _find_first_key(
            cfg_dict,
            [
                "num_experts",
                "moe_num_experts",
                "num_local_experts",
                "n_routed_experts",
            ],
        )
        moe_inter_size, _ = _find_first_key(
            cfg_dict,
            ["moe_intermediate_size", "ffn_hidden_size", "intermediate_size"],
        )
        top_k, _ = _find_first_key(
            cfg_dict,
            [
                "num_experts_per_tok",
                "num_experts_per_token",
                "moe_top_k",
                "top_k",
            ],
        )

        if args.hidden_size is None and hidden_size is not None:
            args.hidden_size = int(hidden_size)
        if num_experts is not None:
            args.num_experts = int(num_experts)
        if moe_inter_size is not None and args.moe_intermediate_size is None:
            args.moe_intermediate_size = int(moe_inter_size)
        if top_k is not None:
            args.top_k_num = int(top_k)

    if getattr(args, "lora_path", None):
        lora_dir = args.lora_path
        if os.path.isdir(lora_dir):
            rank_cfg, _ = _infer_lora_rank_from_config(lora_dir)
            rank_safe, _ = _infer_lora_rank_from_safetensors(lora_dir)
            rank = rank_cfg if rank_cfg is not None else rank_safe
            if rank is not None and args.lora_rank is None:
                args.lora_rank = int(rank)


def _patched_get_lora_op_configs(
    op_type: str,
    max_loras: int,
    batch: int,
    hidden_size: int,
    rank: int,
    num_slices: int,
    add_inputs: bool | None = None,
    moe_intermediate_size: int | None = None,
) -> dict[str, Any]:
    if _CURRENT_LORA_KERNEL_CONFIG is not None:
        cfg = _CURRENT_LORA_KERNEL_CONFIG
        # For fused_moe_lora ops, benchmark_lora expects
        # BLOCK_SIZE_*/GROUP_SIZE_M/NUM_WARPS/NUM_STAGES/SPLIT_K style keys,
        # while our search space uses lower-case names. Convert here.
        if op_type.startswith("fused_moe_lora_"):
            return {
                "BLOCK_SIZE_M": cfg["block_m"],
                "BLOCK_SIZE_N": cfg["block_n"],
                "BLOCK_SIZE_K": cfg["block_k"],
                "GROUP_SIZE_M": cfg.get("group_size_m", 8),
                "NUM_WARPS": cfg["num_warps"],
                "NUM_STAGES": cfg["num_stages"],
                "SPLIT_K": cfg.get("split_k", 1),
            }
        # For regular shrink/expand, Triton kernels consume lower-case keys directly.
        return cfg

    return _ORIGINAL_GET_LORA_OP_CONFIGS(
        op_type,
        max_loras,
        batch,
        hidden_size,
        rank,
        num_slices,
        add_inputs,
        moe_intermediate_size,
    )


def _ensure_patched() -> None:
    global _PATCHED
    if _PATCHED:
        return

    # Patch utils, benchmarking module, and Triton op modules to use the patched
    # get_lora_op_configs so that the benchmark uses our current candidate
    # kernel configuration.
    for mod in (
        _lora_utils_mod,
        _lora_shrink_mod,
        _lora_expand_mod,
        _fused_moe_lora_mod,
        _benchmark_lora_mod,
    ):
        if hasattr(mod, "get_lora_op_configs"):
            mod.get_lora_op_configs = _patched_get_lora_op_configs

    _PATCHED = True


def _build_search_space() -> list[dict[str, Any]]:
    # Search space as suggested in README_TUNING.md.
    block_m_range = [16, 32, 64, 128, 256]
    block_n_range = [32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256]
    num_warps_range = [4, 8]
    num_stage_range = [2, 3, 4, 5]
    num_ctas_range = [1]
    split_k_range = [4, 8, 16, 32, 64]

    search_space: list[dict[str, Any]] = []
    for (
        block_m,
        block_n,
        block_k,
        num_warps,
        num_stages,
        num_ctas,
        split_k,
    ) in product(
        block_m_range,
        block_n_range,
        block_k_range,
        num_warps_range,
        num_stage_range,
        num_ctas_range,
        split_k_range,
    ):
        cfg: dict[str, Any] = {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "num_warps": num_warps,
            "num_stages": num_stages,
            "num_ctas": num_ctas,
            "split_k": split_k,
            # Fields used by shrink / fused-moe kernels; safe for expand too.
            "group_size_m": 8,
            "max_nreg": None,
        }
        search_space.append(cfg)

    return search_space


def _to_torch_dtype(dt: str) -> torch.dtype:
    if dt == "torch.float16":
        return torch.float16
    if dt == "torch.bfloat16":
        return torch.bfloat16
    if dt == "torch.float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype string: {dt}")


def _benchmark_single_config(
    cfg: dict[str, Any],
    ctx: BenchmarkContext,
    op_type: OpType,
    arg_pool_size: int,
    cuda_graph_nops: int | None,
    expand_fn_add_inputs: bool | None,
    test_correctness: bool,
) -> float | None:
    global _CURRENT_LORA_KERNEL_CONFIG

    _CURRENT_LORA_KERNEL_CONFIG = cfg

    try:
        timer = bench_optype(
            ctx,
            arg_pool_size,
            op_type,
            cuda_graph_nops,
            expand_fn_add_inputs,
            test_correctness,
        )
    except triton.runtime.autotuner.OutOfResources:
        return None

    # torch.utils.benchmark.Measurement.median is in seconds; convert to ms.
    return float(timer.median) * 1e3


def _tune_fused_moe_lora_ops(
    ctx_list: list[BenchmarkContext],
    args: argparse.Namespace,
    search_space: list[dict[str, Any]],
    dtype: torch.dtype,
    fieldnames: list[str],
) -> None:
    fused_ops = [
        OpType.FUSED_MOE_LORA_GATE_UP_SHRINK,
        OpType.FUSED_MOE_LORA_GATE_UP_EXPAND,
        OpType.FUSED_MOE_LORA_DOWN_SHRINK,
        OpType.FUSED_MOE_LORA_DOWN_EXPAND,
    ]

    block_m_values = sorted({cfg["block_m"] for cfg in search_space})

    # Write CSV rows to os.devnull so no CSV file is created.
    with open(os.devnull, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for ctx in ctx_list:
            m_val = ctx.batch_size * ctx.seq_length
            print(
                f"Tuning fused_moe_lora M={m_val} over "
                f"{len(block_m_values)} BLOCK_SIZE_M values "
                f"and {len(search_space)} kernel configs"
            )

            best_joint_time_ms: float = float("inf")
            best_cfg_per_op: dict[OpType, dict[str, Any]] = {}

            for block_m in block_m_values:
                block_cfgs = [
                    cfg for cfg in search_space if cfg.get("block_m") == block_m
                ]
                if not block_cfgs:
                    continue

                per_op_best_cfg: dict[OpType, dict[str, Any]] = {}
                per_op_best_time_ms: dict[OpType, float] = {}
                valid_for_all_ops = True

                total_block_cfgs = len(block_cfgs)
                log_interval = max(1, total_block_cfgs // 10)

                for op in fused_ops:
                    best_time_ms: float = float("inf")
                    best_cfg: dict[str, Any] | None = None

                    for idx, cfg in enumerate(block_cfgs, start=1):
                        median_time_ms = _benchmark_single_config(
                            cfg=cfg,
                            ctx=ctx,
                            op_type=op,
                            arg_pool_size=args.arg_pool_size,
                            cuda_graph_nops=args.cuda_graph_nops,
                            expand_fn_add_inputs=None,
                            test_correctness=args.test_correctness,
                        )

                        status = (
                            "ok" if median_time_ms is not None else "out_of_resources"
                        )

                        row: dict[str, Any] = {
                            "op_type": op.name,
                            "dtype": dtype_to_str(dtype),
                            "batch_size": ctx.batch_size,
                            "seq_length": ctx.seq_length,
                            "hidden_size": ctx.hidden_size,
                            "lora_rank": ctx.lora_rank,
                            "num_loras": ctx.num_loras,
                            "num_active_loras": ctx.num_active_loras,
                            "num_slices": ctx.num_slices,
                            "median_time_ms": (
                                median_time_ms if median_time_ms is not None else ""
                            ),
                            "status": status,
                        }
                        for key in (
                            "block_m",
                            "block_n",
                            "block_k",
                            "num_warps",
                            "num_stages",
                            "num_ctas",
                            "split_k",
                            "group_size_m",
                            "max_nreg",
                        ):
                            row[key] = cfg.get(key)
                        writer.writerow(row)

                        if median_time_ms is not None and median_time_ms < best_time_ms:
                            best_time_ms = median_time_ms
                            best_cfg = dict(cfg)

                        if idx % log_interval == 0 or idx == total_block_cfgs:
                            if best_cfg is not None and best_time_ms != float("inf"):
                                print(
                                    f"[M={m_val}] BLOCK_M={block_m} "
                                    f"{op.name} [{idx}/{total_block_cfgs}] "
                                    f"current best median_time_ms={best_time_ms:.3f}"
                                )
                            else:
                                print(
                                    f"[M={m_val}] BLOCK_M={block_m} "
                                    f"{op.name} [{idx}/{total_block_cfgs}] "
                                    "benchmarking..."
                                )

                    if best_cfg is None:
                        valid_for_all_ops = False
                        break

                    per_op_best_cfg[op] = best_cfg
                    per_op_best_time_ms[op] = best_time_ms

                if not valid_for_all_ops:
                    continue

                joint_time_ms = sum(per_op_best_time_ms.values())
                if joint_time_ms < best_joint_time_ms:
                    best_joint_time_ms = joint_time_ms
                    best_cfg_per_op = {
                        op: dict(cfg) for op, cfg in per_op_best_cfg.items()
                    }

                print(
                    f"[M={m_val}] BLOCK_M={block_m} "
                    f"joint_time_ms={joint_time_ms:.3f}, "
                    f"best_joint_time_ms={best_joint_time_ms:.3f}"
                )

            if not best_cfg_per_op:
                print(
                    f"No valid fused_moe_lora configs found for M={m_val}; "
                    "skipping JSON output for this context."
                )
                continue

            if not args.save_json_dir:
                continue

            save_dir = Path(args.save_json_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            gpu_name = torch.cuda.get_device_name()
            gpu_name = gpu_name.replace(" ", "_").replace("-", "_")

            for op, best_cfg in best_cfg_per_op.items():
                if op in (
                    OpType.FUSED_MOE_LORA_GATE_UP_SHRINK,
                    OpType.FUSED_MOE_LORA_GATE_UP_EXPAND,
                ):
                    op_name = (
                        "fused_moe_lora_w13_shrink"
                        if op.is_fused_moe_lora_shrink_fn()
                        else "fused_moe_lora_w13_expand"
                    )
                elif op in (
                    OpType.FUSED_MOE_LORA_DOWN_SHRINK,
                    OpType.FUSED_MOE_LORA_DOWN_EXPAND,
                ):
                    op_name = (
                        "fused_moe_lora_w2_shrink"
                        if op.is_fused_moe_lora_shrink_fn()
                        else "fused_moe_lora_w2_expand"
                    )
                else:
                    # Should not happen for fused_moe tuning.
                    continue

                json_name = f"{gpu_name}_{op_name.upper()}.json"
                json_path = save_dir / json_name

                if json_path.exists() and not args.json_overwrite:
                    base = json_name[:-5] if json_name.endswith(".json") else json_name
                    idx = 1
                    while True:
                        candidate = save_dir / f"{base}_tuned_{idx}.json"
                        if not candidate.exists():
                            json_path = candidate
                            break
                        idx += 1
                    print(
                        f"JSON file {json_name} exists and "
                        "--json-overwrite is not set; "
                        f"writing tuned config to {json_path.name} instead."
                    )
                    config_data: dict[str, Any] = {}
                else:
                    if json_path.exists():
                        with json_path.open("r") as jf:
                            config_data = json.load(jf)
                    else:
                        config_data = {}

                max_loras_key = str(ctx.num_loras)
                num_slices_key = str(ctx.num_slices)

                # m, k, n follow the same convention as get_lora_op_configs.
                m_val = ctx.batch_size * ctx.seq_length
                is_shrink = op.is_shrink_fn()
                if is_shrink:
                    k_val = ctx.hidden_size
                    n_val = ctx.lora_rank
                else:
                    k_val = ctx.lora_rank
                    n_val = ctx.hidden_size

                m_key = str(m_val)
                k_key = str(k_val)
                n_key = str(n_val)

                config_data.setdefault(max_loras_key, {})
                config_data[max_loras_key].setdefault(num_slices_key, {})
                config_data[max_loras_key][num_slices_key].setdefault(m_key, {})
                config_data[max_loras_key][num_slices_key][m_key].setdefault(k_key, {})

                if (
                    hasattr(args, "moe_intermediate_size")
                    and args.moe_intermediate_size is not None
                ):
                    i_key = str(args.moe_intermediate_size)
                    config_data[max_loras_key][num_slices_key][m_key][k_key].setdefault(
                        n_key, {}
                    )
                    config_data[max_loras_key][num_slices_key][m_key][k_key][n_key][
                        i_key
                    ] = best_cfg
                else:
                    config_data[max_loras_key][num_slices_key][m_key][k_key][n_key] = (
                        best_cfg
                    )

                with json_path.open("w") as jf:
                    json.dump(config_data, jf)
                print(f"Tuned JSON config saved to {json_path}")


def main(args: argparse.Namespace) -> None:
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available; LoRA Triton kernels cannot run.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark requires a GPU.")

    _ensure_patched()

    raw_op_type = args.op_type

    # User-level fused MoE entry point: "fused_moe_lora" triggers joint tuning
    # of all four fused MoE LoRA kernels (gate_up/down × shrink/expand).
    if raw_op_type == "fused_moe_lora":
        op_type: OpType | None = None
        is_fused_moe = True
    else:
        op_type = OpType.from_str(raw_op_type)
        if op_type not in (
            OpType.LORA_SHRINK,
            OpType.LORA_EXPAND,
            OpType.FUSED_MOE_LORA_GATE_UP_SHRINK,
            OpType.FUSED_MOE_LORA_GATE_UP_EXPAND,
            OpType.FUSED_MOE_LORA_DOWN_SHRINK,
            OpType.FUSED_MOE_LORA_DOWN_EXPAND,
        ):
            raise ValueError("Unsupported op_type for tuning.")

        is_fused_moe = op_type in (
            OpType.FUSED_MOE_LORA_GATE_UP_SHRINK,
            OpType.FUSED_MOE_LORA_GATE_UP_EXPAND,
            OpType.FUSED_MOE_LORA_DOWN_SHRINK,
            OpType.FUSED_MOE_LORA_DOWN_EXPAND,
        )

    dtype = _to_torch_dtype(args.dtype)

    num_active_loras = (
        args.num_active_loras if args.num_active_loras is not None else args.num_loras
    )

    _maybe_auto_infer_dims(args, is_fused_moe)

    if args.hidden_size is None:
        raise ValueError(
            "--hidden-size must be set explicitly or inferable from --model-path"
        )
    if args.lora_rank is None:
        raise ValueError(
            "--lora-rank must be set explicitly or inferable from --lora-path"
        )

    # For fused_moe_lora JSON configs, moe_intermediate_size is part of the
    # key hierarchy used at runtime (see get_lora_op_configs). If we save
    # JSON without this dimension, later loading will fail. Enforce that
    # users provide it when tuning fused_moe_lora ops with JSON output.
    if is_fused_moe and args.save_json_dir and args.moe_intermediate_size is None:
        raise ValueError(
            "For fused_moe_lora op types, --moe-intermediate-size must be set "
            "when using --save-json-dir."
        )

    # Build a list of BenchmarkContext objects to support tuning multiple
    # M values (num_tokens) in a single run. When --m-values is provided,
    # each M is mapped to (batch_size=M, seq_length=1). Otherwise, a single
    # context is created from --batch-size and --seq-length.
    ctx_list: list[BenchmarkContext] = []
    if getattr(args, "m_values", None):
        for m in args.m_values:
            ctx_list.append(
                BenchmarkContext(
                    batch_size=m,
                    hidden_size=args.hidden_size,
                    num_loras=args.num_loras,
                    num_active_loras=num_active_loras,
                    lora_rank=args.lora_rank,
                    sort_by_lora_id=bool(args.sort_by_lora_id),
                    dtype=dtype,
                    seq_length=1,
                    num_experts=args.num_experts if is_fused_moe else None,
                    top_k_num=args.top_k_num if is_fused_moe else None,
                    num_slices=args.num_slices,
                )
            )
    else:
        ctx_list.append(
            BenchmarkContext(
                batch_size=args.batch_size,
                hidden_size=args.hidden_size,
                num_loras=args.num_loras,
                num_active_loras=num_active_loras,
                lora_rank=args.lora_rank,
                sort_by_lora_id=bool(args.sort_by_lora_id),
                dtype=dtype,
                seq_length=args.seq_length,
                num_experts=args.num_experts if is_fused_moe else None,
                top_k_num=args.top_k_num if is_fused_moe else None,
                num_slices=args.num_slices,
            )
        )

    expand_fn_add_inputs: bool | None
    # For shrink or any fused_moe_lora op, benchmark_lora expects add_inputs to be None.
    # When using the user-level "fused_moe_lora" op_type, op_type is None and
    # is_fused_moe is True, so we also force expand_fn_add_inputs=None.
    if op_type is None or op_type.is_shrink_fn() or op_type.is_fused_moe_lora_fn():
        expand_fn_add_inputs = None
    else:
        expand_fn_add_inputs = bool(args.expand_add_inputs)

    search_space = _build_search_space()
    if args.max_configs is not None and args.max_configs > 0:
        search_space = search_space[: args.max_configs]

    fieldnames = [
        "op_type",
        "dtype",
        "batch_size",
        "seq_length",
        "hidden_size",
        "lora_rank",
        "num_loras",
        "num_active_loras",
        "num_slices",
        "block_m",
        "block_n",
        "block_k",
        "num_warps",
        "num_stages",
        "num_ctas",
        "split_k",
        "group_size_m",
        "max_nreg",
        "median_time_ms",
        "status",
    ]

    # For fused MoE LoRA ops, tune the four kernels jointly so that they share
    # the same BLOCK_SIZE_M, using the sum of their median times as the joint
    # objective. This also guarantees BLOCK_SIZE_M matches the value used in
    # moe_lora_align_block_size.
    if is_fused_moe:
        _tune_fused_moe_lora_ops(ctx_list, args, search_space, dtype, fieldnames)
        return

    # Write CSV rows to os.devnull so no CSV file is created.
    with open(os.devnull, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Tune over each BenchmarkContext (potentially different M values)
        for ctx in ctx_list:
            total_configs = len(search_space)
            m_val = ctx.batch_size * ctx.seq_length
            print(f"Tuning M={m_val} over {total_configs} kernel configs")
            log_interval = max(1, total_configs // 10)

            best_cfg: dict[str, Any] | None = None
            best_time_ms: float = float("inf")

            for idx, cfg in enumerate(search_space, start=1):
                median_time_ms = _benchmark_single_config(
                    cfg=cfg,
                    ctx=ctx,
                    op_type=op_type,
                    arg_pool_size=args.arg_pool_size,
                    cuda_graph_nops=args.cuda_graph_nops,
                    expand_fn_add_inputs=expand_fn_add_inputs,
                    test_correctness=args.test_correctness,
                )

                status = "ok" if median_time_ms is not None else "out_of_resources"
                if median_time_ms is not None and median_time_ms < best_time_ms:
                    best_time_ms = median_time_ms
                    best_cfg = dict(cfg)

                row: dict[str, Any] = {
                    "op_type": op_type.name,
                    "dtype": dtype_to_str(dtype),
                    "batch_size": ctx.batch_size,
                    "seq_length": ctx.seq_length,
                    "hidden_size": ctx.hidden_size,
                    "lora_rank": ctx.lora_rank,
                    "num_loras": ctx.num_loras,
                    "num_active_loras": ctx.num_active_loras,
                    "num_slices": ctx.num_slices,
                    "median_time_ms": median_time_ms
                    if median_time_ms is not None
                    else "",
                    "status": status,
                }

                for key in (
                    "block_m",
                    "block_n",
                    "block_k",
                    "num_warps",
                    "num_stages",
                    "num_ctas",
                    "split_k",
                    "group_size_m",
                    "max_nreg",
                ):
                    row[key] = cfg.get(key)

                writer.writerow(row)

                if idx % log_interval == 0 or idx == total_configs:
                    if best_cfg is not None and best_time_ms != float("inf"):
                        print(
                            f"[M={m_val}] [{idx}/{total_configs}] "
                            f"current best median_time_ms={best_time_ms:.3f}"
                        )
                    else:
                        print(f"[M={m_val}] [{idx}/{total_configs}] benchmarking...")

            # Optionally save best configuration for this ctx into JSON.
            if args.save_json_dir and best_cfg is not None:
                save_dir = Path(args.save_json_dir)
                save_dir.mkdir(parents=True, exist_ok=True)

                gpu_name = torch.cuda.get_device_name()
                gpu_name = gpu_name.replace(" ", "_").replace("-", "_")

                # Determine op name for JSON file
                # NOTE: fused_moe_lora_* uses separate *_shrink / *_expand
                # names and cannot rely on is_shrink_fn() (which only returns
                # True for LORA_SHRINK), otherwise shrink/expand would write
                # to the same file.
                if op_type in (
                    OpType.FUSED_MOE_LORA_GATE_UP_SHRINK,
                    OpType.FUSED_MOE_LORA_GATE_UP_EXPAND,
                ):
                    op_name = (
                        "fused_moe_lora_w13_shrink"
                        if op_type.is_fused_moe_lora_shrink_fn()
                        else "fused_moe_lora_w13_expand"
                    )
                elif op_type in (
                    OpType.FUSED_MOE_LORA_DOWN_SHRINK,
                    OpType.FUSED_MOE_LORA_DOWN_EXPAND,
                ):
                    op_name = (
                        "fused_moe_lora_w2_shrink"
                        if op_type.is_fused_moe_lora_shrink_fn()
                        else "fused_moe_lora_w2_expand"
                    )
                else:
                    op_name = "shrink" if op_type.is_shrink_fn() else "expand"

                if op_name == "expand":
                    add_inputs = bool(args.expand_add_inputs)
                    json_name = (
                        f"{gpu_name}_{op_name.upper()}_{str(add_inputs).upper()}.json"
                    )
                else:
                    json_name = f"{gpu_name}_{op_name.upper()}.json"

                json_path = save_dir / json_name

                if json_path.exists() and not args.json_overwrite:
                    base = json_name[:-5] if json_name.endswith(".json") else json_name
                    idx = 1
                    while True:
                        candidate = save_dir / f"{base}_tuned_{idx}.json"
                        if not candidate.exists():
                            json_path = candidate
                            break
                        idx += 1
                    print(
                        f"JSON file {json_name} exists and "
                        "--json-overwrite is not set; "
                        f"writing tuned config to {json_path.name} instead."
                    )
                    config_data = {}
                else:
                    if json_path.exists():
                        with json_path.open("r") as jf:
                            config_data = json.load(jf)
                    else:
                        config_data = {}

                max_loras_key = str(ctx.num_loras)
                num_slices_key = str(ctx.num_slices)

                # m, k, n follow the same convention as get_lora_op_configs.
                m_val = ctx.batch_size * ctx.seq_length
                is_shrink = op_type.is_shrink_fn()
                if is_shrink:
                    k_val = ctx.hidden_size
                    n_val = ctx.lora_rank
                else:
                    k_val = ctx.lora_rank
                    n_val = ctx.hidden_size

                m_key = str(m_val)
                k_key = str(k_val)
                n_key = str(n_val)

                config_data.setdefault(max_loras_key, {})
                config_data[max_loras_key].setdefault(num_slices_key, {})
                config_data[max_loras_key][num_slices_key].setdefault(m_key, {})
                config_data[max_loras_key][num_slices_key][m_key].setdefault(k_key, {})

                # For fused_moe_lora, add moe_intermediate_size dimension if needed
                if (
                    is_fused_moe
                    and hasattr(args, "moe_intermediate_size")
                    and args.moe_intermediate_size
                ):
                    i_key = str(args.moe_intermediate_size)
                    config_data[max_loras_key][num_slices_key][m_key][k_key].setdefault(
                        n_key, {}
                    )
                    config_data[max_loras_key][num_slices_key][m_key][k_key][n_key][
                        i_key
                    ] = best_cfg
                else:
                    config_data[max_loras_key][num_slices_key][m_key][k_key][n_key] = (
                        best_cfg
                    )

                with json_path.open("w") as jf:
                    json.dump(config_data, jf)
                print(f"Tuned JSON config saved to {json_path}")

    # CSV results are no longer saved; tuned configs are written to JSON only.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Grid-search LoRA Triton kernel configs over a predefined "
            "search space and save benchmark results to CSV."
        )
    )

    parser.add_argument(
        "--op-type",
        type=str,
        default="lora_shrink",
        choices=[
            "lora_shrink",
            "lora_expand",
            "fused_moe_lora",
            "fused_moe_lora_gate_up_shrink",
            "fused_moe_lora_gate_up_expand",
            "fused_moe_lora_down_shrink",
            "fused_moe_lora_down_expand",
        ],
        help="LoRA op type to benchmark.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="torch.float16",
        help="Data type string, e.g. 'torch.float16' or 'torch.bfloat16'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=True,
        help="Batch size used in the benchmark.",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        required=True,
        help="Sequence length used in the benchmark.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        required=False,
        default=None,
        help="Hidden size (K dimension for shrink).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        required=False,
        default=None,
        help="LoRA rank (N dimension for shrink).",
    )
    parser.add_argument(
        "--num-loras",
        type=int,
        required=True,
        help="Total number of LoRA adapters.",
    )
    parser.add_argument(
        "--num-active-loras",
        type=int,
        default=None,
        help="Number of active LoRAs. Defaults to num-loras.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=1,
        help="Number of slices for the LoRA kernel.",
    )
    parser.add_argument(
        "--sort-by-lora-id",
        type=int,
        default=1,
        help="Whether to sort by LoRA ID (1 or 0).",
    )
    parser.add_argument(
        "--arg-pool-size",
        type=int,
        default=32,
        help="Argument pool size reused from benchmark_lora.",
    )
    parser.add_argument(
        "--cuda-graph-nops",
        type=int,
        default=None,
        help="Number of ops inside a CUDA graph; forwarded to benchmark_lora.",
    )
    parser.add_argument(
        "--expand-add-inputs",
        dest="expand_add_inputs",
        type=int,
        default=0,
        help="For lora_expand, whether to add inputs (1 or 0).",
    )
    parser.add_argument(
        "--test-correctness",
        action="store_true",
        help="Whether to run correctness checks in benchmark_lora.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="lora_tuning_results.csv",
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--max-configs",
        type=int,
        default=None,
        help="Limit the number of kernel configs to benchmark (for quick demos).",
    )
    parser.add_argument(
        "--save-json-dir",
        type=str,
        default=None,
        help=(
            "Directory to save tuned JSON config files. "
            "When set, the best configuration is written to a JSON file "
            "compatible with get_lora_op_configs."
        ),
    )
    parser.add_argument(
        "--json-overwrite",
        action="store_true",
        help=(
            "When used with --save-json-dir, overwrite an existing JSON file "
            "with the tuned config. By default, a new file name is generated "
            "if the target file already exists."
        ),
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Number of experts for fused MoE LoRA.",
    )
    parser.add_argument(
        "--top-k-num",
        type=int,
        default=2,
        help="Top-k value for fused MoE LoRA.",
    )
    parser.add_argument(
        "--moe-intermediate-size",
        type=int,
        default=None,
        help="MoE intermediate size (optional, for JSON config hierarchy).",
    )
    parser.add_argument(
        "--m-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional list of M values (num_tokens = batch_size * seq_length) "
            "to tune in a single run. When set, each M is mapped to a separate "
            "BenchmarkContext with batch_size=M and seq_length=1, and results "
            "for all Ms are written into the same CSV/JSON."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Optional base model path or Hugging Face repo id. "
            "When tuning fused_moe_lora ops and transformers is installed, "
            "hidden_size/num_experts/moe_intermediate_size/top_k_num can be "
            "auto-inferred from the model config if not explicitly set."
        ),
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default=None,
        help=(
            "Optional LoRA weights directory. When provided and --lora-rank "
            "is not set, the script will try to infer lora_rank from "
            "adapter_config.json or safetensors weights."
        ),
    )

    main(parser.parse_args())
