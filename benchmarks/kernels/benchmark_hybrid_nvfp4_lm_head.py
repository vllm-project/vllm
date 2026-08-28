# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import median

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.model_executor.layers.hybrid_nvfp4_lm_head import (
    _global_scale,
    _select_candidate_tile,
    indexed_bf16_dot,
    select_lm_head_candidates,
)
from vllm.model_executor.layers.argmax_triton import (
    indexed_argmax_triton,
    reduce_global_argmax_triton,
)
from vllm.triton_utils import HAS_TRITON
from vllm.utils.flashinfer import (
    flashinfer_nvfp4_quantize_128x4,
    flashinfer_scaled_fp4_mm,
)

_EMBEDDING_WEIGHT_KEYS = (
    "lm_head.weight",
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)


@dataclass
class Nvfp4Weight:
    weight: torch.Tensor
    scale: torch.Tensor
    global_scale: torch.Tensor
    output_size: int


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value}"
        )
    return parsed


def _candidate_tile(value: str) -> int | None:
    if value == "auto":
        return None
    parsed = int(value)
    if parsed not in (1, 2, 4, 8):
        raise argparse.ArgumentTypeError(
            f"expected auto, 1, 2, 4, or 8, got {value}"
        )
    return parsed


def _resolve_weight(model_dir: Path) -> tuple[Path, str]:
    index_paths = sorted(model_dir.glob("*safetensors.index.json"))
    if len(index_paths) != 1:
        raise ValueError(
            f"expected one safetensors index in {model_dir}, got {index_paths}"
        )
    weight_map = json.loads(index_paths[0].read_text(encoding="utf-8"))["weight_map"]
    for key in _EMBEDDING_WEIGHT_KEYS:
        if key in weight_map:
            return model_dir / weight_map[key], key
    raise KeyError(f"none of {_EMBEDDING_WEIGHT_KEYS} is present in {index_paths[0]}")


def _load_weight_shard(
    model_dir: Path,
    *,
    tp_size: int,
    tp_rank: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, str]:
    shard_path, weight_key = _resolve_weight(model_dir)
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        weight_slice = handle.get_slice(weight_key)
        shape = weight_slice.get_shape()
        if len(shape) != 2:
            raise ValueError(f"lm-head weight must be 2D, got {shape}")
        if shape[0] % tp_size:
            raise ValueError(
                f"vocabulary {shape[0]} is not divisible by TP size {tp_size}"
            )
        rows_per_rank = shape[0] // tp_size
        start = tp_rank * rows_per_rank
        weight = weight_slice[start : start + rows_per_rank]
    if weight.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 lm-head weight, got {weight.dtype}")
    return weight.contiguous().to(device), start, weight_key


def _quantize_nvfp4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_scale = _global_scale(tensor)
    quantized, scale = flashinfer_nvfp4_quantize_128x4(
        tensor.contiguous(),
        global_scale,
    )
    return quantized, scale, global_scale


def _quantize_weight(weight: torch.Tensor) -> Nvfp4Weight:
    output_size = weight.shape[0]
    padded_output_size = (output_size + 31) // 32 * 32
    if padded_output_size != output_size:
        weight = F.pad(weight, (0, 0, 0, padded_output_size - output_size))
    quantized, scale, global_scale = _quantize_nvfp4(weight)
    return Nvfp4Weight(quantized, scale, global_scale, output_size)


def _nvfp4_mm(
    hidden_q: torch.Tensor,
    hidden_scale: torch.Tensor,
    hidden_global_scale: torch.Tensor,
    weight: Nvfp4Weight,
) -> torch.Tensor:
    logits = flashinfer_scaled_fp4_mm(
        hidden_q,
        weight.weight,
        hidden_scale,
        weight.scale,
        alpha=torch.reciprocal(hidden_global_scale * weight.global_scale),
        out_dtype=torch.bfloat16,
        backend="b12x",
        block_size=16,
        use_nvfp4=True,
    )
    return logits[:, : weight.output_size]


def _nvfp4_linear(hidden: torch.Tensor, weight: Nvfp4Weight) -> torch.Tensor:
    hidden_q, hidden_scale, hidden_global_scale = _quantize_nvfp4(hidden)
    return _nvfp4_mm(hidden_q, hidden_scale, hidden_global_scale, weight)


def _native_linear(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(hidden, weight)


def _native_topk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _native_linear(hidden, weight)
    if top_k > 1:
        logits = logits.float()
    return torch.topk(logits, top_k, dim=-1)


def _candidate_indexed_argmax(
    exact_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    index_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        HAS_TRITON
        and exact_logits.is_cuda
        and 0 < exact_logits.shape[-1] <= 1024
    ):
        return indexed_argmax_triton(
            exact_logits,
            candidate_indices,
            index_offset=index_offset,
        )
    values, positions = exact_logits.max(dim=-1)
    indices = candidate_indices.gather(1, positions.unsqueeze(-1)).squeeze(-1)
    return values.float(), indices.to(torch.int32) + index_offset


def _refine_selected_topk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    candidate_indices: torch.Tensor,
    *,
    top_k: int,
    candidate_tile: int | None = None,
    index_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    exact_logits = indexed_bf16_dot(
        hidden,
        weight,
        candidate_indices,
        candidate_tile=candidate_tile,
    )
    if top_k == 1:
        values, indices = _candidate_indexed_argmax(
            exact_logits,
            candidate_indices,
            index_offset=index_offset,
        )
        return values.unsqueeze(-1), indices.unsqueeze(-1)
    if top_k > 1:
        exact_logits = exact_logits.float()
    values, positions = torch.topk(exact_logits, top_k, dim=-1)
    indices = candidate_indices.gather(1, positions).to(torch.int32) + index_offset
    return values, indices


def _hybrid_topk(
    hidden: torch.Tensor,
    bf16_weight: torch.Tensor,
    nvfp4_weight: Nvfp4Weight,
    *,
    top_k: int,
    candidates: int,
    candidate_tile: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    coarse_logits = _nvfp4_linear(hidden, nvfp4_weight)
    coarse_indices = select_lm_head_candidates(coarse_logits, candidates)
    return _refine_selected_topk(
        hidden,
        bf16_weight,
        coarse_indices,
        top_k=top_k,
        candidate_tile=candidate_tile,
    )


def _reduce_tp_pairs(
    local_values: torch.Tensor,
    global_indices: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    if tp_size == 1:
        return global_indices.to(torch.int32)
    local_pairs = torch.stack(
        [local_values.float(), global_indices.float()],
        dim=-1,
    )
    gathered_pairs = (
        local_pairs[:, None, :]
        .expand(-1, tp_size, -1)
        .contiguous()
        .view(local_pairs.shape[0], tp_size * 2)
    )
    if HAS_TRITON and gathered_pairs.is_cuda:
        return reduce_global_argmax_triton(
            gathered_pairs,
            tp_size=tp_size,
        ).to(torch.int32)
    gathered_pairs = gathered_pairs.view(gathered_pairs.shape[0], tp_size, 2)
    winner = gathered_pairs[:, :, 0].argmax(dim=-1, keepdim=True)
    return gathered_pairs[:, :, 1].gather(1, winner).squeeze(1).to(torch.int32)


def _hybrid_greedy_integrated(
    hidden: torch.Tensor,
    bf16_weight: torch.Tensor,
    nvfp4_weight: Nvfp4Weight,
    *,
    candidates: int,
    vocab_start: int,
    tp_size: int,
    candidate_tile: int | None = None,
) -> torch.Tensor:
    coarse_logits = _nvfp4_linear(hidden, nvfp4_weight)
    candidate_indices = select_lm_head_candidates(coarse_logits, candidates)
    exact_logits = indexed_bf16_dot(
        hidden,
        bf16_weight,
        candidate_indices,
        candidate_tile=candidate_tile,
    )
    local_values, global_indices = _candidate_indexed_argmax(
        exact_logits,
        candidate_indices,
        index_offset=vocab_start,
    )
    return _reduce_tp_pairs(local_values, global_indices, tp_size)


def _time_cuda_graph(
    function: Callable[[], object],
    *,
    steps: int,
    warmup: int,
    trials: int,
) -> dict[str, float | list[float]]:
    outputs = function()
    torch.cuda.synchronize()
    del outputs
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = function()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(steps):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / steps)
    del outputs
    return {
        "median_us": median(samples),
        "samples_us": samples,
    }


def _candidate_recall(
    native_indices: torch.Tensor,
    coarse_indices: torch.Tensor,
) -> tuple[int, int]:
    matches = native_indices.unsqueeze(-1) == coarse_indices.unsqueeze(-2)
    hits = matches.any(dim=-1).sum().item()
    return int(hits), native_indices.numel()


def _relative_error(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    expected_abs = expected.float().abs()
    return {
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "rmse": float(difference.square().mean().sqrt().item()),
        "relative_l2": float(
            difference.square().sum().sqrt().item()
            / expected_abs.square().sum().sqrt().clamp_min(1.0).item()
        ),
    }


@torch.inference_mode()
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tp-size", type=_positive_int, default=2)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument(
        "--batch-sizes",
        type=_positive_int,
        nargs="+",
        default=[1, 8, 16, 32, 64],
    )
    parser.add_argument("--top-k", type=_positive_int, nargs="+", default=[1, 20])
    parser.add_argument("--candidates", type=_positive_int, default=128)
    parser.add_argument(
        "--candidate-tiles",
        type=_candidate_tile,
        nargs="+",
        default=[1, 2, 4, 8],
    )
    parser.add_argument("--seeds", type=_positive_int, default=20)
    parser.add_argument("--steps", type=_positive_int, default=40)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--trials", type=_positive_int, default=3)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if not 0 <= args.tp_rank < args.tp_size:
        raise ValueError("tp-rank must be in [0, tp-size)")
    if args.warmup < 0:
        raise ValueError("warmup must be non-negative")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    weight, vocab_start, weight_key = _load_weight_shard(
        args.model_dir,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        device=device,
    )
    local_vocab, hidden_size = weight.shape
    if hidden_size % 32:
        raise ValueError(f"hidden size must be divisible by 32, got {hidden_size}")
    if max(args.top_k) > local_vocab:
        raise ValueError("top-k exceeds the local vocabulary")
    if args.candidates < max(args.top_k) or args.candidates > local_vocab:
        raise ValueError("candidates must cover top-k and fit local vocabulary")

    quant_start = torch.cuda.Event(enable_timing=True)
    quant_end = torch.cuda.Event(enable_timing=True)
    quant_start.record()
    nvfp4_weight = _quantize_weight(weight)
    quant_end.record()
    quant_end.synchronize()
    weight_quantize_us = quant_start.elapsed_time(quant_end) * 1000.0

    payload: dict[str, object] = {
        "standard": "nvfp4-b12x-coarse-bf16-refined-lm-head",
        "model_dir": str(args.model_dir),
        "weight_key": weight_key,
        "tp_size": args.tp_size,
        "tp_rank": args.tp_rank,
        "vocab_start": vocab_start,
        "shape": {"n_local": local_vocab, "k": hidden_size},
        "backend": "b12x",
        "candidates": args.candidates,
        "greedy_integration": {
            "path": "coarse+select+refine+indexed_argmax+tp_pair_reduce",
            "tp_pair_reduce": "simulated_local_pair_gather",
        },
        "weight_quantize_us": weight_quantize_us,
        "results": {},
    }

    for batch_size in args.batch_sizes:
        generator = torch.Generator(device=device).manual_seed(20260828 + batch_size)
        hidden = torch.randn(
            (batch_size, hidden_size),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        hidden_q, hidden_scale, hidden_global_scale = _quantize_nvfp4(hidden)
        batch_results: dict[str, object] = {}

        native_linear_timing = _time_cuda_graph(
            partial(_native_linear, hidden, weight),
            steps=args.steps,
            warmup=args.warmup,
            trials=args.trials,
        )
        nvfp4_gemm_timing = _time_cuda_graph(
            partial(
                _nvfp4_mm,
                hidden_q,
                hidden_scale,
                hidden_global_scale,
                nvfp4_weight,
            ),
            steps=args.steps,
            warmup=args.warmup,
            trials=args.trials,
        )
        coarse_timing = _time_cuda_graph(
            partial(_nvfp4_linear, hidden, nvfp4_weight),
            steps=args.steps,
            warmup=args.warmup,
            trials=args.trials,
        )
        linear_speedup = float(native_linear_timing["median_us"]) / float(
            nvfp4_gemm_timing["median_us"]
        )

        native_logits = _native_linear(hidden, weight)
        coarse_logits = _nvfp4_linear(hidden, nvfp4_weight)
        coarse_error = _relative_error(coarse_logits, native_logits)

        selected_indices = torch.randint(
            0,
            local_vocab,
            (batch_size, args.candidates),
            dtype=torch.int64,
            device=device,
            generator=generator,
        ).contiguous()
        refined_logits = indexed_bf16_dot(hidden, weight, selected_indices)
        refined_reference = torch.bmm(
            hidden.unsqueeze(1),
            weight[selected_indices].transpose(1, 2),
        ).squeeze(1)
        refine_error = _relative_error(refined_logits, refined_reference)

        for top_k in args.top_k:
            exact_rows = 0
            candidate_hits = 0
            candidate_total = 0
            for seed in range(args.seeds):
                seed_generator = torch.Generator(device=device).manual_seed(seed)
                seed_hidden = torch.randn(
                    (batch_size, hidden_size),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=seed_generator,
                )
                native_values, native_indices = _native_topk(
                    seed_hidden,
                    weight,
                    top_k=top_k,
                )
                seed_coarse_logits = _nvfp4_linear(seed_hidden, nvfp4_weight)
                seed_coarse_indices = select_lm_head_candidates(
                    seed_coarse_logits,
                    args.candidates,
                )
                _, refined_indices = _refine_selected_topk(
                    seed_hidden,
                    weight,
                    seed_coarse_indices,
                    top_k=top_k,
                    candidate_tile=None,
                )
                del native_values
                hits, total = _candidate_recall(
                    native_indices,
                    seed_coarse_indices,
                )
                candidate_hits += hits
                candidate_total += total
                exact_matches = (
                    refined_indices.sort(dim=-1).values
                    == native_indices.sort(dim=-1).values
                )
                if exact_matches.ndim > 1:
                    exact_matches = exact_matches.all(dim=-1)
                exact_rows += int(exact_matches.sum().item())

            native_timing = _time_cuda_graph(
                partial(_native_topk, hidden, weight, top_k=top_k),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            coarse_indices = select_lm_head_candidates(
                coarse_logits,
                args.candidates,
            )
            selector_timing = _time_cuda_graph(
                partial(
                    select_lm_head_candidates,
                    coarse_logits,
                    args.candidates,
                ),
                steps=args.steps,
                warmup=args.warmup,
                trials=args.trials,
            )
            native_us = float(native_timing["median_us"])
            result = {
                "exact_set_rows": exact_rows,
                "rows_tested": args.seeds * batch_size,
                "candidate_hits": candidate_hits,
                "candidate_total": candidate_total,
                "native": native_timing,
                "selected_candidate_tile": _select_candidate_tile(
                    batch_size,
                    args.candidates,
                    hidden_size,
                ),
                "components": {
                    "selector": selector_timing,
                },
            }
            hybrid_by_tile: dict[str, object] = {}
            refine_by_tile: dict[str, object] = {}
            greedy_integrated_by_tile: dict[str, object] = {}
            for candidate_tile in args.candidate_tiles:
                tile_label = (
                    "auto" if candidate_tile is None else str(candidate_tile)
                )
                refine_timing = _time_cuda_graph(
                    partial(
                        _refine_selected_topk,
                        hidden,
                        weight,
                        coarse_indices,
                        top_k=top_k,
                        candidate_tile=candidate_tile,
                    ),
                    steps=args.steps,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                hybrid_timing = _time_cuda_graph(
                    partial(
                        _hybrid_topk,
                        hidden,
                        weight,
                        nvfp4_weight,
                        top_k=top_k,
                        candidates=args.candidates,
                        candidate_tile=candidate_tile,
                    ),
                    steps=args.steps,
                    warmup=args.warmup,
                    trials=args.trials,
                )
                if top_k == 1:
                    integrated_exact_logits = indexed_bf16_dot(
                        hidden,
                        weight,
                        coarse_indices,
                        candidate_tile=candidate_tile,
                    )
                    integrated_values, integrated_indices = _candidate_indexed_argmax(
                        integrated_exact_logits,
                        coarse_indices,
                        index_offset=vocab_start,
                    )
                    indexed_argmax_timing = _time_cuda_graph(
                        partial(
                            _candidate_indexed_argmax,
                            integrated_exact_logits,
                            coarse_indices,
                            index_offset=vocab_start,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    )
                    pair_reduce_timing = _time_cuda_graph(
                        partial(
                            _reduce_tp_pairs,
                            integrated_values,
                            integrated_indices,
                            args.tp_size,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    )
                    integrated_timing = _time_cuda_graph(
                        partial(
                            _hybrid_greedy_integrated,
                            hidden,
                            weight,
                            nvfp4_weight,
                            candidates=args.candidates,
                            vocab_start=vocab_start,
                            tp_size=args.tp_size,
                            candidate_tile=candidate_tile,
                        ),
                        steps=args.steps,
                        warmup=args.warmup,
                        trials=args.trials,
                    )
                else:
                    indexed_argmax_timing = None
                    pair_reduce_timing = None
                    integrated_timing = None
                hybrid_us = float(hybrid_timing["median_us"])
                refine_by_tile[tile_label] = refine_timing
                hybrid_by_tile[tile_label] = {
                    **hybrid_timing,
                    "speedup": native_us / hybrid_us,
                }
                if integrated_timing is not None:
                    integrated_us = float(integrated_timing["median_us"])
                    greedy_integrated_by_tile[tile_label] = {
                        **integrated_timing,
                        "speedup": native_us / integrated_us,
                        "overhead_vs_head_us": integrated_us - hybrid_us,
                        "components": {
                            "candidate_indexed_argmax": indexed_argmax_timing,
                            "tp_pair_reduce_simulated": pair_reduce_timing,
                        },
                    }
                    print(
                        f"m={batch_size} top_k={top_k} tile={tile_label} "
                        f"native={native_us:.3f}us hybrid={hybrid_us:.3f}us "
                        f"hybrid_speedup={native_us / hybrid_us:.3f}x "
                        f"integrated={integrated_us:.3f}us "
                        f"integrated_speedup={native_us / integrated_us:.3f}x "
                        f"integration_overhead={integrated_us - hybrid_us:.3f}us",
                        flush=True,
                    )
            result["hybrid_by_tile"] = hybrid_by_tile
            result["components"]["refine_by_tile"] = refine_by_tile
            if greedy_integrated_by_tile:
                result["greedy_integrated_by_tile"] = greedy_integrated_by_tile
            batch_results[str(top_k)] = result
            print(
                f"m={batch_size} top_k={top_k} candidates={args.candidates} "
                f"exact_set={exact_rows}/{args.seeds * batch_size} "
                f"recall={candidate_hits}/{candidate_total} "
                f"native_gemm={native_linear_timing['median_us']:.3f}us "
                f"nvfp4_gemm={nvfp4_gemm_timing['median_us']:.3f}us "
                f"gemm_speedup={linear_speedup:.3f}x "
                f"auto_tile={result['selected_candidate_tile']}",
                flush=True,
            )

        payload["results"][str(batch_size)] = {
            "native_linear": native_linear_timing,
            "nvfp4_gemm": nvfp4_gemm_timing,
            "coarse": coarse_timing,
            "gemm_speedup": linear_speedup,
            "coarse_error": coarse_error,
            "indexed_refine_error": refine_error,
            "top_k": batch_results,
        }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
