#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark scalar overhead in attention boundary computation.

This script compares the current vectorized helpers against local legacy-style
baselines that intentionally use Python scalar extraction in the hot path. It
is a micro-benchmark for boundary/routing computation only, not a full model
throughput benchmark.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

BoundaryFn = Callable[..., tuple[int, ...]]


@dataclass(frozen=True)
class BoundaryCase:
    name: str
    query_start_loc: torch.Tensor
    num_reqs: int
    num_tokens: int
    max_query_len: int
    decode_threshold: int
    query_lens: torch.Tensor | None = None
    require_uniform: bool = False
    treat_short_extends_as_decodes: bool = True
    is_prefilling: torch.Tensor | None = None


@dataclass(frozen=True)
class ExtendCase:
    name: str
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    num_tokens: int
    max_query_len: int
    decode_threshold: int


@dataclass(frozen=True)
class ChunkCase:
    name: str
    seq_lens: torch.Tensor
    workspace_size: int


def _default_vllm_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_ascend_root() -> Path:
    return _default_vllm_root().parent / "vllm-ascend-hust"


def _prepend_repo_paths(repo_root: Path, ascend_root: Path | None) -> None:
    paths = [str(repo_root)]
    if ascend_root is not None:
        paths.append(str(ascend_root))
    for path in reversed(paths):
        if path not in sys.path:
            sys.path.insert(0, path)


def _extract_functions_from_source(
    path: Path, names: set[str]
) -> dict[str, Callable[..., object]]:
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    found = {node.name for node in selected}
    missing = names - found
    if missing:
        raise RuntimeError(f"missing functions in {path}: {sorted(missing)}")

    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(path), "exec"), namespace)
    return {name: namespace[name] for name in names}


def _load_vllm_source_helpers(
    repo_root: Path, names: set[str]
) -> dict[str, Callable[..., object]]:
    return _extract_functions_from_source(
        repo_root / "vllm/v1/attention/backends/utils.py", names
    )


def _load_ascend_source_helpers(
    ascend_root: Path, names: set[str]
) -> dict[str, Callable[..., object]]:
    return _extract_functions_from_source(
        ascend_root / "vllm_ascend/attention/utils.py", names
    )


def _load_boundary_helper(
    repo: str, repo_root: Path, ascend_root: Path | None, load_mode: str
) -> BoundaryFn:
    if repo == "vllm":
        module_name = "vllm.v1.attention.backends.utils"
        source_root = repo_root
    elif repo == "ascend":
        module_name = "vllm_ascend.attention.utils"
        if ascend_root is None:
            raise ValueError("--ascend-root is required for --repo ascend")
        source_root = ascend_root
    else:
        raise ValueError(f"unknown repo: {repo}")

    if load_mode in ("auto", "import"):
        try:
            module = importlib.import_module(module_name)
            return module._split_decode_prefill_boundary
        except ModuleNotFoundError:
            if load_mode == "import":
                raise

    helper_names = {"_first_true_index", "_split_decode_prefill_boundary"}
    if repo == "vllm":
        helpers = _load_vllm_source_helpers(source_root, helper_names)
    else:
        helpers = _load_ascend_source_helpers(source_root, helper_names)
    return helpers["_split_decode_prefill_boundary"]


def _load_extend_helper(repo_root: Path, load_mode: str) -> BoundaryFn:
    if load_mode in ("auto", "import"):
        try:
            module = importlib.import_module("vllm.v1.attention.backends.utils")
            return module._split_decode_extend_prefill_boundary
        except ModuleNotFoundError:
            if load_mode == "import":
                raise

    helpers = _load_vllm_source_helpers(
        repo_root, {"_first_true_index", "_split_decode_extend_prefill_boundary"}
    )
    return helpers["_split_decode_extend_prefill_boundary"]


def _load_chunk_helper(
    repo_root: Path, load_mode: str
) -> Callable[[torch.Tensor, int], list[tuple[int, int]]]:
    if load_mode in ("auto", "import"):
        try:
            module = importlib.import_module("vllm.v1.attention.backends.utils")
            return module.split_prefill_chunks
        except ModuleNotFoundError:
            if load_mode == "import":
                raise

    helpers = _load_vllm_source_helpers(repo_root, {"split_prefill_chunks"})
    return helpers["split_prefill_chunks"]


def _make_query_start_loc(query_lens: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    query_start_loc = torch.zeros(query_lens.numel() + 1, dtype=dtype)
    query_start_loc[1:] = query_lens.to(dtype).cumsum(0)
    return query_start_loc


def _legacy_split_decode_prefill_boundary(
    query_start_loc: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
    decode_threshold: int = 1,
    *,
    query_lens: torch.Tensor | None = None,
    require_uniform: bool = False,
    treat_short_extends_as_decodes: bool = True,
    is_prefilling: torch.Tensor | None = None,
) -> tuple[int, int, int, int]:
    if num_reqs == 0:
        return 0, 0, 0, 0

    if (
        max_query_len <= decode_threshold
        and (not require_uniform or decode_threshold <= 1)
        and treat_short_extends_as_decodes
    ):
        return num_reqs, 0, num_tokens, 0

    query_start_loc = query_start_loc[: num_reqs + 1]
    if query_lens is None:
        query_lens = torch.diff(query_start_loc)
    else:
        query_lens = query_lens[:num_reqs]

    first_prefill = num_reqs
    if require_uniform:
        first_query_len = int(query_lens[0].item())
        uniform_or_pad = True
        for i in range(num_reqs):
            q_len = int(query_lens[i].item())
            if q_len != first_query_len and q_len != 0:
                uniform_or_pad = False
                break

        first_is_prefill = first_query_len > decode_threshold
        force_all_decode = uniform_or_pad and not first_is_prefill
        for i in range(num_reqs):
            q_len = int(query_lens[i].item())
            if force_all_decode:
                is_prefill = False
            elif first_is_prefill:
                is_prefill = True
            else:
                is_prefill = q_len != first_query_len

            if not treat_short_extends_as_decodes:
                assert is_prefilling is not None
                is_prefill = is_prefill or (
                    not force_all_decode and bool(is_prefilling[i].item())
                )
            if is_prefill:
                first_prefill = i
                break
    else:
        for i in range(num_reqs):
            q_len = int(query_lens[i].item())
            is_prefill = q_len > decode_threshold
            if not treat_short_extends_as_decodes:
                assert is_prefilling is not None
                is_prefill = is_prefill or bool(is_prefilling[i].item())
            if is_prefill:
                first_prefill = i
                break

    num_decodes = first_prefill
    num_prefills = num_reqs - num_decodes
    if first_prefill < num_reqs:
        num_decode_tokens = int(query_start_loc[first_prefill].item())
    else:
        num_decode_tokens = num_tokens
    num_prefill_tokens = num_tokens - num_decode_tokens
    return num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens


def _legacy_split_decode_extend_prefill_boundary(
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_reqs: int,
    num_tokens: int,
    max_query_len: int,
    decode_threshold: int = 1,
) -> tuple[int, int, int, int, int, int]:
    if num_reqs == 0:
        return 0, 0, 0, 0, 0, 0

    if max_query_len <= decode_threshold:
        return num_reqs, 0, 0, num_tokens, 0, 0

    first_extend = num_reqs
    first_prefill = num_reqs
    for i in range(num_reqs):
        q_len = int((query_start_loc[i + 1] - query_start_loc[i]).item())
        if q_len > decode_threshold:
            first_extend = i
            break

    for i in range(num_reqs):
        q_len = int((query_start_loc[i + 1] - query_start_loc[i]).item())
        seq_len = int(seq_lens[i].item())
        if q_len > decode_threshold and seq_len == q_len:
            first_prefill = i
            break

    num_decodes = first_extend
    if first_extend < num_reqs:
        num_decode_tokens = int(query_start_loc[first_extend].item())
    else:
        num_decode_tokens = num_tokens
    num_prefill_or_extend_tokens = num_tokens - num_decode_tokens
    num_extends = first_prefill - num_decodes
    num_prefills = num_reqs - first_prefill
    if first_prefill < num_reqs:
        num_prefill_tokens = num_tokens - int(query_start_loc[first_prefill].item())
    else:
        num_prefill_tokens = 0
    num_extend_tokens = num_prefill_or_extend_tokens - num_prefill_tokens
    return (
        num_decodes,
        num_extends,
        num_prefills,
        num_decode_tokens,
        num_extend_tokens,
        num_prefill_tokens,
    )


def _legacy_split_prefill_chunks(
    seq_lens_cpu: torch.Tensor, workspace_size: int, request_offset: int = 0
) -> list[tuple[int, int]]:
    chunk_bounds = []
    i, n = 0, len(seq_lens_cpu)
    assert all(seq_lens_cpu[i].item() <= workspace_size for i in range(n))

    while i < n:
        start, chunk_total = i, 0
        while i < n and chunk_total + seq_lens_cpu[i].item() <= workspace_size:
            chunk_total += seq_lens_cpu[i].item()
            i += 1
        chunk_bounds.append((start + request_offset, i + request_offset))
    return chunk_bounds


def _boundary_cases(dtype: torch.dtype) -> list[BoundaryCase]:
    mixed_lens = torch.cat(
        [
            torch.ones(3072, dtype=torch.int64),
            torch.full((1024,), 512, dtype=torch.int64),
        ]
    )
    mixed_qsl = _make_query_start_loc(mixed_lens, dtype)

    long_lens = torch.cat(
        [
            torch.ones(4096, dtype=torch.int64),
            torch.full((1024,), 8192, dtype=torch.int64),
        ]
    )
    long_qsl = _make_query_start_loc(long_lens, dtype)

    short_extend_lens = torch.cat(
        [
            torch.ones(2048, dtype=torch.int64),
            torch.full((1024,), 2, dtype=torch.int64),
            torch.full((1024,), 256, dtype=torch.int64),
        ]
    )
    short_extend_qsl = _make_query_start_loc(short_extend_lens, dtype)
    is_prefilling = torch.zeros(short_extend_lens.numel(), dtype=torch.bool)
    is_prefilling[2048:] = True

    uniform_lens = torch.cat(
        [
            torch.full((4095,), 2, dtype=torch.int64),
            torch.zeros(1, dtype=torch.int64),
        ]
    )
    uniform_qsl = _make_query_start_loc(uniform_lens, dtype)

    return [
        BoundaryCase(
            name="mixed_decode_prefill_4096req",
            query_start_loc=mixed_qsl,
            num_reqs=mixed_lens.numel(),
            num_tokens=int(mixed_lens.sum().item()),
            max_query_len=int(mixed_lens.max().item()),
            decode_threshold=1,
        ),
        BoundaryCase(
            name="long_context_mixed_5120req",
            query_start_loc=long_qsl,
            num_reqs=long_lens.numel(),
            num_tokens=int(long_lens.sum().item()),
            max_query_len=int(long_lens.max().item()),
            decode_threshold=1,
        ),
        BoundaryCase(
            name="short_extends_as_prefill_4096req",
            query_start_loc=short_extend_qsl,
            num_reqs=short_extend_lens.numel(),
            num_tokens=int(short_extend_lens.sum().item()),
            max_query_len=int(short_extend_lens.max().item()),
            decode_threshold=4,
            treat_short_extends_as_decodes=False,
            is_prefilling=is_prefilling,
        ),
        BoundaryCase(
            name="uniform_padded_decode_4096req",
            query_start_loc=uniform_qsl,
            num_reqs=uniform_lens.numel(),
            num_tokens=int(uniform_lens.sum().item()) + 2,
            max_query_len=int(uniform_lens.max().item()),
            decode_threshold=3,
            require_uniform=True,
        ),
    ]


def _extend_cases(dtype: torch.dtype) -> list[ExtendCase]:
    query_lens = torch.cat(
        [
            torch.ones(2048, dtype=torch.int64),
            torch.full((1024,), 64, dtype=torch.int64),
            torch.full((1024,), 512, dtype=torch.int64),
        ]
    )
    seq_lens = torch.cat(
        [
            torch.full((2048,), 4096, dtype=torch.int64),
            torch.full((1024,), 4096, dtype=torch.int64),
            torch.full((1024,), 512, dtype=torch.int64),
        ]
    )
    query_start_loc = _make_query_start_loc(query_lens, dtype)
    return [
        ExtendCase(
            name="decode_extend_prefill_4096req",
            query_start_loc=query_start_loc,
            seq_lens=seq_lens.to(dtype),
            num_reqs=query_lens.numel(),
            num_tokens=int(query_lens.sum().item()),
            max_query_len=int(query_lens.max().item()),
            decode_threshold=4,
        )
    ]


def _chunk_cases() -> list[ChunkCase]:
    base = torch.tensor([8192, 16384, 4096, 32768, 2048, 4096], dtype=torch.int64)
    seq_lens = base.repeat(1024)
    return [
        ChunkCase(
            name="long_context_prefill_chunks_6144req",
            seq_lens=seq_lens,
            workspace_size=65536,
        )
    ]


def _time_call(fn: Callable[[], object], *, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        end = time.perf_counter_ns()
        samples.append((end - start) / iters / 1000.0)
    return statistics.median(samples)


def _print_result(name: str, legacy_us: float, current_us: float) -> None:
    speedup = legacy_us / current_us if current_us > 0 else float("inf")
    delta = (legacy_us - current_us) / legacy_us * 100.0 if legacy_us > 0 else 0.0
    print(
        f"{name:42s} legacy={legacy_us:10.2f} us  "
        f"current={current_us:10.2f} us  speedup={speedup:7.2f}x  "
        f"python_overhead_reduction={delta:6.1f}%"
    )


def run(args: argparse.Namespace) -> None:
    repo_root = Path(args.vllm_root).resolve()
    ascend_root = Path(args.ascend_root).resolve() if args.ascend_root else None
    _prepend_repo_paths(repo_root, ascend_root)

    dtype = torch.int64 if args.dtype == "int64" else torch.int32
    boundary_helper = _load_boundary_helper(
        args.repo, repo_root, ascend_root, args.load_mode
    )
    extend_helper = (
        _load_extend_helper(repo_root, args.load_mode) if args.repo == "vllm" else None
    )
    chunk_helper = (
        _load_chunk_helper(repo_root, args.load_mode) if args.repo == "vllm" else None
    )

    print(f"repo={args.repo} dtype={dtype} iters={args.iters} warmup={args.warmup}")
    print("All cases validate output equivalence before timing.\n")

    for case in _boundary_cases(dtype):
        legacy_args = (
            case.query_start_loc,
            case.num_reqs,
            case.num_tokens,
            case.max_query_len,
            case.decode_threshold,
        )
        legacy_kwargs = {
            "query_lens": case.query_lens,
            "require_uniform": case.require_uniform,
            "treat_short_extends_as_decodes": case.treat_short_extends_as_decodes,
            "is_prefilling": case.is_prefilling,
        }

        expected = _legacy_split_decode_prefill_boundary(*legacy_args, **legacy_kwargs)
        actual = boundary_helper(*legacy_args, **legacy_kwargs)
        if actual != expected:
            raise AssertionError((case.name, actual, expected))

        legacy_us = _time_call(
            lambda legacy_args=legacy_args, legacy_kwargs=legacy_kwargs: (
                _legacy_split_decode_prefill_boundary(*legacy_args, **legacy_kwargs)
            ),
            iters=args.iters,
            warmup=args.warmup,
        )
        current_us = _time_call(
            lambda legacy_args=legacy_args, legacy_kwargs=legacy_kwargs: (
                boundary_helper(*legacy_args, **legacy_kwargs)
            ),
            iters=args.iters,
            warmup=args.warmup,
        )
        _print_result(case.name, legacy_us, current_us)

    if extend_helper is not None:
        for case in _extend_cases(dtype):
            call_args = (
                case.query_start_loc,
                case.seq_lens,
                case.num_reqs,
                case.num_tokens,
                case.max_query_len,
                case.decode_threshold,
            )
            expected = _legacy_split_decode_extend_prefill_boundary(*call_args)
            actual = extend_helper(*call_args)
            if actual != expected:
                raise AssertionError((case.name, actual, expected))

            legacy_us = _time_call(
                lambda call_args=call_args: (
                    _legacy_split_decode_extend_prefill_boundary(*call_args)
                ),
                iters=args.iters,
                warmup=args.warmup,
            )
            current_us = _time_call(
                lambda call_args=call_args: extend_helper(*call_args),
                iters=args.iters,
                warmup=args.warmup,
            )
            _print_result(case.name, legacy_us, current_us)

    if chunk_helper is not None:
        chunk_iters = max(1, args.iters // 10)
        chunk_warmup = max(1, args.warmup // 10)
        for case in _chunk_cases():
            expected = _legacy_split_prefill_chunks(case.seq_lens, case.workspace_size)
            actual = chunk_helper(case.seq_lens, case.workspace_size)
            if actual != expected:
                raise AssertionError((case.name, actual[:5], expected[:5]))

            legacy_us = _time_call(
                lambda seq_lens=case.seq_lens, workspace_size=case.workspace_size: (
                    _legacy_split_prefill_chunks(seq_lens, workspace_size)
                ),
                iters=chunk_iters,
                warmup=chunk_warmup,
            )
            current_us = _time_call(
                lambda seq_lens=case.seq_lens, workspace_size=case.workspace_size: (
                    chunk_helper(seq_lens, workspace_size)
                ),
                iters=chunk_iters,
                warmup=chunk_warmup,
            )
            _print_result(case.name, legacy_us, current_us)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        choices=("vllm", "ascend"),
        default="vllm",
        help="which modified helper to benchmark",
    )
    parser.add_argument(
        "--vllm-root",
        default=str(_default_vllm_root()),
        help="path to the vllm-hust repository",
    )
    parser.add_argument(
        "--ascend-root",
        default=str(_default_ascend_root()),
        help="path to the vllm-ascend-hust repository",
    )
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--dtype", choices=("int32", "int64"), default="int32")
    parser.add_argument(
        "--load-mode",
        choices=("auto", "import", "source"),
        default="auto",
        help="load helpers by importing modules, source extraction, or auto fallback",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
