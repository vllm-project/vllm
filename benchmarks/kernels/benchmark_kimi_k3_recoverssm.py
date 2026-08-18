# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA-graph/CUPTI microbenchmark for Kimi-K3 RecoverSSM.

The benchmark deliberately keeps the production relationship between the
maximum speculative window and every request's query length: ``T`` is derived
from ``--num-speculative-tokens`` (default: seven drafts plus one target).
Accepted length is a separate runtime sweep for the commit path.
"""

import argparse
import importlib.util
import multiprocessing as mp
import statistics
import sys
from types import SimpleNamespace
import types
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import torch

try:
    from benchmarks.kernels.kimi_k3_cupti import CuptiKernelTimer
except ModuleNotFoundError:
    from kimi_k3_cupti import CuptiKernelTimer


def _install_local_kernel_dependencies() -> None:
    vllm = types.ModuleType("vllm")
    triton_utils = types.ModuleType("vllm.triton_utils")
    import triton
    import triton.language as tl

    triton_utils.triton = triton
    triton_utils.tl = tl
    utils = types.ModuleType("vllm.v1.attention.backends.utils")
    utils.NULL_BLOCK_ID = 0
    utils.PAD_SLOT_ID = -1
    mamba_utils = types.ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    mamba_utils.is_conv_state_dim_first = lambda: False
    platforms = types.ModuleType("vllm.platforms")
    platforms.current_platform = SimpleNamespace(
        is_cpu=lambda: False, is_arch_support_pdl=lambda: True
    )
    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.triton_utils": triton_utils,
            "vllm.v1.attention.backends.utils": utils,
            "vllm.model_executor.layers.mamba.mamba_utils": mamba_utils,
            "vllm.platforms": platforms,
        }
    )


def _load_local_kernel_module(name: str):
    path = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_install_local_kernel_dependencies()
_conv = _load_local_kernel_module("kimi_k3_recoverssm_conv.py")
_recoverssm = _load_local_kernel_module("kimi_k3_recoverssm_kernels.py")
causal_conv1d_update = _conv.causal_conv1d_update
KDARecoverSSMCommitContext = _recoverssm.KDARecoverSSMCommitContext
kda_recoverssm_verify = _recoverssm.kda_recoverssm_verify


DEFAULT_NUM_SPECULATIVE_TOKENS = 7
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64)
KIMI_K3_NUM_HEADS = 96
KIMI_K3_HEAD_DIM = 128
KIMI_K3_TP_SIZE = 8
# Kimi K3 has 69 KDA layers. This is not the model's total transformer-layer
# count; RecoverSSM commit runs once across this KDA-only set.
NUM_KDA_LAYERS = 69
CONV_WIDTH = 4
L2_FLUSH_FLOATS = 128 * 1024 * 1024


def local_num_heads(tp_size: int) -> int:
    if KIMI_K3_NUM_HEADS % tp_size:
        raise ValueError("Kimi-K3 head count must divide TP size")
    return KIMI_K3_NUM_HEADS // tp_size


class Inputs:
    def __init__(self, batch: int, spec_query_len: int, tp_size: int) -> None:
        h = local_num_heads(tp_size)
        d = KIMI_K3_HEAD_DIM
        tokens = batch * spec_query_len
        slots = batch + 1
        dtype = torch.bfloat16
        device = "cuda"
        self.batch = batch
        self.t = spec_query_len
        self.h = h
        self.mixed_qkv_source = torch.randn(
            tokens, 3 * h * d, device=device, dtype=dtype
        )
        self.mixed_qkv = torch.empty_like(self.mixed_qkv_source)
        self.conv_out = torch.empty_like(self.mixed_qkv)
        self.gate = torch.randn(1, tokens, h, d, device=device, dtype=dtype)
        self.beta = torch.randn(1, tokens, h, device=device, dtype=dtype)
        self.query_start_loc = torch.arange(
            0, tokens + 1, spec_query_len, device=device, dtype=torch.int32
        )
        self.state_indices = torch.arange(1, batch + 1, device=device, dtype=torch.int32)
        self.accepted = torch.ones(batch, device=device, dtype=torch.int32)
        self.weights = torch.randn(3 * h * d, CONV_WIDTH, device=device, dtype=dtype)
        self.bias = torch.randn(3 * h * d, device=device, dtype=dtype)
        self.conv_state = torch.randn(
            slots, 3 * h * d, CONV_WIDTH - 1 + spec_query_len - 1,
            device=device, dtype=dtype,
        )
        self.state = torch.randn(slots, h, d, d, device=device, dtype=dtype)
        self.correction = torch.randn(
            slots, h, spec_query_len, d, device=device, dtype=torch.float32
        )
        self.kg = torch.randn(
            slots, h, spec_query_len, 2 * d, device=device, dtype=dtype
        )
        self.a_log = torch.randn(h, device=device, dtype=torch.float32)
        self.dt_bias = torch.randn(h * d, device=device, dtype=torch.float32)
        self._layers = [
            SimpleNamespace(
                kv_cache=(
                    self.conv_state.transpose(-1, -2).clone(),
                    self.state.clone(),
                    self.correction.clone(),
                    self.kg.clone(),
                ),
                A_log=self.a_log.clone(),
                dt_bias=self.dt_bias.clone(),
                local_num_heads=h,
                head_dim=d,
                gate_lower_bound=None,
            )
            for _ in range(NUM_KDA_LAYERS)
        ]
        self.commit_context = KDARecoverSSMCommitContext.create(
            self._layers, spec_query_len=spec_query_len, max_num_reqs=batch
        )
        self._verify_conv_state = self.conv_state.clone()
        self._commit_conv_states = tuple(layer.kv_cache[0].clone() for layer in self._layers)
        self._commit_states = tuple(layer.kv_cache[1].clone() for layer in self._layers)

    def verify(self) -> None:
        x = causal_conv1d_update(
            self.mixed_qkv, self.conv_state, self.weights, self.bias, activation="silu",
            conv_state_indices=self.state_indices, num_accepted_tokens=self.accepted,
            query_start_loc=self.query_start_loc, max_query_len=self.t, out=self.conv_out,
        )
        q, k, v = (part.view(1, -1, self.h, KIMI_K3_HEAD_DIM) for part in x.split(self.h * KIMI_K3_HEAD_DIM, dim=-1))
        kda_recoverssm_verify(q, k, v, self.gate, self.beta, self.a_log, self.dt_bias,
                              None, self.state, self.correction, self.kg,
                              self.query_start_loc, self.state_indices, self.t)

    def reset_verify(self) -> None:
        self.conv_state.copy_(self._verify_conv_state)

    def reset_commit(self) -> None:
        for layer, conv_state, state in zip(
            self._layers, self._commit_conv_states, self._commit_states, strict=True
        ):
            layer.kv_cache[0].copy_(conv_state)
            layer.kv_cache[1].copy_(state)

    def warm_projection_output(self) -> None:
        self.mixed_qkv.copy_(self.mixed_qkv_source)

    def commit(self, accepted_tokens: int) -> None:
        self.accepted.fill_(accepted_tokens)
        self.commit_context.commit(self.accepted, self.state_indices, self.query_start_loc)


def capture_graph(run, reset, l2_flush: torch.Tensor, setup=None) -> torch.cuda.CUDAGraph:
    reset()
    if setup is not None:
        setup()
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reset()
        l2_flush.fill_(0.0)
        if setup is not None:
            setup()
        run()
    return graph


def _compile_warmup_worker(batch: int, spec_query_len: int, accepted: int, tp_size: int) -> None:
    inputs = Inputs(batch, spec_query_len, tp_size)
    inputs.warm_projection_output()
    inputs.verify()
    inputs.commit(accepted)
    torch.cuda.synchronize()


def compile_warmup(batch_sizes: tuple[int, ...], spec_query_len: int,
                   accepted: int, tp_size: int, workers: int) -> None:
    if workers <= 0:
        return
    print(f"compile warmup: {len(batch_sizes)} shapes across {workers} processes")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = [
            executor.submit(_compile_warmup_worker, batch, spec_query_len, accepted, tp_size)
            for batch in batch_sizes
        ]
        for future in futures:
            future.result()


def _percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def measure_graph(graph: torch.cuda.CUDAGraph, *, tag: str, targets: tuple[str, ...],
                  warmup: int, iters: int) -> None:
    timer = CuptiKernelTimer.get()
    calibration = timer.capture_names(graph.replay)
    names = tuple(name if name and any(key in name for key in targets) else None
                  for name, *_ in calibration[0])
    expected_kernels = len(targets)
    if sum(name is not None for name in names) != expected_kernels:
        found = [name for name, *_ in calibration[0] if name]
        raise RuntimeError(f"{tag}: expected {expected_kernels} kernels, found {found}")
    plan = ((len(calibration[0]), names),) * (warmup + iters)
    timer.start(plan)
    for _ in range(warmup + iters):
        graph.replay()
    torch.cuda.synchronize()
    records, _, _, _ = timer.stop()
    if len(records) != expected_kernels * (warmup + iters):
        raise RuntimeError(f"{tag}: CUPTI returned {len(records)} target records")
    spans = []
    per_kernel: dict[str, list[float]] = {}
    for i in range(warmup, warmup + iters):
        group = records[i * expected_kernels : (i + 1) * expected_kernels]
        starts = [record[1] for record in group]
        ends = [record[2] for record in group]
        spans.append((max(ends) - min(starts)) / 1e3)
        for name, start, end, *_ in group:
            per_kernel.setdefault(name or "unknown", []).append((end - start) / 1e3)
    kernel_summary = ", ".join(
        f"{name}={statistics.median(values):.2f}us" for name, values in per_kernel.items()
    )
    print(
        f"{tag}: mean={statistics.mean(spans):.2f}us median={statistics.median(spans):.2f}us "
        f"p95={_percentile(spans, .95):.2f}us p99={_percentile(spans, .99):.2f}us; "
        f"{kernel_summary}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64")
    parser.add_argument(
        "--num-speculative-tokens", type=int, default=DEFAULT_NUM_SPECULATIVE_TOKENS
    )
    parser.add_argument("--accepted-tokens", default="1")
    parser.add_argument("--tp-size", type=int, default=KIMI_K3_TP_SIZE)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_query_len = args.num_speculative_tokens + 1
    if spec_query_len < 1:
        raise ValueError("--num-speculative-tokens must be non-negative")
    batch_sizes = tuple(int(value) for value in args.batch_sizes.split(","))
    accepted_tokens = tuple(int(value) for value in args.accepted_tokens.split(","))
    if any(value < 0 or value > spec_query_len for value in accepted_tokens):
        raise ValueError("accepted tokens must be in [0, T]")
    l2_flush = torch.empty(L2_FLUSH_FLOATS, device="cuda", dtype=torch.float32)
    print(
        f"RecoverSSM benchmark: heads={local_num_heads(args.tp_size)}, "
        f"head_dim={KIMI_K3_HEAD_DIM}, T={spec_query_len}, batches={batch_sizes}, "
        f"accepted={accepted_tokens}, device={torch.cuda.get_device_name()}"
    )
    compile_warmup(
        batch_sizes, spec_query_len, accepted_tokens[0], args.tp_size,
        min(args.compile_workers, len(batch_sizes)),
    )
    for batch in batch_sizes:
        inputs = Inputs(batch, spec_query_len, args.tp_size)
        verify_graph = capture_graph(
            inputs.verify, inputs.reset_verify, l2_flush, inputs.warm_projection_output
        )
        measure_graph(
            verify_graph,
            tag=f"verify batch={batch}",
            targets=("_causal_conv1d_update_kernel", "_kda_recoverssm_verify_kernel"),
            warmup=args.warmup,
            iters=args.iters,
        )
        for accepted in accepted_tokens:
            commit_graph = capture_graph(
                lambda accepted=accepted: inputs.commit(accepted),
                inputs.reset_commit,
                l2_flush,
            )
            measure_graph(
                commit_graph,
                tag=f"commit batch={batch} accepted={accepted}",
                targets=(
                    "_prepare_commit_plan_kernel",
                    "_compact_conv_state_kernel",
                    "_commit_kda_state_kernel",
                ),
                warmup=args.warmup,
                iters=args.iters,
            )


if __name__ == "__main__":
    main()
