"""Compare the local RecoverSSM commit copy with the production implementation."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_benchmark_module():
    sys.path.insert(0, str(BENCHMARK_DIR))
    return load_module(
        "recoverssm_benchmark",
        BENCHMARK_DIR / "benchmark_kimi_k3_recoverssm.py",
    )


def clone_layers(layers):
    return [
        SimpleNamespace(
            kv_cache=tuple(value.clone() for value in layer.kv_cache),
            A_log=layer.A_log.clone(),
            dt_bias=layer.dt_bias.clone(),
            local_num_heads=layer.local_num_heads,
            head_dim=layer.head_dim,
            gate_lower_bound=layer.gate_lower_bound,
        )
        for layer in layers
    ]


def compare(batch: int, spec_query_len: int, accepted_tokens: int, tp_size: int):
    benchmark = load_benchmark_module()
    production = load_module(
        "production_recoverssm",
        REPO_ROOT / "vllm/models/kimi_k3/nvidia/ops/recoverssm.py",
    )
    inputs = benchmark.Inputs(batch, spec_query_len, tp_size)
    reference_layers = clone_layers(inputs._layers)
    reference = production.KDARecoverSSMCommitContext.create(
        reference_layers,
        spec_query_len=spec_query_len,
        max_num_reqs=batch,
    )
    inputs.accepted.fill_(accepted_tokens)
    reference.commit(inputs.accepted, inputs.state_indices, inputs.query_start_loc)
    inputs.commit_context.commit(
        inputs.accepted, inputs.state_indices, inputs.query_start_loc
    )
    torch.cuda.synchronize()

    max_abs = 0.0
    max_rel = 0.0
    mean_abs_sum = 0.0
    numel = 0
    differing = 0
    for candidate, expected in zip(inputs._layers, reference_layers, strict=True):
        for actual, wanted in zip(candidate.kv_cache[:2], expected.kv_cache[:2], strict=True):
            diff = (actual.float() - wanted.float()).abs()
            max_abs = max(max_abs, diff.max().item())
            max_rel = max(
                max_rel,
                (diff / wanted.float().abs().clamp_min(1e-5)).max().item(),
            )
            mean_abs_sum += diff.sum().item()
            numel += diff.numel()
            differing += (diff != 0).sum().item()
    return max_abs, max_rel, mean_abs_sum / numel, differing / numel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,32")
    parser.add_argument("--num-speculative-tokens", type=int, default=7)
    parser.add_argument("--accepted-tokens", type=int, default=5)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--require-bitwise", action="store_true")
    args = parser.parse_args()

    spec_query_len = args.num_speculative_tokens + 1
    all_bitwise = True
    for batch in (int(value) for value in args.batch_sizes.split(",")):
        max_abs, max_rel, mean_abs, differing = compare(
            batch, spec_query_len, args.accepted_tokens, args.tp_size
        )
        all_bitwise &= differing == 0.0
        print(
            f"batch={batch}: max_abs={max_abs} max_rel={max_rel} "
            f"mean_abs={mean_abs} differing={differing:.2%}"
        )
    if args.require_bitwise and not all_bitwise:
        raise SystemExit("local RecoverSSM commit is not bitwise-identical")


if __name__ == "__main__":
    main()
