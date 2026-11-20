# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense BF16 and compiled hierarchical scoring using shared inputs."""

import argparse
import importlib.util
import inspect
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import vllm
from vllm.model_executor.layers.hierarchical_lm_head import (
    HierarchicalLMHead,
    selected_logits,
)
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-size", type=int, default=2688)
    parser.add_argument("--vocab-size", type=int, default=131072)
    parser.add_argument("--num-centroids", type=int, default=4096)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument(
        "--sidecar", help="Optional router safetensors; otherwise use random centroids"
    )
    parser.add_argument(
        "--upstream-source",
        type=Path,
        help="Optional upstream gemma4_mtp.py; benchmark its exact masked embedder",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Local Gemma4 assistant snapshot; use its BF16 weights and router",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--checkpoint-repo", help="Download/use a cached assistant checkpoint"
    )
    parser.add_argument("--checkpoint-revision", default="main")
    parser.add_argument("--input-pool-sizes", nargs="+", type=int, default=[1])
    parser.add_argument("--report-stages", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--rep-ms", type=int, default=200)
    args = parser.parse_args()
    if min(args.input_pool_sizes) < 1 or args.repeats < 1 or args.rep_ms < 1:
        parser.error("pool sizes, repeats and rep-ms must be positive")
    if args.checkpoint_repo:
        if args.checkpoint:
            parser.error("use checkpoint or checkpoint-repo, not both")
        from vllm.transformers_utils.repo_utils import hf_api

        args.checkpoint = Path(
            hf_api().snapshot_download(
                args.checkpoint_repo,
                revision=args.checkpoint_revision,
                allow_patterns=["config.json", "model.safetensors"],
            )
        )
    if args.checkpoint:
        config = json.loads((args.checkpoint / "config.json").read_text())
        text_config = config.get("text_config", config)
        args.hidden_size = text_config["hidden_size"]
        args.vocab_size = text_config["vocab_size"]
        args.num_centroids = config["num_centroids"]
        args.top_k = config["centroid_intermediate_top_k"]
    torch.manual_seed(0)
    heads = {
        "compiled": HierarchicalLMHead(
            args.hidden_size,
            args.vocab_size,
            args.num_centroids,
            args.top_k,
        ).to(device="cuda", dtype=torch.bfloat16)
    }
    if args.upstream_source:
        spec = importlib.util.spec_from_file_location(
            "vllm.model_executor.models.gemma4_mtp_benchmark_reference",
            args.upstream_source,
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        heads["upstream_einsum"] = module.Gemma4MTPMaskedEmbedder(
            args.hidden_size,
            args.vocab_size,
            args.num_centroids,
            args.top_k,
        ).to(device="cuda", dtype=torch.bfloat16)
    first = next(iter(heads.values()))
    first.token_ordering.copy_(torch.arange(args.vocab_size, device="cuda"))
    weight = torch.randn(
        args.vocab_size, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    if args.checkpoint:
        from safetensors import safe_open

        with safe_open(
            args.checkpoint / "model.safetensors", framework="pt", device="cpu"
        ) as tensors:
            for name, destination in [
                ("masked_embedding.centroids.weight", first.centroids.weight),
                ("masked_embedding.token_ordering", first.token_ordering),
            ]:
                destination.copy_(tensors.get_tensor(name))
            key = (
                "lm_head.weight"
                if "lm_head.weight" in set(tensors.keys())
                else "model.embed_tokens.weight"
            )
            weight.copy_(tensors.get_tensor(key))
    elif args.sidecar:
        from safetensors.torch import load_file

        state = load_file(args.sidecar, device="cpu")
        loaded = first.load_weights(
            (name.removeprefix("masked_embedding."), value)
            for name, value in state.items()
            if name.startswith("masked_embedding.")
        )
        first.validate_loaded_weights({"masked_embedding." + name for name in loaded})
    for head in heads.values():
        head.load_state_dict(first.state_dict())

    def route(hidden):
        ids = first.centroids(hidden).topk(args.top_k, dim=-1).indices
        selected = first.token_ordering.view(args.num_centroids, -1)[ids].flatten(1)
        return ids, selected

    metadata = {
        "vllm_source": vllm.__file__,
        "torch": str(torch.__version__),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "head_sources": {
            name: str(inspect.getfile(type(head))) for name, head in heads.items()
        },
        "settings": vars(args),
        "timing_scope": (
            "CUDA graph replay, per invocation; "
            "fixed or explicitly unrolled rotating inputs"
        ),
    }
    print("METADATA", json.dumps(metadata, default=str), flush=True)
    print("batch,pool,stage,backend,median_us,min_us,max_us", flush=True)
    rows = []
    with torch.inference_mode():
        for batch in args.batch_sizes:
            pool = [
                torch.randn(
                    batch, args.hidden_size, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(max(args.input_pool_sizes))
            ]
            routes = [route(hidden) for hidden in pool]
            selected = [value[1] for value in routes]
            sparse_logits = [
                selected_logits(weight, sel, hidden)
                for sel, hidden in zip(selected, pool)
            ]
            checks = {}
            for name, head in heads.items():
                errors, agreements = [], []
                for index in range(min(3, len(pool))):
                    logits, actual_selected = head._select_and_score(
                        pool[index], weight
                    )
                    torch.testing.assert_close(
                        actual_selected, selected[index], rtol=0, atol=0
                    )
                    reference = (
                        weight[actual_selected].float()
                        * pool[index][:, None, :].float()
                    ).sum(-1)
                    torch.testing.assert_close(
                        logits.float(),
                        reference.to(logits.dtype).float(),
                        atol=0.125,
                        rtol=0.01,
                    )
                    predicted = head.get_top_tokens(pool[index], weight)
                    torch.testing.assert_close(
                        predicted,
                        actual_selected.gather(
                            -1, logits.argmax(-1, keepdim=True)
                        ).squeeze(-1),
                        rtol=0,
                        atol=0,
                    )
                    errors.append((logits.float() - reference).abs().max().item())
                    agreements.append(
                        (
                            predicted
                            == selected[index]
                            .gather(-1, sparse_logits[index].argmax(-1, keepdim=True))
                            .squeeze(-1)
                        )
                        .float()
                        .mean()
                        .item()
                    )
                checks[name] = {
                    "max_abs_logit_error": max(errors),
                    "argmax_agreement_with_compiled": agreements,
                }
            functions = {}
            if not args.skip_dense:
                functions[("full", "dense")] = lambda i, pool=pool: F.linear(
                    pool[i], weight
                ).argmax(-1)
            for name, head in heads.items():
                functions[("full", name)] = lambda i, head=head, pool=pool: (
                    head.get_top_tokens(pool[i], weight)
                )
            if args.report_stages:
                functions[("routing", "shared")] = lambda i, pool=pool: route(pool[i])
                functions[("argmax", "shared")] = (
                    lambda i, selected=selected, sparse_logits=sparse_logits: (
                        selected[i]
                        .gather(-1, sparse_logits[i].argmax(-1, keepdim=True))
                        .squeeze(-1)
                    )
                )
                for name in heads:
                    if name == "compiled":
                        functions[("scoring", name)] = (
                            lambda i, selected=selected, pool=pool: selected_logits(
                                weight, selected[i], pool[i]
                            )
                        )
                    else:

                        def upstream_score(
                            i, selected=selected, batch=batch, pool=pool
                        ):
                            embeddings = weight[selected[i].reshape(-1)].view(
                                batch, first.num_selected, args.hidden_size
                            )
                            return torch.einsum("td,tsd->ts", pool[i], embeddings)

                        functions[("scoring", name)] = upstream_score
                # The isolated scorers must reproduce their actual head outputs.
                for name, head in heads.items():
                    actual, _ = head._select_and_score(pool[0], weight)
                    torch.testing.assert_close(
                        functions[("scoring", name)](0), actual, rtol=0, atol=0
                    )
            for fn in functions.values():
                for index in range(len(pool)):
                    fn(index)
            torch.accelerator.synchronize()
            for pool_size in args.input_pool_sizes:
                timings = {key: [] for key in functions}
                for repeat in range(args.repeats):
                    keys = list(functions)
                    shift = repeat % len(keys)
                    keys = keys[shift:] + keys[:shift]
                    if repeat % 2:
                        keys.reverse()
                    for key in keys:
                        fn = functions[key]

                        def invoke(fn=fn, pool_size=pool_size):
                            for index in range(pool_size):
                                fn(index)

                        elapsed = triton.testing.do_bench_cudagraph(
                            invoke, rep=args.rep_ms
                        )
                        timings[key].append(elapsed * 1000 / pool_size)
                for (stage, name), samples in timings.items():
                    median = statistics.median(samples)
                    row = {
                        "batch": batch,
                        "input_pool_size": pool_size,
                        "stage": stage,
                        "backend": name,
                        "median_us": median,
                        "samples_us": samples,
                        "correctness": checks.get(name),
                    }
                    rows.append(row)
                    print(
                        f"{batch},{pool_size},{stage},{name},{median:.3f},{min(samples):.3f},{max(samples):.3f}",
                        flush=True,
                    )
                if args.json_output:
                    args.json_output.write_text(
                        json.dumps(
                            {"metadata": metadata, "results": rows},
                            default=str,
                            indent=2,
                        )
                    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
