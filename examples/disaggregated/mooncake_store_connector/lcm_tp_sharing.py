# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate cross-TP Store hits and decode writeback with a real model.

Requires an existing Mooncake Store configured by MOONCAKE_CONFIG_PATH.
Run with enough visible GPUs for max(--tp-sizes), for example:

    python lcm_tp_sharing.py --model facebook/opt-125m --tp-sizes 4 3 \
        --result-dir /path/to/results

Each stage starts a fresh engine, so hits cannot come from local prefix cache.
EOS is ignored to exercise a fixed decode-writeback boundary; this is a
cache-correctness fixture, not a natural-length generation benchmark.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path


def run_stage(args):
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    inputs = json.loads(args.input.read_text())
    kwargs = {}
    if args.stage != "reference":
        kwargs["kv_transfer_config"] = KVTransferConfig(
            kv_connector="MooncakeStoreConnector",
            kv_role="kv_consumer" if args.stage == "decode" else "kv_both",
            kv_connector_extra_config={
                "enable_store_tp_lcm": True,
                "tp_sizes": args.tp_sizes,
                "cache_prefix": args.cache_prefix,
                "lookup_rpc_port": args.lookup_port,
                "save_decode_cache": args.stage == "decode",
            },
        )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.local_tp,
        enforce_eager=True,
        dtype="float16",
        max_model_len=512,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        block_size=16,
        gpu_memory_utilization=0.25,
        enable_prefix_caching=True,
        disable_custom_all_reduce=True,
        **kwargs,
    )
    output = llm.generate(
        [{"prompt_token_ids": inputs["prompt_token_ids"]}],
        SamplingParams(temperature=0, max_tokens=inputs["max_tokens"], ignore_eos=True),
    )[0]
    args.output.write_text(
        json.dumps(
            {
                "cached_tokens": output.num_cached_tokens,
                "token_ids": list(output.outputs[0].token_ids),
                "tp_size": args.local_tp,
                "prompt_tokens": len(inputs["prompt_token_ids"]),
                "max_tokens": inputs["max_tokens"],
                "ignore_eos": True,
                "finish_reason": output.outputs[0].finish_reason,
            },
            indent=2,
        )
        + "\n"
    )
    llm.llm_engine.engine_core.shutdown()


def run_matrix(args):
    from transformers import AutoTokenizer

    args.result_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    seed = tokenizer.encode(
        "Explain how a computer stores and retrieves information. ",
        add_special_tokens=False,
    )
    prompt = (seed * math.ceil(129 / len(seed)))[:129]
    env = dict(os.environ, PYTHONHASHSEED="0")
    script = Path(__file__).resolve()
    results = []
    run_id = uuid.uuid4().hex

    def stage(label, role, tp, tokens, max_tokens, namespace):
        input_path = args.result_dir / f"{label}-input.json"
        output_path = args.result_dir / f"{label}-output.json"
        input_path.write_text(
            json.dumps({"prompt_token_ids": tokens, "max_tokens": max_tokens})
        )
        command = [
            sys.executable,
            str(script),
            "--stage",
            role,
            "--model",
            args.model,
            "--local-tp",
            str(tp),
            "--tp-sizes",
            *map(str, args.tp_sizes),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--cache-prefix",
            namespace,
            "--lookup-port",
            str(args.lookup_port),
        ]
        with (args.result_dir / f"{label}.log").open("w") as log:
            subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=1200,
            )
        result = json.loads(output_path.read_text())
        assert len(result["token_ids"]) == max_tokens, label
        assert result["finish_reason"] == "length", label
        return result

    for producer_tp, decode_tp in (args.tp_sizes, args.tp_sizes[::-1]):
        label = f"tp{producer_tp}-to-tp{decode_tp}"
        namespace = f"lcm-{run_id}-{label}"
        cold = stage(label + "-cold", "reference", decode_tp, prompt, 48, namespace)
        producer = stage(
            label + "-prefill", "prefill", producer_tp, prompt, 1, namespace
        )
        decode = stage(label + "-decode", "decode", decode_tp, prompt, 48, namespace)
        assert producer["cached_tokens"] == 0, producer
        assert decode["cached_tokens"] == 128, decode
        assert decode["token_ids"] == cold["token_ids"], label

        extended = prompt + decode["token_ids"][:32]
        extended_cold = stage(
            label + "-extended-cold", "reference", producer_tp, extended, 16, namespace
        )
        reloaded = stage(
            label + "-writeback", "prefill", producer_tp, extended, 16, namespace
        )
        assert reloaded["cached_tokens"] == 160, reloaded
        assert reloaded["token_ids"] == extended_cold["token_ids"], label
        results.append(
            {
                "direction": label,
                "store_tp": math.lcm(*args.tp_sizes),
                "decode_cached_tokens": decode["cached_tokens"],
                "writeback_cached_tokens": reloaded["cached_tokens"],
                "decode_reference_equal": True,
                "writeback_reference_equal": True,
            }
        )
        print(json.dumps(results[-1]), flush=True)
    (args.result_dir / "results.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "tp_sizes": args.tp_sizes,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/opt-125m")
    parser.add_argument("--tp-sizes", nargs=2, type=int, default=[4, 3])
    parser.add_argument("--result-dir", type=Path)
    parser.add_argument("--stage", choices=["reference", "prefill", "decode"])
    parser.add_argument("--local-tp", type=int)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-prefix", default="")
    parser.add_argument("--lookup-port", type=int, default=57301)
    args = parser.parse_args()
    if any(tp <= 0 for tp in args.tp_sizes):
        parser.error("--tp-sizes must contain positive integers")
    if args.stage:
        run_stage(args)
    else:
        if args.result_dir is None:
            parser.error("--result-dir is required")
        run_matrix(args)


if __name__ == "__main__":
    main()
