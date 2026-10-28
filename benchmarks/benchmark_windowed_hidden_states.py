# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check window capture against full capture, then measure paired short runs."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.distributed.aux_output_connector.hidden_states import HiddenStateCapturePlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft-model")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--window-start", type=int, default=64)
    parser.add_argument("--window-end", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--aux-layer-ids", default="")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--native-baseline", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    assert 0 <= args.window_start < args.window_end <= args.output_tokens
    spec = (
        None
        if not args.draft_model
        else {
            "method": "eagle3",
            "model": args.draft_model,
            "num_speculative_tokens": 3,
        }
    )
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        max_model_len=max(256, args.output_tokens + 160),
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=256,
        gpu_memory_utilization=0.45,
        enforce_eager=args.eager,
        enable_prefix_caching=False,
        async_scheduling=False,
        aux_output_config={"enable_hidden_states": not args.native_baseline},
        speculative_config=spec,
        logprobs_mode="raw_logprobs",
        seed=123,
    )
    prompts = [
        {"prompt_token_ids": ([100, 200, 300, 400] * 32)[: 128 - i]}
        for i in range(args.batch_size)
    ]

    def generate(mode, length=args.output_tokens, topk=0):
        params = []
        for i, prompt in enumerate(prompts):
            extra = None
            if mode != "off" and not (mode == "mixed" and i % 2):
                start, end = (
                    (0, length)
                    if mode == "full"
                    else (
                        (length + 16, length + 32)
                        if mode == "never"
                        else (args.window_start, args.window_end)
                    )
                )
                plan = HiddenStateCapturePlan.from_window(
                    "",
                    len(prompt["prompt_token_ids"]),
                    start,
                    end,
                    "benchmark",
                    min_rows=end - start,
                    logprob_topk=topk,
                    aux_layer_ids=tuple(
                        int(i) for i in args.aux_layer_ids.split(",") if i
                    ),
                )
                extra = {"hidden_state_capture": asdict(plan)}
            params.append(
                SamplingParams(
                    temperature=0,
                    max_tokens=length,
                    min_tokens=length,
                    ignore_eos=True,
                    detokenize=False,
                    extra_args=extra,
                    logprobs=topk or None,
                )
            )
        return llm.generate(prompts, params, use_tqdm=False)

    if args.native_baseline:
        generate("off")
        records = []
        for round_id in range(args.rounds):
            started = perf_counter()
            generate("off")
            elapsed = perf_counter() - started
            records.append(
                {
                    "round": round_id,
                    "mode": "native",
                    "seconds": elapsed,
                    "tokens_per_second": args.output_tokens * len(prompts) / elapsed,
                }
            )
        args.result.write_text(json.dumps({"records": records}, indent=2))
        print("NATIVE_BASELINE", json.dumps(records), flush=True)
        return

    off = generate("off")
    full = generate("full", topk=16)
    max_logprob_error = 0.0
    for mode in ("window", "never", "mixed"):
        outputs = generate(mode, topk=16)
        for i, (expected, actual, reference) in enumerate(zip(full, outputs, off)):
            assert actual.outputs[0].token_ids == reference.outputs[0].token_ids
            assert expected.outputs[0].token_ids == reference.outputs[0].token_ids
            result = actual.hidden_state_capture
            if mode == "never" or (mode == "mixed" and i % 2):
                assert result is None
                if mode == "never":
                    assert actual.hidden_capture_skip_reason == "ended_before_window"
                continue
            assert result is not None, actual.hidden_capture_skip_reason
            offset = len(prompts[i]["prompt_token_ids"]) - 1
            np.testing.assert_array_equal(
                result.hidden_positions,
                np.arange(offset + args.window_start, offset + args.window_end),
            )
            torch.testing.assert_close(
                result.hidden_states,
                expected.hidden_state_capture.hidden_states[
                    args.window_start : args.window_end
                ],
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                result.logprobs,
                expected.hidden_state_capture.logprobs[
                    args.window_start : args.window_end
                ],
                atol=0,
                rtol=0,
            )
            for row, values, ids in zip(
                actual.outputs[0].logprobs[args.window_start : args.window_end],
                result.logprobs,
                result.logprob_token_ids,
            ):
                expected_values = torch.tensor([row[t].logprob for t in ids.tolist()])
                torch.testing.assert_close(
                    values, expected_values, atol=2e-5, rtol=1e-5
                )
                max_logprob_error = max(
                    max_logprob_error, float((values - expected_values).abs().max())
                )
    correctness = {
        "tokens_unchanged": True,
        "full_window_hidden_max_error": 0,
        "max_logprob_error": max_logprob_error,
        "mixed_batch": True,
    }
    if args.reference:
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
            .to("cuda:0")
            .eval()
        )
        ids = prompts[0]["prompt_token_ids"] + full[0].outputs[0].token_ids[:-1]
        with torch.inference_mode():
            reference = model.model(torch.tensor([ids], device="cuda:0"))
        start = len(prompts[0]["prompt_token_ids"]) - 1
        actual = full[0].hidden_state_capture.hidden_states
        reference = reference.last_hidden_state[0, start:].cpu()
        torch.testing.assert_close(actual, reference, atol=0.05, rtol=0.01)
        correctness["hf_reference_max_error"] = float((actual - reference).abs().max())
        del model, reference
        torch.accelerator.empty_cache()
    print("CORRECTNESS_PASS", json.dumps(correctness), flush=True)
    del off, full

    if args.check_only:
        args.result.write_text(
            json.dumps(
                {"correctness": correctness, "args": vars(args)}, default=str, indent=2
            )
        )
        return

    records = []
    for mode in ("off", "full", "window", "never"):
        generate(mode)
    for round_id in range(args.rounds):
        modes = ("off", "full", "window", "never")
        if round_id % 2:
            modes = tuple(reversed(modes))
        for mode in modes:
            started = perf_counter()
            outputs = generate(mode)
            elapsed = perf_counter() - started
            copied = sum(
                o.hidden_state_capture.copied_bytes
                for o in outputs
                if o.hidden_state_capture is not None
            )
            record = {
                "round": round_id,
                "mode": mode,
                "seconds": elapsed,
                "tokens_per_second": args.output_tokens * len(prompts) / elapsed,
                "copied_bytes": copied,
            }
            records.append(record)
            print("MEASURE", json.dumps(record), flush=True)
    result = {
        "args": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "correctness": correctness,
        "records": records,
        "gpu": torch.cuda.get_device_name(),
        "runner_v2": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0"),
    }
    args.result.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
