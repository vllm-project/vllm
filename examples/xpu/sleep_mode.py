# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio

import torch

from vllm import LLM, AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils.mem_constants import GiB_bytes


def fmt_mem_info() -> str:
    free_bytes, total_bytes = torch.xpu.mem_get_info()
    used_bytes = total_bytes - free_bytes
    gib = 1024**3
    return (
        f"free={free_bytes / gib:.2f} GiB, "
        f"used={used_bytes / gib:.2f} GiB, "
        f"total={total_bytes / gib:.2f} GiB"
    )


def run_end_to_end(model: str, prompt: str, max_tokens: int) -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available in this environment")

    free, total = torch.xpu.mem_get_info()
    used_bytes_baseline = total - free

    print("[1/8] Initial memory:", fmt_mem_info())
    llm = LLM(
        model=model,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=4096,
    )
    sampling_params = SamplingParams.from_optional(
        temperature=0,
        max_tokens=max_tokens,
    )

    print("[2/8] First generate")
    output = llm.generate(prompt, sampling_params)
    text1 = output[0].outputs[0].text
    print(f"generate output: {text1}")
    print("[3/8] Sleep level=1")
    llm.sleep(level=1)
    print("       After sleep:", fmt_mem_info())

    free_after_sleep, total = torch.xpu.mem_get_info()
    used_after_sleep = total - free_after_sleep - used_bytes_baseline
    print(f"       Approx used after sleep: {used_after_sleep / GiB_bytes:.2f} GiB")

    print("[4/8] Wake all + generate")
    llm.wake_up()
    output2 = llm.generate(prompt, sampling_params)
    text2 = output2[0].outputs[0].text
    if text1 != text2:
        raise RuntimeError("Mismatch after full wake_up")

    print("[5/8] Sleep level=1 again")
    llm.sleep(level=1)

    print("[6/8] Wake weights only")
    llm.wake_up(tags=["weights"])
    free_after_weights, total = torch.xpu.mem_get_info()
    used_after_weights = total - free_after_weights - used_bytes_baseline
    print("       After waking weights:", fmt_mem_info())
    print(
        "       Approx used after weights wake: "
        f"{used_after_weights / GiB_bytes:.2f} GiB"
    )

    print("[7/8] Wake kv_cache + generate")
    llm.wake_up(tags=["kv_cache"])
    output3 = llm.generate(prompt, sampling_params)
    text3 = output3[0].outputs[0].text
    if text1 != text3:
        raise RuntimeError("Mismatch after staged wake_up (weights + kv_cache)")

    print("[8/8] Sleep+Wake once more + generate")
    llm.sleep(level=1)
    llm.wake_up()
    output4 = llm.generate(prompt, sampling_params)
    text4 = output4[0].outputs[0].text
    if text1 != text4:
        raise RuntimeError("Mismatch after second sleep/wake cycle")

    print("[done] end_to_end flow complete")
    print("       Output consistency check passed")
    print("       Sample output:", repr(text1))


def run_deep_sleep(model: str, prompt: str, max_tokens: int) -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available in this environment")

    free, total = torch.xpu.mem_get_info()
    used_bytes_baseline = total - free

    print("[1/6] Initial memory:", fmt_mem_info())
    llm = LLM(
        model=model,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=4096,
        tensor_parallel_size=2,
    )
    sampling_params = SamplingParams.from_optional(
        temperature=0,
        max_tokens=max_tokens,
    )

    print("[2/6] First generate")
    output = llm.generate(prompt, sampling_params)
    text1 = output[0].outputs[0].text

    print("[3/6] Deep sleep level=2")
    llm.sleep(level=2)
    print("       After deep sleep:", fmt_mem_info())

    free_after_sleep, total = torch.xpu.mem_get_info()
    used_after_sleep = total - free_after_sleep - used_bytes_baseline
    print(
        f"       Approx used after deep sleep: {used_after_sleep / GiB_bytes:.2f} GiB"
    )

    print("[4/6] Wake weights + reload")
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    print("       After waking weights:", fmt_mem_info())

    print("[5/6] Wake kv_cache + generate")
    llm.wake_up(tags=["kv_cache"])
    output2 = llm.generate(prompt, sampling_params)
    text2 = output2[0].outputs[0].text
    if text1 != text2:
        raise RuntimeError("Mismatch after deep-sleep wake sequence")

    print("[6/6] Sleep+Wake once more + generate")
    llm.sleep(level=2)
    llm.wake_up(tags=["weights"])
    llm.collective_rpc("reload_weights")
    llm.wake_up(tags=["kv_cache"])
    output3 = llm.generate(prompt, sampling_params)
    text3 = output3[0].outputs[0].text
    if text1 != text3:
        raise RuntimeError("Mismatch after second deep-sleep wake sequence")

    print("[done] deep_sleep flow complete")
    print("       Output consistency check passed")
    print("       Sample output:", repr(text1))


async def run_deep_sleep_async(model: str, prompt: str, max_tokens: int) -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available in this environment")

    free, total = torch.xpu.mem_get_info()
    used_bytes_baseline = total - free

    print("[1/6] Initial memory:", fmt_mem_info())
    engine_args = AsyncEngineArgs(
        model=model,
        enable_sleep_mode=True,
        enforce_eager=True,
        max_model_len=4096,
    )
    llm = AsyncLLMEngine.from_engine_args(engine_args)
    sampling_params = SamplingParams.from_optional(
        temperature=0,
        max_tokens=max_tokens,
    )

    print("[2/6] First async generate")
    output_stream = llm.generate(prompt, sampling_params, request_id="sleep-example-1")
    output1 = None
    async for item in output_stream:
        output1 = item
    if output1 is None:
        raise RuntimeError("No output received from first async generate")
    text1 = output1.outputs[0].text

    print("[3/6] Deep sleep level=2")
    await llm.sleep(level=2)
    print("       After deep sleep:", fmt_mem_info())

    free_after_sleep, total = torch.xpu.mem_get_info()
    used_after_sleep = total - free_after_sleep - used_bytes_baseline
    print(
        f"       Approx used after deep sleep: {used_after_sleep / GiB_bytes:.2f} GiB"
    )

    print("[4/6] Wake weights + reload")
    await llm.wake_up(tags=["weights"])
    await llm.collective_rpc("reload_weights")
    print("       After waking weights:", fmt_mem_info())

    print("[5/6] Wake kv_cache + async generate")
    await llm.wake_up(tags=["kv_cache"])
    output_stream2 = llm.generate(prompt, sampling_params, request_id="sleep-example-2")
    output2 = None
    async for item in output_stream2:
        output2 = item
    if output2 is None:
        raise RuntimeError("No output received from second async generate")
    text2 = output2.outputs[0].text
    if text1 != text2:
        raise RuntimeError("Mismatch after async deep-sleep wake sequence")

    print("[6/6] Sleep+Wake once more + async generate")
    await llm.sleep(level=2)
    await llm.wake_up(tags=["weights"])
    await llm.collective_rpc("reload_weights")
    await llm.wake_up(tags=["kv_cache"])
    output_stream3 = llm.generate(prompt, sampling_params, request_id="sleep-example-3")
    output3 = None
    async for item in output_stream3:
        output3 = item
    if output3 is None:
        raise RuntimeError("No output received from third async generate")
    text3 = output3.outputs[0].text
    if text1 != text3:
        raise RuntimeError("Mismatch after second async deep-sleep wake sequence")

    print("[done] deep_sleep_async flow complete")
    print("       Output consistency check passed")
    print("       Sample output:", repr(text1))


def run_normal_inference(model: str, prompt: str, max_tokens: int) -> None:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        raise RuntimeError("XPU is not available in this environment")

    print("[1/2] Initial memory:", fmt_mem_info())
    llm = LLM(
        model=model,
        enable_sleep_mode=False,
        enforce_eager=True,
        max_model_len=4096,
    )
    sampling_params = SamplingParams.from_optional(
        temperature=0,
        max_tokens=max_tokens,
    )

    print("[2/2] Normal inference generate")
    output = llm.generate(prompt, sampling_params)
    text = output[0].outputs[0].text

    print("[done] normal_inference flow complete")
    print("       Output:", repr(text))
    print("       Final memory:", fmt_mem_info())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XPU sleep-mode examples based on "
        "test_end_to_end/deep_sleep/deep_sleep_async."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "end_to_end",
            "deep_sleep",
            "deep_sleep_async",
            "normal_inference",
            "all",
        ],
        default="end_to_end",
        help="Which flow to run",
    )
    parser.add_argument(
        "--model",
        default="hmellor/tiny-random-LlamaForCausalLM",
        help="Model name or local path",
    )
    parser.add_argument(
        "--prompt",
        default="How are you?",
        help="Prompt text",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10,
        help="Max generated tokens",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode in ("end_to_end", "all"):
        run_end_to_end(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )

    if args.mode in ("deep_sleep", "all"):
        run_deep_sleep(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )

    if args.mode in ("deep_sleep_async", "all"):
        asyncio.run(
            run_deep_sleep_async(
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
            )
        )

    if args.mode in ("normal_inference", "all"):
        run_normal_inference(
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
        )
