#!/usr/bin/env python3
# 基于vllm/tests/v1/test_async_llm_dp.py创建的离线测试脚本

import asyncio
import os
import time
from contextlib import ExitStack
from typing import Optional, List

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import PromptType
from vllm.platforms import current_platform
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.core_client import DPAsyncMPClient


async def generate(engine: AsyncLLM,
                   request_id: str,
                   prompt: PromptType,
                   output_kind: RequestOutputKind,
                   max_tokens: int,
                   prompt_logprobs: Optional[int] = None) -> tuple[int, str]:
    # 确保generate不会太快完成，以便测试取消功能
    await asyncio.sleep(0.2)

    count = 0
    sampling_params = SamplingParams(max_tokens=max_tokens,
                                     ignore_eos=True,
                                     output_kind=output_kind,
                                     temperature=0,
                                     prompt_logprobs=prompt_logprobs)
    async for out in engine.generate(request_id=request_id,
                                     prompt=prompt,
                                     sampling_params=sampling_params):

        num_tokens = len(out.outputs[0].token_ids)
        if output_kind == RequestOutputKind.DELTA:
            count += num_tokens
        else:
            count = num_tokens

        await asyncio.sleep(0.)

    return count, request_id


async def test_load(output_kind: RequestOutputKind, num_requests: int = 10):
    print(f"开始测试，输出类型: {output_kind}，请求数量: {num_requests}")
    start_time = time.time()

    with ExitStack() as after:
        prompt = "This is a test of data parallel"

        # 创建引擎参数
        engine_args = AsyncEngineArgs(
            model="ibm-research/PowerMoE-3b",
            enforce_eager=True,
            disable_log_requests=True,
            tensor_parallel_size=int(os.getenv("TP_SIZE", 1)),
            data_parallel_size=int(os.getenv("DP_SIZE", 2)),
        )

        # 检查平台是否支持V1
        if not current_platform.supports_v1(engine_args.create_model_config()):
            print("错误: 当前平台不支持V1，无法继续测试")
            return

        print("正在初始化引擎...")
        engine = AsyncLLM.from_engine_args(engine_args)
        after.callback(engine.shutdown)

        NUM_EXPECTED_TOKENS = 10
        request_ids = [f"request-{i}" for i in range(num_requests)]

        # 创建并发请求
        print(f"创建{num_requests}个并发请求...")
        tasks: List[asyncio.Task] = []
        for request_id in request_ids:
            tasks.append(
                asyncio.create_task(
                    generate(engine, request_id, prompt, output_kind,
                             NUM_EXPECTED_TOKENS)))

        # 等待所有请求完成并确认结果
        print("等待请求完成...")
        done, pending = await asyncio.wait(tasks,
                                          return_when=asyncio.FIRST_EXCEPTION)
        
        # 取消未完成的任务
        for task in pending:
            task.cancel()
        
        # 检查完成的任务
        success_count = 0
        for task in done:
            try:
                num_generated_tokens, request_id = await task
                if num_generated_tokens == NUM_EXPECTED_TOKENS:
                    success_count += 1
                else:
                    print(f"警告: {request_id} 生成了 {num_generated_tokens} 个token，"
                          f"但期望值是 {NUM_EXPECTED_TOKENS}")
            except Exception as e:
                print(f"错误: 任务执行失败: {e}")

        print(f"成功完成的请求: {success_count}/{len(done)}")

        # 检查是否有未完成的请求
        if engine.output_processor.has_unfinished_requests():
            print("警告: 引擎仍有未完成的请求")
        else:
            print("所有请求已完成")

        # 检查引擎内部状态(注意:这是内部实现，可能会变化)
        core_client: DPAsyncMPClient = engine.engine_core
        # 引擎只在每N步同步停止，所以这里允许一些时间
        for i in range(10):
            if not core_client.engines_running:
                print(f"引擎已停止运行(检查次数: {i+1})")
                break
            print(f"等待引擎停止运行(检查次数: {i+1})...")
            await asyncio.sleep(0.5)

        if not core_client.engines_running:
            print("引擎已停止运行")
        else:
            print("警告: 引擎仍在运行")

        if not core_client.reqs_in_flight:
            print("没有未完成的请求")
        else:
            print(f"警告: 仍有 {len(core_client.reqs_in_flight)} 个请求在处理中")

    duration = time.time() - start_time
    print(f"测试完成，耗时: {duration:.2f}秒")


async def run_all_tests():
    print("=== 测试 DELTA 输出模式 ===")
    await test_load(RequestOutputKind.DELTA, num_requests=10)
    print("\n=== 测试 FINAL_ONLY 输出模式 ===")
    await test_load(RequestOutputKind.FINAL_ONLY, num_requests=10)


if __name__ == "__main__":
    print("开始离线测试数据并行异步LLM...")
    
    # 可以根据需要修改环境变量以调整并行设置
    # os.environ["TP_SIZE"] = "1"  # 张量并行大小
    # os.environ["DP_SIZE"] = "2"  # 数据并行大小
    
    # 运行测试
    asyncio.run(run_all_tests())