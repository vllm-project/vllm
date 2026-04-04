import time
import argparse
import numpy as np
from vllm import LLM, SamplingParams
from vllm.sampling_params import TreeSearchParams

def main():
    parser = argparse.ArgumentParser(description="Benchmark Tree Decoding vs Normal Decoding in vLLM")
    parser.add_argument("--model", type=str, default="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
                        help="Model to use for benchmarking")
    parser.add_argument("--num-prompts-mc", type=int, default=100,
                        help="Number of prompts to process in batch")
    parser.add_argument("--num-prompts-hs", type=int, default=100,
                        help="Number of prompts to process in batch")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate per prompt")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size")
    
    # Tree decoding hyperparams
    parser.add_argument("--entropy-threshold", type=float, default=0.5)
    parser.add_argument("--branching-factor", type=int, default=3)
    parser.add_argument("--max-tree-depth", type=int, default=3)
    # parser.add_argument("--tau-importance", type=float, default=0.01)
    
    args = parser.parse_args()

    # 1. 准备测试用的 prompts
    # 找一些涵盖不同场景的prompt，并且复制多次以测试批量处理性能
    base_prompts = [
        "Explain the process of photosynthesis in detail.",
        "Write a Python script to scrape a website.",
        "What are the main causes of the French Revolution?",
        "Describe the architecture of a transformer model.",
        "Write a short story about a time traveler."
    ]
    prompts_mc = (base_prompts * (args.num_prompts_mc // len(base_prompts) + 1))[:args.num_prompts_mc]
    prompts_hs = (base_prompts * (args.num_prompts_hs // len(base_prompts) + 1))[:args.num_prompts_hs]
    
    # print(f"准备了 {len(prompts)} 个 prompt 用于批量测试。")

    # 2. 初始化 LLM Engine
    print(f"\n正在加载模型 {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="float16",
        gpu_memory_utilization=0.9,
        enforce_eager=True  # 根据测试脚本，启用eager模式
    )

    # ---------- Normal Decoding Benchmark ----------
    print("\n" + "="*50)
    print("开始测试: 不启用 Tree Decoding (Normal Decoding)")
    normal_params = SamplingParams(
        temperature=0.8,
        max_tokens=args.max_tokens,
        n=27,
        # tree_search_params=TreeSearchParams(enable_tree_search=False)
    )
    
    # Warmup
    print("预热模型...")
    llm.generate(["Warmup prompt"], normal_params, use_tqdm=False)
    
    start_time = time.time()
    normal_outputs = llm.generate(prompts_mc, normal_params, use_tqdm=True)
    normal_time = time.time() - start_time
    
    # 统计普通生成的 tokens
    normal_total_tokens = sum(len(out.outputs[0].token_ids) for out in normal_outputs)
    normal_total_tokens = 0
    for out in normal_outputs:
        for output in out.outputs:
            # print(out.outputs[0].token_ids)
            normal_total_tokens += len(output.token_ids)
    normal_tps = normal_total_tokens / normal_time
    normal_qps = len(prompts_mc) / normal_time
    
    print(f"\n[Normal Decoding 结果]")
    print(f"总耗时: {normal_time:.2f} 秒")
    print(f"总生成Tokens: {normal_total_tokens}")
    print(f"吞吐量 (Tokens/s): {normal_tps:.2f}")
    print(f"吞吐量 (Queries/s): {normal_qps:.2f}")

    # ---------- Tree Decoding Benchmark ----------
    print("\n" + "="*50)
    print("开始测试: 启用 Tree Decoding")
    print(f"参数: entropy_threshold={args.entropy_threshold}, branching_factor={args.branching_factor}, "
          f"max_tree_depth={args.max_tree_depth}")
          
    tree_params = SamplingParams(
        temperature=0.8,
        max_tokens=args.max_tokens,
        tree_search_params=TreeSearchParams(
            enable_tree_search=True,
            entropy_threshold=args.entropy_threshold,
            branching_factor=args.branching_factor,
            max_tree_depth=args.max_tree_depth,
            # tau_importance=args.tau_importance
        )
    )
    
    start_time = time.time()
    tree_outputs = llm.generate(prompts_hs, tree_params, use_tqdm=True)
    tree_time = time.time() - start_time
    
    # 统计 Tree Decoding 生成的 tokens 和叶子节点数
    tree_total_tokens = 0
    total_leaf_nodes = 0
    for request_output in tree_outputs:
        leaf_nodes = [out for out in request_output.outputs if hasattr(out, 'is_leaf') and out.is_leaf]
        total_leaf_nodes += len(leaf_nodes) if leaf_nodes else 1
        
        # 简单计算：将所有分支生成的token长度加起来
        # 注意: 如果有共享的前缀会被重复计算，这里主要用来衡量输出的丰富度
        tree_total_tokens += sum(len(out.tree_ids) for out in request_output.outputs)

    tree_tps = tree_total_tokens / tree_time
    tree_qps = len(prompts_hs) / tree_time
    
    print(f"\n[Tree Decoding 结果]")
    print(f"总耗时: {tree_time:.2f} 秒")
    print(f"总生成Tokens (包含所有分支): {tree_total_tokens}")
    print(f"共生成叶子节点数: {total_leaf_nodes}")
    print(f"吞吐量 (Tokens/s): {tree_tps:.2f}")
    print(f"吞吐量 (Queries/s): {tree_qps:.2f}")

    # ---------- 对比总结 ----------
    print("\n" + "="*50)
    print("性能对比总结")
    print("="*50)
    # print(f"批量查询数量: {len(prompts)}")
    print(f"请求最大序列长度: {args.max_tokens}")
    print("-" * 50)
    print(f"{'指标':<25} | {'Normal':<10} | {'Tree Decoding':<15} | {'比变 (Tree/Normal)':<15}")
    print("-" * 50)
    print(f"{'耗时 (秒)':<25} | {normal_time:<10.2f} | {tree_time:<15.2f} | {tree_time/normal_time:<15.2f}x")
    print(f"{'Queries/s (整体请求吞吐)':<21} | {normal_qps:<10.2f} | {tree_qps:<15.2f} | {tree_qps/normal_qps:<15.2f}x")
    print(f"{'Tokens/s (输出序列吞吐)':<22} | {normal_tps:<10.2f} | {tree_tps:<15.2f} | {tree_tps/normal_tps:<15.2f}x")
    print(f"{'平均每个Request输出的序列数':<20} | {1:<10.2f} | {total_leaf_nodes/len(prompts_hs):<15.2f} | -")
    print("="*50)

if __name__ == "__main__":
    main()
