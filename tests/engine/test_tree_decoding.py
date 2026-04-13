#!/usr/bin/env python3
"""
Tree Decoding 测试脚本

这个脚本测试vLLM的tree decoding功能是否正常工作。
Tree decoding是一种通过创建多个分支来探索不同生成路径的技术。
"""

# from transformers import ProcessorMixin
import torch
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import TreeSearchParams


def test_tree_decoding():
    """测试tree decoding功能"""
    print("=== 开始Tree Decoding测试 ===")
    
    # 创建支持tree decoding的采样参数
    tree_config = TreeSearchParams(
        enable_tree_search=True,
        entropy_threshold=1.0,
        branching_factor=3,
        max_tree_depth=2
    )
    
    sampling_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
        tree_search_params=tree_config
    )
    
    print(f"Tree decoding配置:")
    print(f"  - 启用tree search: {tree_config.enable_tree_search}")
    print(f"  - 熵阈值: {tree_config.entropy_threshold}")
    print(f"  - 分支因子: {tree_config.branching_factor}")
    print(f"  - 最大树深度: {tree_config.max_tree_depth}")
    
    # 创建LLM实例
    model_path = "/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct"
    print("\n正在加载模型...")
    try:
        llm = LLM(
            model=model_path,
            dtype="float16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.8,
            # enable_chunked_prefill=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return False
    
    # 测试提示词
    test_prompts = [
        "What is the meaning of life?",
        # "In a galaxy far far away",
    ]
    
    print(f"\n准备测试 {len(test_prompts)} 个提示词...")
    
    all_success = True
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1}: '{prompt}' ---")
        
        # 生成文本
        outputs = llm.generate(prompt, sampling_params)
        
        if outputs and len(outputs) > 0:
            seq_map = {output.seq_id: output for output in outputs[0].outputs}
            leaf_outputs = [output for output in outputs[0].outputs if output.is_leaf]
            
            for leaf_out in leaf_outputs:
                # Traverse up to collect texts and ids
                path_texts = []
                path_ids = []
                current = leaf_out
                while current is not None:
                    path_texts.append(current.tree_text)
                    path_ids.append(list(current.tree_ids))
                    if current.parent_seq_id is not None and current.parent_seq_id in seq_map:
                        current = seq_map[current.parent_seq_id]
                    else:
                        current = None

                # The path gives leaf to root, so we reverse it
                full_text = "".join(reversed(path_texts))
                full_ids = []
                for ids in reversed(path_ids):
                    full_ids.extend(ids)
                decoded_from_ids = tokenizer.decode(full_ids, skip_special_tokens=False)

                print(f"\n--- 序列 ID: {leaf_out.seq_id} | 父节点 ID: {leaf_out.parent_seq_id} | 深度: {leaf_out.tree_depth} (叶子节点) ---")
                print(f"tree_text 拼接: {full_text!r}")
                print(f"tree_ids  解码: {decoded_from_ids!r}")
                match = full_text == decoded_from_ids
                print(f"一致性验证: {'✓ 一致' if match else '✗ 不一致'}")
                if not match:
                    all_success = False
                
            # 检查是否成功生成
            if leaf_outputs:
                print("\n✓ 成功生成叶子节点!")
            else:
                print("✗ 生成失败: 没有找到叶子节点")
                all_success = False
        else:
            print("✗ 生成失败: 无输出")
            all_success = False
            
    return all_success


def test_tree_decoding_with_comparison():
    """对比测试：启用和禁用tree decoding的效果"""
    print("\n=== Tree Decoding对比测试 ===")
    
    # 禁用tree decoding的配置
    normal_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
        tree_search_params=TreeSearchParams(enable_tree_search=False)
    )
    
    # 启用tree decoding的配置
    tree_params = SamplingParams(
        temperature=0.8,
        max_tokens=50,
        tree_search_params=TreeSearchParams(
            enable_tree_search=True,
            entropy_threshold=1.0,
            branching_factor=2,
            max_tree_depth=2
        )
    )
    
    test_prompt = "The key to happiness is"
    
    try:
        llm = LLM(
            model="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
            dtype="float16",
            tensor_parallel_size=1,
        )
        
        print(f"测试提示词: '{test_prompt}'")
        
        # 正常生成
        print("\n--- 正常生成 (Tree Decoding 禁用) ---")
        normal_outputs = llm.generate(test_prompt, normal_params)
        normal_text = normal_outputs[0].outputs[0].text
        print(f"结果: {normal_text}")
        
        # Tree decoding生成
        print("\n--- Tree Decoding 生成 ---")
        tree_outputs = llm.generate(test_prompt, tree_params)
        tree_text = tree_outputs[0].outputs[0].text
        print(f"结果: {tree_text}")
        
        print(f"\n对比结果:")
        print(f"正常生成长度: {len(normal_text)} 字符")
        print(f"Tree decoding长度: {len(tree_text)} 字符")
        print(f"内容差异: {'有' if normal_text != tree_text else '无'}")
        
        return True
        
    except Exception as e:
        print(f"对比测试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Tree Decoding 测试脚本")
    parser.add_argument("--model", type=str, default="/inspire/hdd/global_public/public_models/Qwen/Qwen2.5-7B-Instruct",
                       help="要测试的模型名称")
    # parser.add_argument("--compare", action="store_true",
    #                    help="运行对比测试")
    
    args = parser.parse_args()
    
    print("VLLM Tree Decoding 功能测试")
    print("=" * 50)
    
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        print(f"CUDA可用，设备数量: {torch.cuda.device_count()}")
        print(f"当前设备: {torch.cuda.current_device()}")
    else:
        print("警告: CUDA不可用，测试可能运行缓慢")
    
    success = True
    
    # if args.compare:
    #     success = test_tree_decoding_with_comparison()
    # else:
    #     success = test_tree_decoding()
    success = test_tree_decoding()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过!")
    else:
        print("✗ 部分测试失败!")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())