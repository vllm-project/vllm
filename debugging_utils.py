
def who_am_i():
    import traceback
    stack = traceback.extract_stack()
    print(stack)
    filename, codeline, funcName, text = stack[-2]

    # import inspect
    # this_function_name = inspect.currentframe().f_code.co_name
    # print(this_function_name)

    return funcName


import gc
import torch
# print_memory_usage func

_MB = 1 << 20


def print_memory_usage(info: str, sync: bool = True, empty_cache: bool = False, collect: bool = False):
    if sync:
        torch.cuda.synchronize()
    if empty_cache:
        torch.cuda.empty_cache()
    if collect:
        gc.collect()
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    print(
        f"{info}: "
        f"use_gpu_memory: {(total_gpu_memory - free_gpu_memory) / _MB:.4f} MB, "
        f"free_gpu_memory: {free_gpu_memory / _MB:.4f} MB, "
        f"total_gpu_memory: {total_gpu_memory / _MB:.4f} MB"
    )


def print_shape(name:str, cache:torch.Tensor):
    print(f'{name}: {cache.shape}')



def calculate_memory_usage_example():
    import math

    layer = 32
    max_token_count = 512
    dimension = 4096
    type_size = 2  # type_size = bfloat16 / 8 = 2
    batch_size = 64
    model_size = 7.72 * math.pow(10, 9)
    kv = 2

    sum = layer * dimension * batch_size * type_size * max_token_count * kv + model_size * type_size
    sum = sum * math.pow(10, -9)
    print(sum)

    # gpt2, dimension 768, model size 137M params, tensor type F32
    # (num_blocks, num_heads, head_size // x, block_size, x)
    # key_cache: Tensor(8177,12,8,16,8)
    num_blocks = 8177
    num_heads = 12
    head_size = 64  # dimension = num_heads*head_size
    # head_size // x : 8
    block_size = 16
    # x: 8

    # gpu cache shape is (2, 7823, 16, 32, 128), (layer=32, type_size = bfloat16 / 8 = 2)
    num_blocks = 7823
    num_heads = 32
    head_size = 128  # dimension = num_heads*head_size
    # head_size // x : 8
    block_size = 16
    # x: 8

    total = layer * type_size * num_blocks * block_size * (num_heads * head_size) * kv
    total = total * math.pow(10, -9)  # Byte to GigaByte
    print(total)
def calculate_memory_usage_example(layer, type_size, num_blocks, block_size , num_heads , head_size, kv):
    import math

    total = layer * type_size * num_blocks * block_size * (num_heads * head_size) * kv
    total = total * math.pow(10, -9)  # Byte to GigaByte
    print(total)