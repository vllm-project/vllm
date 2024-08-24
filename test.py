from vllm import LLM
# from fait360brew.inference.vllm_patch_utils import patch_vllm_compressed_tensors_config_get_schema
# patch_vllm_compressed_tensors_config_get_schema()

# model = LLM("/shared/public/models/quantized_model/Meta-Llama-3-8B-Instruct-W8A8-test", tensor_parallel_size=1)
# model = LLM("/shared/public/models/quantized_model/Mixtral-8x22B-Instruct-v0.1-W8A8", tensor_parallel_size=8, max_num_seqs=64)  # , max_num_batched_tokens= 128 * 4201
model = LLM("/shared/public/models/Mixtral-8x22B-Instruct-v0.1", tensor_parallel_size=8, max_num_seqs=64) # ,  max_num_batched_tokens= 128 * 4201

# model = LLM("/shared/public/models/Meta-Llama-3-70B-Instruct/", tensor_parallel_size=8, max_num_seqs=32, max_num_batched_tokens= 64 * 2048)

# /shared/public/models/Meta-Llama-3-8B-Instruct
# /shared/public/models/quantized_model/Meta-Llama-3-8B-Instruct-W8A8-test
# model = LLM("/shared/public/models/Meta-Llama-3-8B-Instruct", tensor_parallel_size=8)
# model = LLM("/shared/public/models/quantized_model/Meta-Llama-3-8B-Instruct-W8A8-test", tensor_parallel_size=1, distributed_executor_backend="ray")

for i in range(3):
    qq = model.generate(["how is vllm  " * 700][:4096] * 2)
    print(len(qq[0].prompt_token_ids))
    print(qq[0].metrics.finished_time - qq[0].metrics.arrival_time)


# #llm is a vllm.LLM object
# import gc
# import torch
# from vllm.distributed.parallel_state import destroy_model_parallel
# import os

# #avoid huggingface/tokenizers process dead lock
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# destroy_model_parallel()
# #del a vllm.executor.ray_gpu_executor.RayGPUExecutor object
# del model.llm_engine.model_executor
# del model
# gc.collect()
# torch.cuda.empty_cache()
# import ray
# ray.shutdown()


# model = LLM("/shared/public/models/Meta-Llama-3-8B-Instruct", tensor_parallel_size=1)