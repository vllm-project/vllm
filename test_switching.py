import vllm

import torch_xla.debug.profiler as xp

from vllm.lora.request import LoRARequest

lora_paths = ["/mnt/ssd0/adapters/1", "/mnt/ssd0/adapters/2", "/mnt/ssd0/adapters/3", "/mnt/ssd0/adapters/4"]

lora_requests = [
    LoRARequest("lora_adapter", i+1, lora_path)
    for i, lora_path in enumerate(lora_paths)
]

llm = vllm.LLM(
    model="/mnt/ssd0/work_collection/downloaded_Qwen2.5-3b-Instruct_model/",
    num_scheduler_steps=1,
    swap_space=16,
    max_model_len=256,
    max_seq_len_to_capture=256,
    max_num_seqs=8,
    enable_lora=True,
    # enforce_eager=True,
    max_loras=2,
    max_lora_rank=8
)

for _ in range(2):
    for i, req in enumerate(lora_requests):
        print(i, llm.generate(
            "What's 1+1?",
            sampling_params=vllm.SamplingParams(
                max_tokens=256,
                temperature=0
            ),
            lora_request=req
        ))