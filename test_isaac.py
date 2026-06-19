import torch
from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.models.isaac import IsaacForConditionalGeneration

vllm_config = VllmConfig(model_config=ModelConfig("PerceptronAI/Isaac-0.1", "isaac", tokenizer_mode="auto", trust_remote_code=True, seed=0, dtype="float16"))
model = IsaacForConditionalGeneration(vllm_config=vllm_config)
print("Initialization successful!")
