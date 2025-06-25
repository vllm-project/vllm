import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QWenRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True, 
            attn_implementation="flash_attention_2"
        ).to(device).eval()

        self.hidden_size = self.model.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(device)

        reward_head_path = os.path.join(config.model_path, "reward_head.pt")
        if not os.path.exists(reward_head_path):
            raise FileNotFoundError(f"未找到 reward_head.pt: {reward_head_path}")

        self.reward_head.load_state_dict(torch.load(reward_head_path, map_location=device))
        self.reward_head.half()  # 强制转 float16
        self.reward_head.eval()

    def score(self, input_ids, attention_mask):

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            hidden = outputs.last_hidden_state  # [B, L, H]

            eos_mask = input_ids == self.tokenizer.eos_token_id
            eos_index = eos_mask.int().argmax(dim=1)
            eos_hidden = hidden[torch.arange(input_ids.size(0)), eos_index]  # [B, H]

            score = self.reward_head(eos_hidden)  # [B, 1]
            return score.squeeze(-1)

# # ✅ vLLM 注册
# from vllm.model_executor.models.registry import ModelRegistry
# ModelRegistry.register("qwen3-reward", QWenRewardModel)
