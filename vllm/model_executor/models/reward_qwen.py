import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class QWenRewardModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path, trust_remote_code=True
        )

        self.model = AutoModel.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).eval()

        self.hidden_size = self.model.config.hidden_size

        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        reward_head_path = os.path.join(config.model_path, "reward_head.pt")
        if not os.path.exists(reward_head_path):
            raise FileNotFoundError(f"未找到 reward_head.pt: {reward_head_path}")
        self.reward_head.load_state_dict(torch.load(reward_head_path, map_location="cpu"))
        self.reward_head.half()  # 强制转 float16
        self.reward_head.eval()

    def score(self, input_ids, attention_mask):
        # === 获取最后一层 hidden state ===
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden = outputs.last_hidden_state  # [B, L, H]

        # === 取出 eos_token 对应位置的 hidden state ===
        eos_mask = input_ids == self.tokenizer.eos_token_id
        eos_index = eos_mask.int().argmax(dim=1)  # [B]
        eos_hidden = hidden[torch.arange(input_ids.size(0)), eos_index]  # [B, H]

        # === 打分 ===
        score = self.reward_head(eos_hidden)  # [B, 1]
        return score.squeeze(-1)  # [B]
