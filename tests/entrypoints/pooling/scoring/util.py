# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from tests.conftest import HfRunner
from vllm.entrypoints.pooling.scoring.utils import compute_maxsim_score


class ColBERTScoringHfRunner(torch.nn.Module):
    def __init__(self, model_name, linear_weights_key):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        extra = {}
        if self.device.type == "cpu":
            extra["attn_implementation"] = "eager"

        self.model = AutoModel.from_pretrained(
            model_name,
            **extra,
        ).to(self.device)
        self.model.eval()

        path = hf_hub_download(model_name, filename="model.safetensors")
        weights = load_file(path)

        self.linear_weight = weights[linear_weights_key].to(self.device).float()

    @torch.inference_mode()
    def forward(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            hidden = self.model(**inputs).last_hidden_state.float()
            projected = F.linear(hidden, self.linear_weight.float())
            normalised = F.normalize(projected, p=2, dim=-1)
            embeddings.append(normalised.squeeze(0).cpu())
        return embeddings

    @torch.inference_mode()
    def predict(self, prompts: list[list[str]], *args, **kwargs):
        hf_embeddings = [self(prompt) for prompt in prompts]
        hf_outputs = [
            compute_maxsim_score(*map(torch.tensor, pair)).item()
            for pair in hf_embeddings
        ]
        return torch.as_tensor(hf_outputs)


class EncoderScoringHfRunner(HfRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, is_sentence_transformer=True)

    @torch.inference_mode()
    def predict(self, prompts: list[list[str]], *args, **kwargs):
        hf_embeddings = [self.encode(prompt) for prompt in prompts]
        hf_outputs = [
            F.cosine_similarity(*map(torch.tensor, pair), dim=0)
            for pair in hf_embeddings
        ]
        return torch.as_tensor(hf_outputs)
