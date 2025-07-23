# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import pytest
import torch

from tests.conftest import HfRunner

from ...utils import LASTPoolingRerankModelInfo, RerankModelInfo
from .mteb_utils import mteb_test_rerank_models

RERANK_MODELS = [
    LASTPoolingRerankModelInfo("mixedbread-ai/mxbai-rerank-base-v2",
                               architecture="Qwen2ForSequenceClassification",
                               enable_test=True),
    LASTPoolingRerankModelInfo("mixedbread-ai/mxbai-rerank-large-v2",
                               architecture="Qwen2ForSequenceClassification",
                               enable_test=False)
]


class MxbaiRerankerHfRunner(HfRunner):

    def __init__(self,
                 model_name: str,
                 dtype: str = "auto",
                 *args: Any,
                 **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        super().__init__(model_name, dtype, auto_cls=AutoModelForCausalLM)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side='left')
        self.yes_loc = self.tokenizer.convert_tokens_to_ids("1")
        self.no_loc = self.tokenizer.convert_tokens_to_ids("0")

    def predict(self, prompts: list[list[str]], *args,
                **kwargs) -> torch.Tensor:

        def process_inputs(pairs):
            inputs = self.tokenizer(pairs,
                                    padding=False,
                                    truncation='longest_first',
                                    return_attention_mask=False)
            for i, ele in enumerate(inputs['input_ids']):
                inputs['input_ids'][i] = ele
            inputs = self.tokenizer.pad(inputs,
                                        padding=True,
                                        return_tensors="pt")
            for key in inputs:
                inputs[key] = inputs[key].to(self.model.device)
            return inputs

        @torch.no_grad()
        def compute_logits(inputs):
            logits = self.model(**inputs).logits[:, -1, :]
            yes_logits = logits[:, self.yes_loc]
            no_logits = logits[:, self.no_loc]
            logits = yes_logits - no_logits
            scores = logits.float().sigmoid()
            return scores

        scores = []
        for prompt in prompts:
            inputs = process_inputs([prompt])
            score = compute_logits(inputs)
            scores.append(score[0].item())
        return torch.Tensor(scores)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo) -> None:
    vllm_extra_kwargs: dict[str, Any] = {}
    if model_info.architecture == "Qwen2ForSequenceClassification":
        vllm_extra_kwargs["hf_overrides"] = {
            "architectures": ["Qwen2ForSequenceClassification"],
            "classifier_from_token": ["0", "1"],
            "method": "from_2_way_softmax",
        }

    mteb_test_rerank_models(MxbaiRerankerHfRunner, vllm_runner, model_info,
                            vllm_extra_kwargs)
