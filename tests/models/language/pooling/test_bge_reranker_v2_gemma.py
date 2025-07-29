# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, Optional

import numpy as np
import pytest
import torch

from tests.conftest import HfRunner

from .mteb_utils import (RerankModelInfo, VllmMtebEncoder,
                         mteb_test_rerank_models)

RERANK_MODELS = [
    RerankModelInfo("BAAI/bge-reranker-v2-gemma",
                    architecture="GemmaForSequenceClassification"),
]

PROMPT = "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."  # noqa: E501


class GemmaRerankerHfRunner(HfRunner):

    def __init__(self,
                 model_name: str,
                 dtype: str = "auto",
                 *args: Any,
                 **kwargs: Any) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        super().__init__(model_name, dtype, auto_cls=AutoModelForCausalLM)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       padding_side='left')
        self.yes_loc = self.tokenizer.convert_tokens_to_ids("Yes")

    @torch.no_grad()
    def predict(self, prompts: list[list[str]], *args,
                **kwargs) -> torch.Tensor:

        def get_inputs(pairs, tokenizer, prompt=None):
            if prompt is None:
                prompt = PROMPT

            sep = "\n"
            prompt_inputs = tokenizer(prompt,
                                      return_tensors=None,
                                      add_special_tokens=False)["input_ids"]
            sep_inputs = tokenizer(sep,
                                   return_tensors=None,
                                   add_special_tokens=False)["input_ids"]
            inputs = []
            for query, passage in pairs:
                query_inputs = tokenizer(
                    f"A: {query}",
                    return_tensors=None,
                    add_special_tokens=False,
                    truncation=True,
                )
                passage_inputs = tokenizer(
                    f"B: {passage}",
                    return_tensors=None,
                    add_special_tokens=False,
                    truncation=True,
                )
                item = tokenizer.prepare_for_model(
                    [tokenizer.bos_token_id] + query_inputs["input_ids"],
                    sep_inputs + passage_inputs["input_ids"],
                    truncation="only_second",
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
                item["input_ids"] = item[
                    "input_ids"] + sep_inputs + prompt_inputs
                item["attention_mask"] = [1] * len(item["input_ids"])
                inputs.append(item)
            return tokenizer.pad(
                inputs,
                padding=True,
                return_tensors="pt",
            )

        scores = []
        for query, doc, *_ in prompts:
            pairs = [(query, doc)]
            inputs = get_inputs(pairs, self.tokenizer)
            inputs = inputs.to(self.model.device)
            _n_tokens = inputs["input_ids"].shape[1]
            logits = self.model(**inputs, return_dict=True).logits
            _scores = (logits[:, -1,
                              self.yes_loc].view(-1, ).float().sigmoid())
            scores.append(_scores[0].item())
        return torch.Tensor(scores)


class GemmaMtebEncoder(VllmMtebEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt = PROMPT
        self.query_template = "A: {query}\n"
        self.document_template = "B: {doc}\n{prompt}"

    def predict(
        self,
        sentences: list[tuple[str, str,
                              Optional[str]]],  # query, corpus, prompt
        *args,
        **kwargs,
    ) -> np.ndarray:

        _sentences = []
        for query, corpus, prompt in sentences:
            query = self.query_template.format(query=query)
            corpus = self.document_template.format(doc=corpus, prompt=prompt)
            _sentences.append((query, corpus, prompt))

        return super().predict(_sentences, *args, **kwargs)


@pytest.mark.parametrize("model_info", RERANK_MODELS)
def test_rerank_models_mteb(vllm_runner, model_info: RerankModelInfo,
                            monkeypatch) -> None:
    monkeypatch.setenv("VLLM_USE_V1", "0")

    assert model_info.architecture == "GemmaForSequenceClassification"

    vllm_extra_kwargs: dict[str, Any] = {
        "hf_overrides": {
            "architectures": ["GemmaForSequenceClassification"],
            "classifier_from_token": ["Yes"],
            "method": "no_post_processing",
        }
    }

    mteb_test_rerank_models(GemmaRerankerHfRunner,
                            vllm_runner,
                            model_info,
                            vllm_extra_kwargs,
                            vllm_mteb_encoder=GemmaMtebEncoder)
