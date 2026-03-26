# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref

import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.entrypoints.pooling.score.utils import compute_maxsim_score
from vllm.platforms import current_platform

MODEL_NAME = "answerdotai/answerai-colbert-small-v1"
COLBERT_DIM = 96

LINEAR_WEIGHTS_KEY = "linear.weight"
PROMPT = "The chef prepared a delicious meal."

TEXTS_1 = [
    "What is the capital of France?",
    "What is the capital of Germany?",
]

TEXTS_2 = [
    "The capital of France is Paris.",
    "The capital of Germany is Berlin.",
]

DTYPE = "half"


class ColBERT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        extra = {}
        if self.device.type == "cpu":
            extra["attn_implementation"] = "eager"

        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            **extra,
        ).to(self.device)
        self.model.eval()

        path = hf_hub_download(MODEL_NAME, filename="model.safetensors")
        weights = load_file(path)

        self.linear_weight = weights[LINEAR_WEIGHTS_KEY].to(self.device).float()

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


@pytest.fixture(scope="module")
def llm():
    # ROCm: Use FLEX_ATTENTION backend as it's the only attention backend
    # that supports encoder-only models on ROCm.
    attention_config = None
    if current_platform.is_rocm():
        attention_config = {"backend": "FLEX_ATTENTION"}

    # pytest caches the fixture so we use weakref.proxy to
    # enable garbage collection
    llm = LLM(
        model=MODEL_NAME,
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        attention_config=attention_config,
    )

    yield weakref.proxy(llm)

    del llm

    cleanup_dist_env_and_memory()


@pytest.fixture(scope="module")
def hf_model(hf_runner):
    return ColBERT()


@pytest.mark.skip_global_cleanup
def test_1_to_1(llm, hf_model):
    text_pair = [TEXTS_1[0], TEXTS_2[0]]

    hf_embeddings = hf_model(text_pair)
    hf_outputs = [compute_maxsim_score(hf_embeddings[0], hf_embeddings[1]).item()]

    vllm_outputs = [
        output.outputs.score for output in llm.score(text_pair[0], text_pair[1])
    ]

    assert len(vllm_outputs) == 1
    assert len(hf_outputs) == 1

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_1_to_n(llm, hf_model):
    hf_embeddings = hf_model([TEXTS_1[0]] + TEXTS_2)
    hf_outputs = [
        compute_maxsim_score(hf_embeddings[0], hf_embeddings[1]).item(),
        compute_maxsim_score(hf_embeddings[0], hf_embeddings[2]).item(),
    ]
    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1[0], TEXTS_2)]

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


@pytest.mark.skip_global_cleanup
def test_n_to_n(llm, hf_model):
    hf_embeddings = hf_model(TEXTS_1 + TEXTS_2)
    hf_outputs = [
        compute_maxsim_score(hf_embeddings[0], hf_embeddings[2]).item(),
        compute_maxsim_score(hf_embeddings[1], hf_embeddings[3]).item(),
    ]
    vllm_outputs = [output.outputs.score for output in llm.score(TEXTS_1, TEXTS_2)]

    assert len(vllm_outputs) == 2
    assert len(hf_outputs) == 2

    assert hf_outputs[0] == pytest.approx(vllm_outputs[0], rel=0.01)
    assert hf_outputs[1] == pytest.approx(vllm_outputs[1], rel=0.01)


def test_token_embed(llm):
    outputs = llm.encode(PROMPT, pooling_task="token_embed", use_tqdm=False)
    assert len(outputs) == 1
    assert outputs[0].outputs.data.shape == (9, COLBERT_DIM)
