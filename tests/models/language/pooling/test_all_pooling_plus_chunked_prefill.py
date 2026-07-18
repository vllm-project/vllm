# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModel

from tests.models.utils import check_embeddings_close
from tests.utils import VLLM_PATH
from vllm import TokensPrompt
from vllm.config import PoolerConfig


@pytest.mark.parametrize(
    "model",
    ["Qwen/Qwen3-Embedding-0.6B"],
)
@torch.inference_mode
def test_embed_models(hf_runner, vllm_runner, model: str):
    chunk_size = 10
    n_prompt_tokens = [55, 56, 57]
    token_prompts = [[1024 + i for i in range(n)] for n in n_prompt_tokens]

    with vllm_runner(
        model,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        max_model_len=128,
        max_num_batched_tokens=chunk_size,
        enforce_eager=True,
        # `enable_chunked_prefill`: Set to `False` instead of `None` in VllmRunner
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
    ) as vllm_model:
        vllm_outputs = vllm_model.token_embed(
            [TokensPrompt(prompt_token_ids=t) for t in token_prompts],
        )

    with hf_runner(
        model,
        auto_cls=AutoModel,
    ) as hf_model:
        hf_outputs = []
        for token_prompt in token_prompts:
            inputs = hf_model.wrap_device({"input_ids": torch.tensor([token_prompt])})
            input_ids = inputs["input_ids"]
            output = hf_model.model(input_ids)
            hf_outputs.append(output.last_hidden_state.cpu().float()[0])

    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        check_embeddings_close(
            embeddings_0_lst=hf_output,
            embeddings_1_lst=vllm_output,
            name_0="hf",
            name_1="vllm",
            tol=1e-2,
        )


_RERANKER_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
_RERANKER_TEMPLATE = VLLM_PATH / "examples/pooling/score/template/qwen3_reranker.jinja"


@pytest.mark.parametrize("model", ["Qwen/Qwen3-Reranker-0.6B"])
@torch.inference_mode
def test_last_pool_score_chunked_prefill_matches_unchunked(vllm_runner, model: str):
    """LAST-pooling score must not depend on whether the prompt is chunked.

    Regression test for a wrong-score bug where, under ``torch.compile`` (i.e.
    not ``enforce_eager``), a query+document pair long enough to be split across
    prefill chunks produced a corrupted last-token hidden state and thus a
    wrong relevance score, while the same input scored correctly unchunked.
    The existing pooling+chunked-prefill coverage runs ``enforce_eager=True``
    and so never exercised the compiled path.
    """
    chat_template = _RERANKER_TEMPLATE.read_text()
    query = "What organelle produces energy in the cell?"
    # A long, matching document so query + doc exceeds the chunk size below.
    document = (
        "The mitochondria is the powerhouse of the cell. It generates most of "
        "the cell chemical energy through oxidative phosphorylation. "
    ) * 400

    def score_with(max_num_batched_tokens: int) -> float:
        with vllm_runner(
            model,
            runner="pooling",
            hf_overrides=_RERANKER_HF_OVERRIDES,
            max_model_len=16384,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=True,
            enable_prefix_caching=False,
        ) as vllm_model:
            return vllm_model.score(query, document, chat_template=chat_template)[0]

    # Chunk size forces the prompt across several prefill chunks; the large
    # budget keeps it in a single chunk as the reference.
    chunked = score_with(2048)
    unchunked = score_with(16384)

    assert chunked == pytest.approx(unchunked, abs=5e-2), (
        f"chunked score {chunked} diverged from unchunked {unchunked}"
    )
