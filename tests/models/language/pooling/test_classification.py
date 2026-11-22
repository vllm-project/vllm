# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch
from transformers import AutoModelForSequenceClassification

from vllm.platforms import current_platform


@pytest.mark.core_model
@pytest.mark.skipif(not current_platform.is_cuda(), reason="requires CUDA graph replay")
def test_bert_segment_ids_with_breakable_cudagraph(vllm_runner, tmp_path, monkeypatch):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "c"]
    BertTokenizer(vocab={s: i for i, s in enumerate(vocab)}).save_pretrained(tmp_path)
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        max_position_embeddings=64,
        num_labels=1,
        initializer_range=0.12,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(123)
        BertForSequenceClassification(config).save_pretrained(tmp_path)

    monkeypatch.setenv("VLLM_USE_BREAKABLE_CUDAGRAPH", "1")

    def scores(enforce_eager):
        with vllm_runner(
            str(tmp_path),
            runner="pooling",
            dtype="half",
            enforce_eager=enforce_eager,
            max_model_len=32,
            max_num_seqs=4,
            max_num_batched_tokens=64,
            compilation_config={
                "cudagraph_mode": "PIECEWISE",
                "cudagraph_capture_sizes": [8],
            },
        ) as model:
            # Same token IDs; only the query/document boundary moves or disappears.
            return (
                model.score("a", "b [SEP] c")[0],
                model.score("a [SEP] b", "c")[0],
                model.classify(["a [SEP] b [SEP] c"])[0][0],
            )

    eager = scores(True)
    assert eager[0] != pytest.approx(eager[1])
    assert scores(False) == pytest.approx(eager, abs=1e-4)


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(
            "vllm-project/Qwen2.5-1.5B-apeach",
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["half"] if current_platform.is_rocm() else ["float"])
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model: str,
    dtype: str,
) -> None:
    with vllm_runner(model, max_model_len=512, dtype=dtype) as vllm_model:
        vllm_outputs = vllm_model.classify(example_prompts)

    with hf_runner(
        model, dtype=dtype, auto_cls=AutoModelForSequenceClassification
    ) as hf_model:
        hf_outputs = hf_model.classify(example_prompts)

    # check logits difference
    for hf_output, vllm_output in zip(hf_outputs, vllm_outputs):
        hf_output = torch.tensor(hf_output)
        vllm_output = torch.tensor(vllm_output)

        # the tolerance value of 1e-2 is selected based on the
        # half datatype tests in
        # tests/models/language/pooling/test_embedding.py
        assert torch.allclose(
            hf_output,
            vllm_output,
            rtol=2e-3 if dtype == "float" else 1e-2,
        )


@pytest.mark.core_model
def test_bert_model_runner_v2(hf_runner, vllm_runner) -> None:
    model = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    score_inputs = (
        "What is the capital of France?",
        [
            "Paris.",
            "Paris is the capital and largest city of France.",
            "William Shakespeare wrote Hamlet in the early seventeenth century.",
        ],
    )
    prompt_batches = [
        ["short input"],
        [
            "short input",
            "a longer input that exercises mixed sequence lengths",
        ],
    ]

    with hf_runner(
        model, dtype="half", auto_cls=AutoModelForSequenceClassification
    ) as hf_model:
        # HfRunner uses problem_type to preserve the model's
        # sbert_ce_default_activation_function=Identity raw logits.
        hf_model.config.problem_type = "regression"
        hf_outputs = [hf_model.classify(prompts) for prompts in prompt_batches]

    text_1, text_2 = score_inputs
    text_pairs = [[text_1, document] for document in text_2]
    with hf_runner(model, dtype="half", is_cross_encoder=True) as hf_model:
        hf_scores = hf_model.predict(text_pairs).tolist()

    with vllm_runner(
        model,
        runner="pooling",
        dtype="half",
        max_model_len=64,
    ) as vllm_model:
        vllm_outputs = [vllm_model.classify(prompts) for prompts in prompt_batches]
        vllm_scores = vllm_model.score(*score_inputs)

    for hf_batch, vllm_batch in zip(hf_outputs, vllm_outputs):
        hf_tensor = torch.tensor(hf_batch)
        vllm_tensor = torch.tensor(vllm_batch)
        assert vllm_tensor.shape == hf_tensor.shape
        assert torch.allclose(vllm_tensor, hf_tensor, rtol=1e-2, atol=1e-4)

    assert torch.allclose(
        torch.tensor(vllm_scores),
        torch.tensor(hf_scores),
        rtol=1e-2,
        atol=1e-4,
    )
