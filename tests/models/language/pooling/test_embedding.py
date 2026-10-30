# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from transformers import AutoModel

from vllm import PoolingParams
from vllm.config import PoolerConfig

from ...utils import check_embeddings_close


@pytest.mark.cpu_test
@pytest.mark.core_model
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kernel_size", [3, 4])
@pytest.mark.parametrize("num_padding_tokens", [0, 3])
def test_lfm2_bidirectional_conv_respects_sequence_boundaries(
    dtype, kernel_size, num_padding_tokens
):
    """Packed short sequences must match independent noncausal convolutions."""
    from vllm.model_executor.models.lfm2 import _bidirectional_conv1d

    generator = torch.Generator().manual_seed(0)
    lengths = [1, 1, 7, 2]
    inputs = [torch.randn(n, 8, generator=generator).to(dtype) for n in lengths]
    weight = torch.randn(8, 1, kernel_size, generator=generator).to(dtype)
    bias = torch.randn(8, generator=generator).to(dtype)
    expected = []
    for x in inputs:
        y = torch.nn.functional.conv1d(
            x.T[None].float(),
            weight.float(),
            bias.float(),
            padding=kernel_size // 2,
            groups=8,
        )
        expected.append(y[0, :, : len(x)].T.to(dtype))

    packed = torch.cat(inputs)
    positions = torch.cat([torch.arange(n) for n in lengths])
    is_padding = None
    if num_padding_tokens:
        # Graph padding may retain positions and values from an earlier batch.
        packed = torch.cat([packed, torch.full((num_padding_tokens, 8), float("nan"))])
        positions = torch.cat([positions, torch.arange(2, 2 + num_padding_tokens)])
        is_padding = torch.arange(len(packed)) >= sum(lengths)
    actual = _bidirectional_conv1d(
        packed.to(dtype),
        weight,
        bias,
        positions,
        is_padding,
    )
    torch.testing.assert_close(actual[: sum(lengths)], torch.cat(expected))


@pytest.mark.core_model
@pytest.mark.parametrize("enforce_eager", [True, False])
def test_lfm2_bidirectional_embeddings_are_batch_independent(
    vllm_runner, enforce_eager
):
    """Neighboring requests must not change the embedding through ShortConv."""
    prompts = [
        "query: capital of France?",
        "document: Paris is the capital of France. " * 20,
        "document: Tokyo.",
    ]
    with vllm_runner(
        "LiquidAI/LFM2.5-Embedding-350M",
        runner="pooling",
        max_model_len=512,
        enforce_eager=enforce_eager,
        gpu_memory_utilization=0.4,
        default_torch_num_threads=1,
        compilation_config={"cudagraph_capture_sizes": [32, 256]},
    ) as model:
        config = model.llm.llm_engine.model_config
        assert config.architectures == ["Lfm2BidirectionalModel"]
        assert config.attn_type == "encoder_only"
        single = [model.embed([prompt])[0] for prompt in prompts]
        batched = model.embed(prompts)
        reversed_batch = model.embed(prompts[::-1])[::-1]

    assert torch.tensor(batched).shape == (3, 1024)
    assert torch.isfinite(torch.tensor(batched)).all()
    for outputs in (batched, reversed_batch):
        check_embeddings_close(
            embeddings_0_lst=single,
            embeddings_1_lst=outputs,
            name_0="single",
            name_1="batched",
            tol=1e-3,
        )


@pytest.mark.parametrize(
    "model",
    [
        # Be careful of the order of models, decoder-only models should be
        # placed before encoder-only models, otherwise `Qwen2.5-0.5B-Instruct`
        # case won't pass because gte-Qwen2-1.5B-instruct will cache custom
        # model code with bidirectional attention.
        # [Decoder-only]
        pytest.param(
            "BAAI/bge-multilingual-gemma2",
            marks=[pytest.mark.core_model, pytest.mark.slow_test],
        ),
        pytest.param(
            "intfloat/e5-mistral-7b-instruct",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
        pytest.param(
            "ssmits/Qwen2-7B-Instruct-embed-base", marks=[pytest.mark.cpu_model]
        ),
        # [Encoder-only]
        pytest.param(
            "BAAI/bge-base-en-v1.5",
            marks=[
                pytest.mark.core_model,
                pytest.mark.cpu_model,
                pytest.mark.slow_test,
            ],
        ),
        pytest.param("sentence-transformers/all-MiniLM-L12-v2"),
        pytest.param("intfloat/multilingual-e5-small"),
        # [Cross-Encoder]
        pytest.param(
            "sentence-transformers/stsb-roberta-base-v2",
            marks=[pytest.mark.core_model, pytest.mark.cpu_model],
        ),
    ],
)
def test_models(
    hf_runner,
    vllm_runner,
    example_prompts,
    model,
) -> None:
    vllm_extra_kwargs = {}
    if model == "ssmits/Qwen2-7B-Instruct-embed-base":
        vllm_extra_kwargs["pooler_config"] = PoolerConfig(
            seq_pooling_type="MEAN", use_activation=False
        )

    max_model_len: int | None = 512
    if model in [
        "sentence-transformers/all-MiniLM-L12-v2",
        "sentence-transformers/stsb-roberta-base-v2",
    ]:
        max_model_len = None

    # The example_prompts has ending "\n", for example:
    # "Write a short story about a robot that dreams for the first time.\n"
    # sentence_transformers will strip the input texts, see:
    # https://github.com/UKPLab/sentence-transformers/blob/v3.1.1/sentence_transformers/models/Transformer.py#L159
    # This makes the input_ids different between hf_model and vllm_model.
    # So we need to strip the input texts to avoid test failing.
    example_prompts = [str(s).strip() for s in example_prompts]

    with hf_runner(model, is_sentence_transformer=True) as hf_model:
        if model == "ssmits/Qwen2-7B-Instruct-embed-base":
            # This checkpoint has no sentence-transformers config. Since
            # sentence-transformers 5.6, a causal LM then gets its chat
            # template applied and last-token pooling. Keep the plain text and
            # the MEAN pooling that vLLM uses above.
            from sentence_transformers.models import Pooling

            st_model = hf_model.model
            getattr(st_model[0], "modality_config", {}).pop("message", None)
            st_model[1] = Pooling(
                st_model.get_sentence_embedding_dimension(), pooling_mode="mean"
            )
            del st_model
        hf_outputs = hf_model.encode(example_prompts)

    with vllm_runner(
        model, runner="pooling", max_model_len=max_model_len, **vllm_extra_kwargs
    ) as vllm_model:
        vllm_outputs = vllm_model.embed(example_prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


@pytest.mark.parametrize(
    "model",
    [
        "BAAI/bge-base-en-v1.5",
        "intfloat/multilingual-e5-small",
    ],
)
@torch.inference_mode()
def test_encoder_only_model_runner_v2_attention(
    hf_runner,
    vllm_runner,
    monkeypatch,
    model: str,
) -> None:
    prompts = [
        "short input",
        "a longer input that exercises mixed sequence lengths",
    ]

    with hf_runner(model, dtype="float", auto_cls=AutoModel) as hf_model:
        hf_outputs = []
        for prompt in prompts:
            inputs = hf_model.tokenizer(prompt, return_tensors="pt")
            output = hf_model.model(**hf_model.wrap_device(inputs))
            embedding = torch.nn.functional.normalize(
                output.last_hidden_state[0, -1].float(), dim=0
            )
            hf_outputs.append(embedding.cpu().tolist())

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        model,
        runner="pooling",
        dtype="float",
        max_model_len=64,
        max_num_seqs=2,
        gpu_memory_utilization=0.25,
        pooler_config=PoolerConfig(
            task="embed", seq_pooling_type="LAST", use_activation=True
        ),
    ) as vllm_model:
        assert vllm_model.llm.llm_engine.vllm_config.use_v2_model_runner
        vllm_outputs = vllm_model.embed(prompts)

    check_embeddings_close(
        embeddings_0_lst=hf_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )


@pytest.mark.core_model
def test_encoder_model_runner_v2(hf_runner, vllm_runner, monkeypatch) -> None:
    model = "sentence-transformers/all-MiniLM-L6-v2"
    prompt_batches = [
        ["short input"],
        [
            "short input",
            "a longer input that exercises mixed sequence lengths",
        ],
    ]

    with hf_runner(model, is_sentence_transformer=True) as hf_model:
        hf_outputs = [hf_model.encode(prompts) for prompts in prompt_batches]

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        model,
        runner="pooling",
        max_model_len=64,
    ) as vllm_model:
        assert vllm_model.llm.llm_engine.vllm_config.use_v2_model_runner
        vllm_outputs = [vllm_model.embed(prompts) for prompts in prompt_batches]

    for hf_batch, vllm_batch in zip(hf_outputs, vllm_outputs):
        check_embeddings_close(
            embeddings_0_lst=hf_batch,
            embeddings_1_lst=vllm_batch,
            name_0="hf",
            name_1="vllm",
            tol=1e-2,
        )


@pytest.mark.core_model
def test_matryoshka_dimensions_model_runner_v2(
    hf_runner, vllm_runner, monkeypatch
) -> None:
    model = "Snowflake/snowflake-arctic-embed-m-v1.5"
    prompts = ["short input", "a longer input for a different output width"]
    dimensions = [None, 256]

    with hf_runner(model, is_sentence_transformer=True) as hf_model:
        hf_outputs = hf_model.encode(prompts)

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with vllm_runner(
        model,
        runner="pooling",
        max_model_len=64,
        gpu_memory_utilization=0.25,
    ) as vllm_model:
        assert vllm_model.llm.llm_engine.vllm_config.use_v2_model_runner
        vllm_outputs = vllm_model.embed(
            prompts,
            pooling_params=[PoolingParams(dimensions=d) for d in dimensions],
        )

    expected_outputs = []
    for output, dimension in zip(hf_outputs, dimensions):
        output = torch.as_tensor(output)
        if dimension is not None:
            output = torch.nn.functional.normalize(output[:dimension], dim=0)
        expected_outputs.append(output.tolist())

    assert [len(output) for output in vllm_outputs] == [768, 256]
    check_embeddings_close(
        embeddings_0_lst=expected_outputs,
        embeddings_1_lst=vllm_outputs,
        name_0="hf",
        name_1="vllm",
        tol=1e-2,
    )
