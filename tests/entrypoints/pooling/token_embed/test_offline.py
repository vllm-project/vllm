# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import weakref

import pytest

from vllm import LLM, PoolingRequestOutput
from vllm.config import PoolerConfig
from vllm.tasks import PoolingTask

MODEL_NAME = "intfloat/multilingual-e5-small"

prompt = "The chef prepared a delicious meal."
prompt_token_ids = [0, 581, 21861, 133888, 10, 8, 150, 60744, 109911, 5, 2]
embedding_size = 384


@pytest.fixture(scope="module")
def llm(vllm_runner):
    with vllm_runner(
        MODEL_NAME,
        max_model_len=None,
        pooler_config=PoolerConfig(task="token_embed"),
        max_num_batched_tokens=32768,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        seed=0,
        enable_chunked_prefill=None,
    ) as runner:
        assert embedding_size == runner.llm.model_config.embedding_size
        # pytest caches yielded fixtures until after teardown, so use a proxy to
        # avoid retaining the LLM while VllmRunner.__exit__ releases ROCm memory.
        yield weakref.proxy(runner.llm)


@pytest.mark.skip_global_cleanup
def test_str_prompts(llm: LLM):
    outputs = llm.encode(prompt, pooling_task="token_embed", use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], PoolingRequestOutput)
    assert outputs[0].outputs.data.shape == (11, 384)


@pytest.mark.skip_global_cleanup
def test_token_ids_prompts(llm: LLM):
    outputs = llm.encode([prompt_token_ids], pooling_task="token_embed", use_tqdm=False)
    assert len(outputs) == 1
    assert isinstance(outputs[0], PoolingRequestOutput)
    assert outputs[0].outputs.data.shape == (11, 384)


@pytest.mark.parametrize("task", ["embed", "classify", "token_classify", "plugin"])
def test_unsupported_tasks(llm: LLM, task: PoolingTask, caplog_vllm):
    if task == "plugin":
        err_msg = "No IOProcessor plugin installed."
    elif task == "embed":
        err_msg = "Try switching the model's pooling_task via.+"
    else:
        err_msg = "Classification API is not supported by this model.+"

    with pytest.raises(ValueError, match=err_msg):
        llm.encode(prompt, pooling_task=task, use_tqdm=False)


@pytest.mark.parametrize("use_v2_runner", [False, True])
def test_nomic_late_chunking_offline(vllm_runner, monkeypatch, use_v2_runner):
    """Compare worker chunking with unnormalized states using both GPU runners."""
    import torch

    from tests.models.utils import check_embeddings_close
    from vllm import PoolingParams
    from vllm.exceptions import VLLMValidationError

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1" if use_v2_runner else "0")
    prompts = [
        "search_document: Berlin is a city. It is in Germany.",
        "search_document: 中文 😀 café e\u0301",
        "   ",
        "search_document: short",
        "search_document: A longer paragraph with several different token lengths.",
        "search_document: repeated text",
    ]
    with vllm_runner(
        "nomic-ai/nomic-embed-text-v1",
        revision="720244025c1a7e15661a174c63cce63c8218e52b",
        trust_remote_code=True,
        runner="pooling",
        pooler_config=PoolerConfig(task="token_embed"),
        dtype="float32",
        max_model_len=128,
        max_num_seqs=2,
        gpu_memory_utilization=0.3,
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
    ) as runner:
        llm = runner.llm
        for add_special_tokens in (True, False):
            # Empty token sequences are checked separately below.
            texts = prompts if add_special_tokens else [p for p in prompts if p.strip()]
            kwargs = {"add_special_tokens": add_special_tokens}
            raw = llm.encode(
                texts,
                pooling_task="token_embed",
                use_tqdm=False,
                pooling_params=PoolingParams(use_activation=False),
                tokenization_kwargs=kwargs,
            )
            normal = llm.encode(
                texts,
                pooling_task="token_embed",
                use_tqdm=False,
                tokenization_kwargs=kwargs,
            )
            for normalize in (False, True):
                sizes = [1, 3, 2, 128, 4, None][: len(texts)]
                params = [
                    PoolingParams(
                        late_chunk_size=c,
                        use_activation=normalize,
                        skip_reading_prefix_cache=False,
                    )
                    for c in sizes
                ]
                actual = llm.encode(
                    texts,
                    pooling_task="token_embed",
                    use_tqdm=False,
                    pooling_params=params,
                    tokenization_kwargs=kwargs,
                )
                for text, size, states, ordinary, output in zip(
                    texts, sizes, raw, normal, actual
                ):
                    assert output.prompt_token_ids == states.prompt_token_ids
                    if size is None:
                        expected = (
                            ordinary.outputs.data if normalize else states.outputs.data
                        )
                        assert output.late_chunking is None
                    else:
                        expected = torch.stack(
                            [
                                part.float().mean(0)
                                for part in states.outputs.data.split(size)
                            ]
                        )
                        if normalize:
                            expected = torch.nn.functional.normalize(
                                expected, p=2, dim=-1
                            )
                        metadata = output.late_chunking
                        assert metadata is not None
                        encoded = llm.get_tokenizer()(
                            text,
                            add_special_tokens=add_special_tokens,
                            return_offsets_mapping=True,
                        )
                        assert encoded["input_ids"] == output.prompt_token_ids
                        assert metadata.input_tokens == len(encoded["input_ids"])
                        assert len(metadata.chunks) == expected.shape[0]
                        for chunk in metadata.chunks:
                            start, end = chunk.token_range
                            offsets = [
                                (a, b)
                                for a, b in encoded["offset_mapping"][start:end]
                                if a < b
                            ]
                            char_range = (
                                (min(a for a, _ in offsets), max(b for _, b in offsets))
                                if offsets
                                else None
                            )
                            assert chunk.char_range == char_range
                    # Use the model suite's cosine criterion across forwards:
                    # batch shapes can change encoder rounding. Unit tests check
                    # the reducer against identical states with tight tolerances.
                    check_embeddings_close(
                        embeddings_0_lst=output.outputs.data.tolist(),
                        embeddings_1_lst=expected.tolist(),
                        name_0="late_chunking",
                        name_1="contextual_token_reference",
                    )
                    torch.testing.assert_close(
                        output.outputs.data.norm(dim=-1),
                        expected.norm(dim=-1),
                        atol=1e-3,
                        rtol=1e-3,
                    )
                assert all(p.task is None for p in params)

        for text, kwargs in [
            ("long " * 256, {}),
            ("   ", {"add_special_tokens": False}),
        ]:
            with pytest.raises(VLLMValidationError):
                llm.encode(
                    text,
                    pooling_task="token_embed",
                    use_tqdm=False,
                    pooling_params=PoolingParams(late_chunk_size=3),
                    tokenization_kwargs=kwargs,
                )
        # Rejections must leave the instance usable, with no retained chunk ranges.
        output = llm.encode(
            "after rejection", pooling_task="token_embed", use_tqdm=False
        )[0]
        assert output.late_chunking is None
