# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch.nn as nn

from vllm.platforms import current_platform

prompt = """
Generals gathered in their masses
Just like witches at black masses
Evil minds that plot destruction
Sorcerer of death's construction
In the fields, the bodies burning
As the war machine keeps turning
Death and hatred to mankind
Poisoning their brainwashed minds
Oh, Lord, yeah

Politicians hide themselves away
They only started the war
Why should they go out to fight?
They leave that all to the poor, yeah
Time will tell on their power minds
Making war just for fun
Treating people just like pawns in chess
Wait till their judgment day comes, yeah

Now, in darkness, world stops turning
Ashes where their bodies burning
No more war pigs have the power
Hand of God has struck the hour
Day of Judgment, God is calling
On their knees, the war pigs crawling
Begging mercies for their sins
Satan, laughing, spreads his wings
Oh, Lord, yeah
"""


class WrapperPooler(nn.Module):
    def __init__(self, pooler):
        super().__init__()
        self.pooler = pooler
        self.chunks = []

    def get_pooling_updates(self, task):
        return self.pooler.get_pooling_updates(task)

    def forward(
        self,
        hidden_states,
        pooling_metadata,
    ):
        self.chunks.append(hidden_states.shape[0])
        return self.pooler(hidden_states, pooling_metadata)


def inject_pooler(self):
    model = self.get_model()
    wrapper = WrapperPooler(model.pooler)
    model.pooler = wrapper


def retrieve_chunks(self):
    model = self.get_model()
    chunks = model.pooler.chunks
    model.pooler.chunks = []
    return chunks


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
def test_pooling_chunked_prefill(vllm_runner, monkeypatch):
    """Test chunked prefill for pooling models with LastPool."""

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        model_id = "Qwen/Qwen3-Embedding-0.6B"

        chunk_size = 10

        # Set chunking parameters to force chunked prefill
        # Note: Chunked prefill is automatically handled by vLLM
        # internally based on the model size and prompt
        with vllm_runner(
            model_id,
            runner="pooling",
            long_prefill_token_threshold=chunk_size,
            tensor_parallel_size=1,
            enforce_eager=True,
            enable_chunked_prefill=True,
        ) as llm:
            llm.get_llm().llm_engine.collective_rpc(inject_pooler)

            tokenizer = llm.get_llm().get_tokenizer()
            tokens = tokenizer(prompt)["input_ids"]
            prompt_len = len(tokens)
            full_chunks, last_chunk = divmod(prompt_len, chunk_size)
            expected_chunks = [chunk_size] * full_chunks
            if last_chunk:
                expected_chunks.append(last_chunk)
            llm.embed([prompt])
            chunks = llm.get_llm().llm_engine.collective_rpc(retrieve_chunks)[0]

        # Check that PoolerWrapper was called and chunks were received
        assert len(chunks) > 1
        assert chunks == expected_chunks

        # Disable chunked prefill
        with vllm_runner(
            model_id,
            runner="pooling",
            tensor_parallel_size=1,
            enforce_eager=True,
        ) as llm:
            llm.get_llm().llm_engine.collective_rpc(inject_pooler)
            llm.embed([prompt])
            chunks = llm.get_llm().llm_engine.collective_rpc(retrieve_chunks)[0]

        # Check that PoolerWrapper was called and no chunks were received
        assert len(chunks) == 1
        assert chunks[0] == prompt_len


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
def test_pooling_prefix_cache(vllm_runner, monkeypatch):
    """Test chunked prefill for pooling models with LastPool."""

    verses = prompt.split("\n\n")

    with monkeypatch.context() as m:
        m.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        model_id = "Qwen/Qwen3-Embedding-0.6B"

        with vllm_runner(
            model_id,
            runner="pooling",
            enable_prefix_caching=True,
            tensor_parallel_size=1,
            enforce_eager=True,
        ) as llm:
            llm.get_llm().llm_engine.collective_rpc(inject_pooler)
            tokenizer = llm.get_llm().get_tokenizer()

            prompt1 = "\n\n".join([verses[0], verses[1]])
            prompt2 = "\n\n".join([verses[0], verses[2]])
            tokens1 = tokenizer(prompt1)["input_ids"]
            tokens2 = tokenizer(prompt2)["input_ids"]
            prompt1_len = len(tokens1)
            prompt2_len = len(tokens2)

            llm.embed([prompt1])
            chunks = llm.get_llm().llm_engine.collective_rpc(retrieve_chunks)[0]

            assert len(chunks) == 1
            assert chunks[0] == prompt1_len

            llm.embed([prompt2])
            chunks = llm.get_llm().llm_engine.collective_rpc(retrieve_chunks)[0]

            assert len(chunks) == 1
            assert chunks[0] <= prompt1_len
            assert chunks[0] < prompt2_len

            cache_config = llm.get_llm().llm_engine.cache_config
            print(f"{cache_config=}")
            # Prefixes are cached in blocks
            assert (prompt2_len - chunks[0]) % cache_config.block_size == 0
