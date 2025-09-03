# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest
from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.layers.pooler import LastPool
from vllm.v1.engine.llm_engine import LLMEngine

def test_chunked_prefill_pooler(monkeypatch):
    """Test chunked prefill for pooling models with LastPool."""
    model_id = "BAAI/bge-multilingual-gemma2"
    config = ModelConfig(model_id)
    config.pooler_config = PoolerConfig(pooling_type="LAST")
    # Use a closure to track chunks
    chunks = []
    class DummyPooler(LastPool):
        def __call__(self, hidden_states, pooling_cursor):
            chunks.append(hidden_states)
            return super().__call__(hidden_states, pooling_cursor)
    monkeypatch.setattr("vllm.model_executor.layers.pooler.LastPool", DummyPooler)
    # Set chunking parameters to force chunked prefill
    engine = LLMEngine(config, enable_chunked_prefill=True, long_prefill_token_threshold=1)
    prompt = "This is a test prompt for chunked prefill."
    output = engine.embed([prompt])
    # Check that DummyPooler was called and chunks were received
    assert len(chunks) > 0
    # Verify the sum of the lengths of the chunks matches the prompt length
    total_chunk_len = sum(len(chunk) for chunk in chunks)
    assert total_chunk_len == len(prompt)
    # Compare with non-chunked output
    engine_non_chunked = LLMEngine(config, enable_chunked_prefill=False)
    output_non_chunked = engine_non_chunked.embed([prompt])
    assert output[0] == output_non_chunked[0]
    # Note: For faster tests, use a smaller model like 'Qwen/Qwen3-Embedding-0.6'.
    # To override the pooler, you can set trust_remote_code=True and use auto_map in hf_config.

def test_chunked_prefill_prefix_caching(monkeypatch):
    """Test chunked prefill with prefix caching for pooling models."""
    model_id = "BAAI/bge-multilingual-gemma2"
    config = ModelConfig(model_id)
    config.pooler_config = PoolerConfig(pooling_type="LAST")
    chunks = []
    class DummyPooler(LastPool):
        def __call__(self, hidden_states, pooling_cursor):
            chunks.append(hidden_states)
            return super().__call__(hidden_states, pooling_cursor)
    monkeypatch.setattr("vllm.model_executor.layers.pooler.LastPool", DummyPooler)
    engine = LLMEngine(config, enable_chunked_prefill=True, long_prefill_token_threshold=1)
    prefix = "Common prefix. "
    prompt1 = prefix + "First input."
    prompt2 = prefix + "Second input."
    engine.embed([prompt1])
    chunks.clear()
    engine.embed([prompt2])
    # Only the last hidden states should be checked (those going into the pooler)
    # Verify the sum of the lengths of the chunks matches the prompt length minus prefix
    total_chunk_len = sum(len(chunk) for chunk in chunks)
    assert total_chunk_len == len(prompt2) - len(prefix)
