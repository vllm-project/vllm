# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.layers.pooler import PoolingType, LastPool
from vllm.engine.llm_engine import LLMEngine

class DummyPooler(LastPool):
    def __init__(self):
        super().__init__()
        self.chunks = []
    def __call__(self, hidden_states, pooling_cursor):
        self.chunks.append(hidden_states)
        return super().__call__(hidden_states, pooling_cursor)

def test_chunked_prefill_pooler(monkeypatch):
    """Test chunked prefill for pooling models with LastPool."""
    model_id = "BAAI/bge-multilingual-gemma2"
    config = ModelConfig(model_id)
    pooler = DummyPooler()
    config.pooler_config = PoolerConfig(pooling_type="LAST")
    # Patch LLMEngine to use DummyPooler
    monkeypatch.setattr("vllm.model_executor.layers.pooler.LastPool", DummyPooler)
    engine = LLMEngine(config)
    prompt = "This is a test prompt for chunked prefill."
    output = engine.generate([prompt], max_tokens=8, enable_chunked_prefill=True)
    # Check that chunks were received
    assert len(pooler.chunks) > 1
    # Compare with non-chunked output
    output_non_chunked = engine.generate([prompt], max_tokens=8, enable_chunked_prefill=False)
    assert output[0] == output_non_chunked[0]

def test_chunked_prefill_prefix_caching(monkeypatch):
    """Test chunked prefill with prefix caching for pooling models."""
    model_id = "BAAI/bge-multilingual-gemma2"
    config = ModelConfig(model_id)
    pooler = DummyPooler()
    config.pooler_config = PoolerConfig(pooling_type="LAST")
    monkeypatch.setattr("vllm.model_executor.layers.pooler.LastPool", DummyPooler)
    engine = LLMEngine(config)
    prefix = "Common prefix. "
    prompt1 = prefix + "First input."
    prompt2 = prefix + "Second input."
    engine.generate([prompt1], max_tokens=8, enable_chunked_prefill=True)
    output2 = engine.generate([prompt2], max_tokens=8, enable_chunked_prefill=True)
    # The pooler should see hidden states of length (total - prefix length)
    assert all(len(chunk) <= len(prompt2) - len(prefix) for chunk in pooler.chunks)
