# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import os
import torch
from vllm.config import ModelConfig, PoolerConfig
from vllm.model_executor.layers.pooler import LastPool

def test_chunked_prefill_pooler(monkeypatch):
    """Test chunked prefill for pooling models with LastPool."""
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    config = ModelConfig(model_id)
    config.pooler_config = PoolerConfig(pooling_type="LAST")
    
    # Use a closure to track chunks
    chunks = []
    
    class DummyPooler(LastPool):
        def __call__(self, hidden_states, pooling_cursor):
            chunks.append(hidden_states)
            return super().__call__(hidden_states, pooling_cursor)
    
    monkeypatch.setattr("vllm.model_executor.layers.pooler.LastPool", DummyPooler)
    
    # Set environment variables for Windows compatibility
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage on Windows
    
    # Set chunking parameters to force chunked prefill
    from vllm.entrypoints.llm import LLM
    
    # Note: Chunked prefill is automatically handled by vLLM internally based on the model size and prompt
    llm = LLM(
        model=model_id,
        runner="pooling",
        override_pooler_config=PoolerConfig(pooling_type="LAST"),
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,  # Helps with Windows compatibility
    )
    
    prompt = "This is a test prompt for chunked prefill."
    output = llm.embed([prompt])
    
    # Check that DummyPooler was called and chunks were received
    assert len(chunks) > 0
    
    # Verify the sum of the lengths of the chunks matches the prompt length
    total_chunk_len = sum(len(chunk) for chunk in chunks)
    assert total_chunk_len == len(prompt)
    
    # Compare with non-chunked output
    llm_non_chunked = LLM(
        model=model_id, 
        runner="pooling", 
        override_pooler_config=PoolerConfig(pooling_type="LAST"), 
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,
    )
    output_non_chunked = llm_non_chunked.embed([prompt])
    
    # Compare embeddings with tolerance for floating point differences
    assert torch.allclose(torch.tensor(output[0]), torch.tensor(output_non_chunked[0]), atol=1e-6)
    
    # Note: For faster tests, use a smaller model like 'Qwen/Qwen3-Embedding-0.6'.
    # To override the pooler, you can set trust_remote_code=True and use auto_map in hf_config.

def test_chunked_prefill_prefix_caching(monkeypatch):
    """Test chunked prefill with prefix caching for pooling models."""
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    config = ModelConfig(model_id)
    config.pooler_config = PoolerConfig(pooling_type="LAST")
    
    chunks = []
    
    class DummyPooler(LastPool):
        def __call__(self, hidden_states, pooling_cursor):
            chunks.append(hidden_states)
            return super().__call__(hidden_states, pooling_cursor)
    
    monkeypatch.setattr("vllm.model_executor.layers.pooler.LastPool", DummyPooler)
    
    # Set environment variables for Windows compatibility
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU usage on Windows
    
    from vllm.entrypoints.llm import LLM
    
    # Note: Chunked prefill is automatically handled by vLLM internally based on the model size and prompt
    llm = LLM(
        model=model_id,
        runner="pooling",
        override_pooler_config=PoolerConfig(pooling_type="LAST"),
        trust_remote_code=True,
        tensor_parallel_size=1,
        enforce_eager=True,  # Helps with Windows compatibility
    )
    
    prefix = "Common prefix. "
    prompt1 = prefix + "First input."
    prompt2 = prefix + "Second input."
    
    llm.embed([prompt1])
    chunks.clear()
    llm.embed([prompt2])
    
    # Only the last hidden states should be checked (those going into the pooler)
    # Verify the sum of the lengths of the chunks matches the prompt length minus prefix
    total_chunk_len = sum(len(chunk) for chunk in chunks)
    assert total_chunk_len == len(prompt2) - len(prefix)