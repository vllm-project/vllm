# tests/models/embedding/vision_language/test_colqwen2vl.py

import torch
from vllm.model_executor.models.colqwen2_vl import ColQwen2VL

def test_colqwen2vl_embeddings():
    model = ColQwen2VL()
    dummy_input = torch.rand((1, 3, 224, 224))  # Example input
    embeddings = model(dummy_input)
    assert embeddings.shape == (1, 128), "Embedding size should be 128."