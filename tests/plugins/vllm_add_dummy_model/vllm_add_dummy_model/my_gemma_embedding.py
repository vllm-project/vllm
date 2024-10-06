import torch

from vllm.model_executor.models.gemma2_embedding import Gemma2EmbeddingModel


class MyGemma2Embedding(Gemma2EmbeddingModel):

    def forward(self, *args, **kwargs) -> torch.Tensor:
        hidden_states = super().forward(*args, **kwargs)

        # We assume PP isn't used in the test
        assert isinstance(hidden_states, torch.Tensor)

        # Return all-zero embeddings
        return torch.zeros_like(hidden_states)
