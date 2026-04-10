# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from collections.abc import Callable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class CustomCallableProposer:
    """Proposer that delegates draft generation to a user-provided Python function.

    The user function must have the signature::

        def generate_drafts(
            batch_input_ids: list[list[int]],
            draft_len: int,
            **kwargs,
        ) -> torch.Tensor:

    and return an int64 tensor of shape ``[batch_size, draft_len]``.

    The function is resolved at construction time from the module path stored in
    ``speculative_config.custom_proposer_backend`` (e.g.
    ``"my_module.my_draft_func"``).
    """

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        spec_config = vllm_config.speculative_config

        backend = spec_config.custom_proposer_backend
        assert backend is not None

        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Dynamically import the user-provided function.
        if "." not in backend:
            raise ValueError(
                f"Invalid custom_proposer_backend '{backend}'. "
                "It must be a full module path (e.g., 'module.function')."
            )

        module_path, func_name = backend.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import module '{module_path}' for custom proposer "
                f"'{backend}': {e}"
            ) from e

        user_func: Callable | None = getattr(module, func_name, None)
        if user_func is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{func_name}' "
                f"(custom_proposer_backend='{backend}')"
            )
        self.user_func = user_func
        self._backend = backend

        logger.info(
            "CustomCallableProposer loaded function '%s' with "
            "num_speculative_tokens=%d",
            backend,
            self.num_speculative_tokens,
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec:int,
        token_ids_cpu:torch.Tensor,
        slot_mappings:Optional[torch.Tensor] = None,
    ) -> list[list[int]]:
        """Generate draft tokens using the user-provided function.

        Args:
            sampled_token_ids: Recently sampled token IDs per request.
            num_tokens_no_spec: Number of non-speculative tokens per request.
            token_ids_cpu: Full token IDs tensor on CPU.
            slot_mappings: Slot mapping for KV cache (unused).

        Returns:
            List of draft token sequences for each request.
        """
        batch_size = len(sampled_token_ids)

        # Build batch_input_ids from token_ids_cpu (numpy array on CPU).
        if token_ids_cpu is None:
            raise RuntimeError(
                "token_ids_cpu is required for CustomCallableProposer but is None. "
                "This proposer requires full sequence history to be available on CPU."
            )

        batch_input_ids = [
            token_ids_cpu[i, : num_tokens_no_spec[i]].tolist()
            for i in range(batch_size)
        ]

        # Call the user function.
        try:
            draft_tokens_tensor = self.user_func(
                batch_input_ids=batch_input_ids,
                draft_len=self.num_speculative_tokens,
            )
        except Exception as e:
            raise RuntimeError(
                f"Custom proposer function '{self._backend}' raised an error: {e}"
            ) from e

        # Validate return type.
        if not isinstance(draft_tokens_tensor, torch.Tensor):
            raise TypeError(
                f"Custom proposer function '{self._backend}' must return a "
                f"torch.Tensor, got {type(draft_tokens_tensor).__name__}"
            )

        # Validate shape: [batch_size, draft_len] and dtype.
        expected_shape = (batch_size, self.num_speculative_tokens)
        if draft_tokens_tensor.shape != expected_shape:
            raise ValueError(
                f"Custom proposer function '{self._backend}' returned tensor "
                f"of shape {draft_tokens_tensor.shape}, expected {expected_shape}. "
                f"The function must return a tensor of shape [batch_size, draft_len]."
            )

        if draft_tokens_tensor.dtype != torch.long:
            raise TypeError(
                f"Custom proposer function '{self._backend}' must return a "
                f"torch.long tensor, got {draft_tokens_tensor.dtype}."
            )

        # Convert to list[list[int]] — .cpu() handles both CPU and GPU tensors
        # without an extra device transfer.
        return draft_tokens_tensor.cpu().tolist()

    def load_model(self, *args, **kwargs):
        """No model to load for custom callable proposer."""
