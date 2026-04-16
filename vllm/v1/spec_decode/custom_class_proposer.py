# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class CustomClassProposer:
    """Proposer that delegates draft generation to a user-provided Proposer class.

    The user class must implement the interface::

        class MyCustomProposer:
            def __init__(self, vllm_config: VllmConfig): ...

            def propose(
                self,
                sampled_token_ids: list[list[int]],
                num_tokens_no_spec: int,
                token_ids_cpu: torch.Tensor,
                slot_mappings: torch.Tensor | None = None,
            ) -> list[list[int]]: ...

    The `propose` method must return a list of draft token sequences for each
    request. Optionally, the class may implement a `load_model` method.

    The class is resolved at construction time from the module path stored in
    ``speculative_config.custom_proposer_backend`` (e.g.
    ``"my_module.MyCustomProposer"``).
    """

    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        spec_config = vllm_config.speculative_config

        backend = spec_config.custom_proposer_backend
        assert backend is not None

        self.num_speculative_tokens = spec_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len

        # Dynamically import the user-provided class.
        if "." not in backend:
            raise ValueError(
                f"Invalid custom_proposer_backend '{backend}'. "
                "It must be a full module path (e.g., 'module.MyProposerClass')."
            )

        module_path, class_name = backend.rsplit(".", 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Cannot import module '{module_path}' for custom proposer "
                f"'{backend}': {e}"
            ) from e

        user_class = getattr(module, class_name, None)
        if user_class is None:
            raise AttributeError(
                f"Module '{module_path}' has no attribute '{class_name}' "
                f"(custom_proposer_backend='{backend}')"
            )

        # Instantiate the user-provided class.
        try:
            self.user_proposer = user_class(vllm_config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate custom proposer class '{backend}': {e}. "
                "The class constructor must accept VllmConfig as argument."
            ) from e

        # Verify the class has the required propose method.
        if not hasattr(self.user_proposer, "propose"):
            raise AttributeError(
                f"Custom proposer class '{backend}' must have a 'propose' method."
            )
        if not callable(self.user_proposer.propose):
            raise AttributeError(
                f"Custom proposer class '{backend}' has a 'propose' attribute "
                "but it is not callable."
            )

        self._backend = backend

        logger.info(
            "CustomClassProposer loaded class '%s' with num_speculative_tokens=%d",
            backend,
            self.num_speculative_tokens,
        )

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: int,
        token_ids_cpu: torch.Tensor,
        slot_mappings: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """Generate draft tokens using the user-provided proposer instance.

        Args:
            sampled_token_ids: Recently sampled token IDs per request.
            num_tokens_no_spec: Number of non-speculative tokens per request.
            token_ids_cpu: Full token IDs tensor on CPU.
            slot_mappings: Slot mapping for KV cache (optional).

        Returns:
            List of draft token sequences for each request.
        """
        # Delegate to the user-provided proposer instance.
        try:
            return self.user_proposer.propose(
                sampled_token_ids=sampled_token_ids,
                num_tokens_no_spec=num_tokens_no_spec,
                token_ids_cpu=token_ids_cpu,
                slot_mappings=slot_mappings,
            )
        except Exception as e:
            raise RuntimeError(
                f"Custom proposer class '{self._backend}' raised an error in "
                f"propose(): {e}"
            ) from e

    def load_model(self, *args, **kwargs):
        """Delegate load_model to the user-provided proposer if available."""
        if hasattr(self.user_proposer, "load_model"):
            return self.user_proposer.load_model(*args, **kwargs)
        # No model to load if user class doesn't implement load_model.
