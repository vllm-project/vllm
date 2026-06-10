# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def create_custom_proposer(vllm_config: VllmConfig):
    """Load and instantiate a user-provided proposer class.

    The class path is read from ``speculative_config.model``
    (e.g., ``"my_module.MyCustomProposer"``).  The class is
    imported, instantiated with *vllm_config*, and returned
    directly so the caller can use it without any wrapper.

    The returned object must expose a callable ``propose`` method.
    """
    assert vllm_config.speculative_config is not None
    spec_config = vllm_config.speculative_config

    backend = spec_config.model
    assert backend is not None

    if "." not in backend:
        raise ValueError(
            f"Invalid custom proposer module path '{backend}'. "
            "It must be a full module path (e.g., 'module.MyProposerClass')."
        )

    module_path, class_name = backend.rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Cannot import module '{module_path}' for custom proposer '{backend}': {e}"
        ) from e

    user_class = getattr(module, class_name, None)
    if user_class is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{class_name}' "
            f"(speculative_config.model='{backend}')"
        )

    try:
        instance = user_class(vllm_config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate custom proposer class '{backend}': {e}. "
            "The class constructor must accept VllmConfig as argument."
        ) from e

    if not hasattr(instance, "propose"):
        raise AttributeError(
            f"Custom proposer class '{backend}' must have a 'propose' method."
        )
    if not callable(instance.propose):
        raise AttributeError(
            f"Custom proposer class '{backend}' has a 'propose' attribute "
            "but it is not callable."
        )

    logger.info(
        "Loaded custom proposer class '%s' with num_speculative_tokens=%d",
        backend,
        spec_config.num_speculative_tokens,
    )

    return instance
