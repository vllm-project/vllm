# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPMRobot model: MiniCPM-V 4.6 VLM + DiTActionPooler (action head in pooler).

This model inherits from MiniCPMV4_6ForConditionalGeneration and uses
DiTActionPooler so the vLLM engine runs the full Vision-LLM-DiT pipeline.
The pooler receives VLM hidden_states and produces actions directly.
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.minicpmv import MiniCPMVDummyInputsBuilder
from vllm.model_executor.models.minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
    MiniCPMV4_6MultiModalProcessor,
    MiniCPMV4_6ProcessingInfo,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    MiniCPMV4_6MultiModalProcessor,
    info=MiniCPMV4_6ProcessingInfo,
    dummy_inputs=MiniCPMVDummyInputsBuilder,
)
class MiniCPMRobotForHiddenStates(
    MiniCPMV4_6ForConditionalGeneration,
    VllmModelForPooling,
):
    """MiniCPM-V 4.6 VLM with DiT ActionHead in pooler.

    llm.encode() with extra_kwargs["robot_state"] produces action tensors.
    Requires action_head_cfg in config.json. See
    docs/models/pooling_models/embed.md for usage.
    """

    is_pooling_model = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        hf_cfg = vllm_config.model_config.hf_config
        ah_cfg = getattr(hf_cfg, "action_head_cfg", None) or {}

        if not ah_cfg:
            raise ValueError(
                "MiniCPMRobotForHiddenStates requires action_head_cfg "
                "in config.json. Set it via --hf-overrides. "
                "See documentation for usage."
            )

        from vllm.model_executor.layers.pooler.dit_action import (
            DiTActionPooler,
        )

        diffusion_cfg = ah_cfg.get("diffusion_model_cfg", None)
        self.pooler = DiTActionPooler(
            action_dim=ah_cfg.get("action_dim", 80),
            state_dim=ah_cfg.get("state_dim", 80),
            action_horizon=ah_cfg.get("action_horizon", 30),
            num_inference_timesteps=ah_cfg.get("num_inference_timesteps", 4),
            num_target_vision_tokens=ah_cfg.get("num_target_vision_tokens", 32),
            max_seq_len=ah_cfg.get("max_seq_len", 1024),
            proprio_inject=ah_cfg.get("proprio_inject", "concat"),
            prediction_type=ah_cfg.get("prediction_type", "clean_action"),
            diffusion_model_cfg=diffusion_cfg,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        """Load VLM and ActionHead weights.

        VLM weights use the vLLM-internal format.
        ActionHead weights are delegated to the pooler.
        """
        loaded: set[str] = set()
        model_sd = self.state_dict()
        ah_weights: dict[str, torch.Tensor] = {}
        ah_prefix = "pooler.action_head."

        for key, tensor in weights:
            if key.startswith(ah_prefix):
                ah_name = key[len(ah_prefix) :]
                ah_weights[ah_name] = tensor
                continue
            if key not in model_sd:
                logger.debug("Skipping unknown VLM weight key: %s", key)
                continue
            target = model_sd[key]
            if tensor.shape == target.shape:
                target.copy_(tensor.to(target.dtype))
                loaded.add(key)
            elif (
                key
                in (
                    "language_model.model.embed_tokens.weight",
                    "language_model.lm_head.weight",
                )
                and tensor.ndim == 2
                and tensor.shape[0] <= target.shape[0]
            ):
                target[: tensor.shape[0]].copy_(tensor.to(target.dtype))
                loaded.add(key)
            elif "language_model.model.embed_tokens.weight" in key:
                vocab_size = min(tensor.shape[0], target.shape[0])
                target[:vocab_size].copy_(tensor[:vocab_size].to(target.dtype))
                loaded.add(key)

        if ah_weights:
            loaded_ah = self.pooler.load_weights(ah_weights)
            loaded.update(f"{ah_prefix}{k}" for k in loaded_ah)

        return loaded
