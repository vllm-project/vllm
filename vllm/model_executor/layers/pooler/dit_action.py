# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiTActionPooler: runs GR00tActionHead inside vLLM's pooling path.

VLM forward -> hidden_states -> pooler.forward() -> DiT denoising -> actions.
Robot state is passed through PoolingParams.extra_kwargs["robot_state"].
"""

from collections.abc import Set

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.minicpm_robot_action_head import (
    GR00tActionHead,
)
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

logger = init_logger(__name__)


class DiTActionPooler(Pooler):
    """Pooler that runs GR00T ActionHead (DiT) inside vLLM's pooling path.

    VLM hidden_states serve as DiT's cross-attention condition.
    Robot state is read from pooling_params.extra_kwargs["robot_state"].
    """

    def __init__(
        self,
        *,
        action_dim: int = 80,
        state_dim: int = 80,
        action_horizon: int = 30,
        num_inference_timesteps: int = 4,
        num_target_vision_tokens: int = 32,
        max_seq_len: int = 1024,
        proprio_inject: str = "concat",
        prediction_type: str = "clean_action",
        diffusion_model_cfg: dict | None = None,
        action_head_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.action_head = GR00tActionHead(
            hidden_size=1024,
            action_model_type="DiT-B",
            action_dim=action_dim,
            state_dim=state_dim,
            num_inference_timesteps=num_inference_timesteps,
            action_horizon=action_horizon,
            proprio_inject=proprio_inject,
            prediction_type=prediction_type,
            max_seq_len=max_seq_len,
            num_target_vision_tokens=num_target_vision_tokens,
            num_timestep_buckets=1000,
            add_pos_embed=True,
            diffusion_model_cfg=diffusion_model_cfg,
        )
        self.action_head = self.action_head.to(dtype=action_head_dtype)
        self.action_head.eval()

    def load_weights(self, state_dict: dict[str, torch.Tensor]) -> set[str]:
        """Load ActionHead weights from a state_dict.

        Keys are bare ActionHead parameter names (without
        "action_head." prefix). Returns the set of keys that
        were actually loaded.
        """
        model_sd = self.action_head.state_dict()
        loaded: set[str] = set()
        for key, tensor in state_dict.items():
            if key in model_sd and tensor.shape == model_sd[key].shape:
                model_sd[key].copy_(tensor.to(model_sd[key].dtype))
                loaded.add(key)
        return loaded

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"embed", "token_embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        """Run ActionHead on VLM hidden_states.

        Args:
            hidden_states: Full VLM hidden_states, shape
                [total_tokens, hidden_dim].
            pooling_metadata: Contains prompt_lens and per-request
                pooling_params.

        Returns:
            List of action tensors, one per request.
        """
        results = []
        offset = 0
        device = hidden_states.device

        for i, length in enumerate(pooling_metadata.prompt_lens):
            length = int(length)
            seq_hidden = hidden_states[offset : offset + length]
            offset += length

            pooling_params = pooling_metadata.pooling_params[i]
            extra = pooling_params.extra_kwargs or {}
            robot_state = extra.get("robot_state")

            if robot_state is None:
                logger.warning(
                    "robot_state is None in request %d; "
                    "returning zero actions. If this is not a warmup "
                    "request, ensure PoolingParams.extra_kwargs includes "
                    "'robot_state'.",
                    i,
                )
                results.append(
                    torch.zeros(
                        self.action_head.action_horizon,
                        self.action_head.action_dim,
                        device=device,
                        dtype=hidden_states.dtype,
                    )
                )
                continue

            if isinstance(robot_state, np.ndarray):
                robot_state = torch.from_numpy(np.ascontiguousarray(robot_state)).to(
                    device=device, dtype=hidden_states.dtype
                )
            elif isinstance(robot_state, list):
                robot_state = torch.tensor(
                    robot_state,
                    device=device,
                    dtype=hidden_states.dtype,
                )
            robot_state = robot_state.reshape(1, 1, -1)

            vl_embs = seq_hidden.unsqueeze(0).float()
            robot_state = robot_state.float()
            actions = self.action_head.predict_action(
                vl_embs=vl_embs,
                state=robot_state,
            )
            results.append(actions.squeeze(0))

        return results
