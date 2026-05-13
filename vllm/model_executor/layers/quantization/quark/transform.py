# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class OrthogonalTransform(torch.nn.Module):
    def __init__(
        self,
        input_rotation: torch.Tensor,
        rotation_config: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.input_rotation = input_rotation
        self.rotation_size = input_rotation.shape[0]
        self.rotation_config = rotation_config

    def forward(self, x: torch.Tensor):
        needs_reshape = False
        if x.shape[-1] != self.rotation_size:
            needs_reshape = True
            x = x.reshape(*x.shape[:-1], -1, self.rotation_size)

        x = x @ self.input_rotation

        if needs_reshape:
            x = x.reshape(*x.shape[:-2], -1)

        return x

    @staticmethod
    def setup_transform(quant_config: dict[str, Any], layer_names: list[str]):
        use_online_rotation = False
        rotation_config = None
        rotation_size = None

        if (
            quant_config.get("algo_config") is not None
            and len(quant_config["algo_config"]) > 0
            and quant_config["algo_config"][0]["name"] == "rotation"
        ):
            rotation_config = quant_config["algo_config"][0]

            online_rotation_layers = rotation_config["online_config"][
                "online_rotation_layers"
            ]

            if online_rotation_layers is not None and any(
                layer_name in online_rotation_layers for layer_name in layer_names
            ):
                use_online_rotation = True
                rotation_size = rotation_config["rotation_size"]

                if rotation_size is None:
                    raise NotImplementedError("rotation_size=None is not supported")

        return use_online_rotation, rotation_config, rotation_size

    def post_process_transform(self):
        if self.rotation_config is not None and not self.rotation_config["trainable"]:
            # In case hadamard transform is used (non-trained case), it is
            # serialized as torch.int8 with only `-1` and `1` values.
            self.input_rotation.data = self.input_rotation.data.to(
                torch.float
            ) / math.sqrt(self.rotation_size)

        rotation_dtype = torch.get_default_dtype()
        self.input_rotation.data = self.input_rotation.data.to(rotation_dtype)

        logger.debug("self.rotation_size: %s", self.rotation_size)
        logger.debug("self.input_rotation.data: %s", self.input_rotation.data)


def rotation_weight_loader(
    param: torch.nn.Parameter,
    loaded_weight: torch.Tensor,
    weight_name: str | None = None,
    shard_id: str | None = None,
    expert_id: int | None = None,
):
    assert param.shape == loaded_weight.shape
    assert param.dtype == loaded_weight.dtype
    param.data.copy_(loaded_weight)
