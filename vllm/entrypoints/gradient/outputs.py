# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gradient computation output types and deserialization.

Contains the data classes returned by the gradient API and a factory
function that reconstructs typed outputs from the serialized dict
produced by gpu_worker.py.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class GradientOutput:
    """The output data of a gradient computation request.

    Args:
        token_log_probs: Per-target-token log-probabilities
            log p(y_t | x, y_{<t}) for each target token.
        token_attributions: Attribution matrix. Shape depends on aggregation:
            - aggregated (l2_norm/abs_sum): [num_selected_targets, num_tokens]
            - full (none): [num_selected_targets, num_tokens, hidden_dim]
            where num_tokens depends on gradient_targets.
        loss: Scalar loss value (when gradient_of is "loss" or "both").
        loss_gradients: Gradient of loss w.r.t. each gradient target.
            Keys are gradient target names, values are tensors.
    """

    token_log_probs: list[float] | None = None
    token_attributions: np.ndarray | None = None
    loss: float | None = None
    loss_gradients: dict[str, np.ndarray] | None = None

    def __repr__(self) -> str:
        attr_shape = None
        if self.token_attributions is not None:
            attr_shape = self.token_attributions.shape
        return (
            f"GradientOutput(loss={self.loss}, "
            f"token_log_probs_len="
            f"{len(self.token_log_probs) if self.token_log_probs else None}, "
            f"token_attributions_shape={attr_shape})"
        )


class GradientRequestOutput:
    """The output data of a gradient computation request to the LLM.

    Args:
        request_id: A unique identifier for the gradient request.
        outputs: The gradient computation results.
        prompt_token_ids: The token IDs of the prompt.
        target_token_ids: The target token IDs for gradient computation.
        finished: Whether the gradient computation is completed.
    """

    def __init__(
        self,
        request_id: str,
        outputs: GradientOutput,
        prompt_token_ids: list[int],
        target_token_ids: list[int],
        finished: bool,
    ):
        self.request_id = request_id
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids
        self.target_token_ids = target_token_ids
        self.finished = finished

    def __repr__(self) -> str:
        return (
            f"GradientRequestOutput(request_id={self.request_id!r}, "
            f"outputs={self.outputs!r}, "
            f"prompt_token_ids=[{len(self.prompt_token_ids)} tokens], "
            f"target_token_ids=[{len(self.target_token_ids)} tokens], "
            f"finished={self.finished})"
        )


def gradient_request_output_from_engine_dict(
    external_req_id: str,
    gradient_output: dict[str, Any],
    prompt_token_ids: list[int],
    finished: bool,
) -> GradientRequestOutput:
    """Reconstruct a GradientRequestOutput from the serialized dict.

    The engine core serializes gradient outputs as a plain dict of
    bytes/metadata (via gpu_worker.py). This function deserializes
    the numpy arrays and constructs the typed output.
    """
    token_attributions = None
    if "token_attributions_bytes" in gradient_output:
        token_attributions = np.frombuffer(
            gradient_output["token_attributions_bytes"],
            dtype=np.dtype(gradient_output["token_attributions_dtype"]),
        ).reshape(gradient_output["token_attributions_shape"])

    loss_gradients = None
    if "loss_gradients_packed" in gradient_output:
        loss_gradients = {}
        for k, packed in gradient_output["loss_gradients_packed"].items():
            loss_gradients[k] = np.frombuffer(
                packed["bytes"],
                dtype=np.dtype(packed["dtype"]),
            ).reshape(packed["shape"])

    return GradientRequestOutput(
        request_id=external_req_id,
        outputs=GradientOutput(
            token_log_probs=gradient_output.get("token_log_probs"),
            token_attributions=token_attributions,
            loss=gradient_output.get("loss"),
            loss_gradients=loss_gradients,
        ),
        prompt_token_ids=prompt_token_ids,
        target_token_ids=gradient_output.get("target_token_ids", []),
        finished=finished,
    )
