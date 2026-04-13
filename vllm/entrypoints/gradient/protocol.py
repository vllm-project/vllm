# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from pydantic import BaseModel, Field

GradientTarget = Literal["input_embeddings", "output_embeddings"]


def _default_gradient_targets() -> list[GradientTarget]:
    return ["input_embeddings"]


class GradientRequest(BaseModel):
    """Request body for the /v1/gradients endpoint."""

    model: str
    prompt: str
    target: str

    gradient_of: Literal[
        "loss",
        "token_log_probs",
        "both",
    ] = "token_log_probs"

    gradient_targets: list[GradientTarget] = Field(
        default_factory=_default_gradient_targets
    )

    target_token_indices: list[int] | None = None

    aggregation: Literal[
        "none",
        "l2_norm",
        "abs_sum",
    ] = "l2_norm"

    loss_function: Literal[
        "cross_entropy",
        "log_prob_sum",
    ] = "cross_entropy"

    return_log_probs: bool = True

    # Optional fields
    encoding: Literal["json", "base64_numpy"] = "json"


class GradientTokenLogProb(BaseModel):
    """Per-token log-probability."""

    token: str
    token_id: int
    log_prob: float


class GradientResponse(BaseModel):
    """Response body for the /v1/gradients endpoint."""

    id: str
    object: str = "gradient"
    model: str
    encoding: Literal["json", "base64_numpy"] = "json"

    token_log_probs: list[GradientTokenLogProb] | None = None

    token_attributions: dict | None = None

    loss: float | None = None
    loss_gradients: dict[str, dict] | None = None
