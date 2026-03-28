# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field


class SetSteeringRequest(BaseModel):
    vectors: dict[str, dict[int, list[float]]] = Field(
        description="Steering vectors keyed by hook point name "
        "(pre_attn, post_attn, post_mlp_pre_ln, post_mlp_post_ln), "
        "then layer index. Each vector must have length equal to "
        "the model's hidden_size.",
    )
    scales: dict[int, float] | None = Field(
        default=None,
        description="Optional mapping from layer index to scale factor. "
        "Defaults to 1.0 for any layer not specified.",
    )
    replace: bool = Field(
        default=False,
        description="When True, clears all existing steering vectors "
        "before applying the new ones, making the operation an atomic "
        "replacement. When False (default), only the specified layers "
        "are updated and other layers keep their current state.",
    )
