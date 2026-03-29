# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pydantic import BaseModel, Field

from vllm.config.steering_types import SteeringVectorSpec


class SetSteeringRequest(BaseModel):
    vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Base steering vectors applied to both prefill and "
        "decode phases. Keyed by hook point name (pre_attn, post_attn, "
        "post_mlp_pre_ln, post_mlp_post_ln), then layer index. Values "
        "are either bare lists (scale=1.0) or "
        '{"vector": [...], "scale": float}.',
    )
    prefill_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Phase-specific steering vectors added to base during "
        "prefill only. Same format as vectors.",
    )
    decode_vectors: SteeringVectorSpec | None = Field(
        default=None,
        description="Phase-specific steering vectors added to base during "
        "decode only. Same format as vectors.",
    )
    replace: bool = Field(
        default=False,
        description="When True, clears all existing steering vectors "
        "before applying the new ones, making the operation an atomic "
        "replacement. When False (default), only the specified layers "
        "are updated and other layers keep their current state.",
    )
