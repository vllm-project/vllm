# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 DSpark draft model for semi-autoregressive speculative decoding.

DSpark drafts a whole block in one parallel pass (DFlash-style: context-KV
precompute + a non-causal query-block forward) and then injects intra-block
dependency with a lightweight sequential Markov head.

The DFlash/DSpark draft backbone is always a standard dense transformer
(DFlashQwen3Model), regardless of whether the target model uses a hybrid
architecture (e.g. Qwen3.5's GDN + Transformer).  The draft model reuses
the same ``Qwen3DSparkModel`` / ``Qwen3DSparkForCausalLM`` classes from
``qwen3_dspark.py``; only the architecture name for registry lookup differs.

See :mod:`qwen3_dspark.py` for the full implementation.
"""

from vllm.model_executor.models.qwen3_dspark import (
    Qwen3DSparkForCausalLM,
    Qwen3DSparkModel,
)


# Re-export under Qwen3.5-specific names for registry / auto-detection.
class Qwen3_5DSparkModel(Qwen3DSparkModel):
    """Alias of Qwen3DSparkModel for Qwen3.5 target models.

    The DFlash draft backbone is architecture-agnostic; Qwen3.5 targets
    use the same draft model as Qwen3 targets.  This class exists solely
    so the speculative config auto-detection can match
    ``"Qwen3_5DSparkModel"`` in the draft model's ``architectures`` field.
    """

    pass


class Qwen3_5DSparkForCausalLM(Qwen3DSparkForCausalLM):
    """Alias of Qwen3DSparkForCausalLM for Qwen3.5 target models.

    See :class:`Qwen3_5DSparkModel` for rationale.
    """

    pass
