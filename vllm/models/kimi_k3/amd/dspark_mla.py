# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 DSpark draft model on ROCm.

Reuses the shared draft model (including its MLA layers) from
`nvidia/dspark_mla.py` and adds only the vocabulary-sharded Markov sampler
(`ops/dspark_markov.py`)."""

from ..nvidia.dspark_mla import K3DSparkForCausalLM as _K3DSparkForCausalLM
from .ops.dspark_markov import ShardedMarkovSampler


class K3DSparkForCausalLM(_K3DSparkForCausalLM):
    def setup_sample_sequential(self, speculator) -> None:
        """Called by DSparkSpeculator.load_model on every TP rank, before graph
        capture. Installs the vocabulary-sharded Markov sampler as
        `sample_sequential` when it applies."""
        sampler = ShardedMarkovSampler.create(speculator)
        if sampler is not None:
            self.sample_sequential = sampler
