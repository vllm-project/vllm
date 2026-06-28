# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for AFD (Attention FFN Disaggregation) distributed
computation."""

import hashlib
from dataclasses import field
from typing import Any, Literal

from vllm.config.utils import config


@config
class AFDConfig:
    """Configuration for AFD (Attention FFN Disaggregation) distributed
    computation."""

    afd_connector: str = "dummy"
    """The AFD connector for vLLM to communicate between attention and FFN
    nodes. Available connectors: 'dummy', 'p2pconnector'"""

    afd_role: Literal["attention", "ffn"] = "attention"
    """Role of this vLLM instance in AFD. 'attention' for attention workers,
    'ffn' for FFN servers."""

    afd_port: int = 1239
    """Port number for stepmesh parameter server communication."""

    afd_host: str = "127.0.0.1"
    """Host address for stepmesh parameter server communication."""

    num_afd_stages: int = 3
    """Number of pipeline stages for stage parallelism."""

    num_attention_servers: int = 1
    """Number of attention servers."""

    num_ffn_servers: int = 1
    """Number of FFN servers."""

    afd_server_rank: int = 0
    """Rank of this AFD server."""

    afd_extra_config: dict[str, Any] = field(default_factory=dict)
    """Extra configuration for specific AFD connectors."""

    compute_gate_on_attention: bool = False
    """Whether to compute the gate on the attention side."""

    multistream_info: dict[str, Any] = field(
        default_factory=lambda: {
            "attn_enable": "False",
            "attn_core_num": "8",
            "ffn_enable": "False",
            "ffn_core_num": "8",
        })
    """
    MultiStream configuration (A-side and F-side independently):
        - attn_enable / attn_core_num: attention-side a2e comm overlap
        - ffn_enable / ffn_core_num:   ffn-side e2a comm overlap
    """

    quant_mode: int = 0
    """Quant mode of this AFD connector."""

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        # AFD configuration affects the computation graph structure
        # as it changes how FFN computation is performed
        factors: list[Any] = [
            self.afd_connector,
            self.afd_role,
            self.num_afd_stages,
            self.num_attention_servers,
            self.num_ffn_servers,
        ]
        return hashlib.sha256(str(factors).encode()).hexdigest()

    @property
    def is_attention_server(self) -> bool:
        """Check if this instance is configured as an attention server."""
        return self.afd_role == "attention"

    @property
    def is_ffn_server(self) -> bool:
        """Check if this instance is configured as an FFN server."""
        return self.afd_role == "ffn"

    @property
    def is_attn_multistream(self) -> bool:
        return str(self.multistream_info["attn_enable"]) == "True"

    @property
    def is_ffn_multistream(self) -> bool:
        return str(self.multistream_info["ffn_enable"]) == "True"

    @property
    def attn_core_num(self) -> int:
        return int(self.multistream_info["attn_core_num"])

    @property
    def ffn_core_num(self) -> int:
        return int(self.multistream_info["ffn_core_num"])
