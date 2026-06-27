# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AFDConnectorBase Class for Distributed AFD FFN computation

The class provides the four core AFD communication interfaces:
1. send_attn_output(): Send attention output to FFN servers (Attention Worker)
2. recv_ffn_output(): Receive FFN computation result (Attention Worker)
3. recv_attn_output(): Receive attention output from workers (FFN Server)
4. send_ffn_output(): Send FFN computation result back (FFN Server)
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .metadata import AFDConnectorMetadata


class AFDConnectorBase(ABC):
    """
    Abstract base class for AFD connectors.

    This provides the four core interfaces for AFD communication between
    attention workers and FFN servers.
    """

    @abstractmethod
    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: "VllmConfig",
    ):
        """Initialize the AFD connector.

        Args:
            rank: Global rank of this process
            local_rank: Local rank within the node
            config: VllmConfig containing AFDConfig
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the connector and release resources."""
        raise NotImplementedError

    @abstractmethod
    def init_afd_connector(self) -> None:
        """Initialize the AFD connector."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the connector is initialized and ready to use.

        Returns:
            bool: True if the connector is initialized, False otherwise.
        """
        raise NotImplementedError

    def get_connector_rank(self) -> int:
        """Get the rank of this connector."""
        return getattr(self, "rank", 0)

    def get_connector_local_rank(self) -> int:
        """Get the local rank of this connector."""
        return getattr(self, "local_rank", 0)

    def configure_metadata(self, metadata: Optional["AFDConnectorMetadata"],
                           **kwargs) -> None:
        """
        Allow connector to inject specific data into metadata.
        Base implementation does nothing.
        """
        return None

    @abstractmethod
    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: Optional["AFDConnectorMetadata"],
        **kwargs
    ) -> Any:
        """Send attention output to FFN servers.

        Args:
            hidden_states: Attention output tensor
            metadata: AFD metadata containing layer_idx, stage_idx, seq_len info
            **kwargs: Additional arguments required by specific connectors 
                      (e.g. topk_weights, topk_ids, router_logits)

        Returns:
            Any: Handle for tracking this request (backend-specific)
        """
        raise NotImplementedError

    @abstractmethod
    def recv_ffn_output(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        metadata: Optional["AFDConnectorMetadata"] = None
    ) -> Optional[torch.Tensor]:
        """Wait for and receive FFN computation result.

        Args:
            hidden_states: Optional hidden states tensor (used by some connectors)
            metadata: Optional metadata

        Returns:
            torch.Tensor: FFN computation result.
        """
        raise NotImplementedError

    @abstractmethod
    def recv_attn_output(
        self,
        metadata: Optional["AFDConnectorMetadata"] = None,
        **kwargs
    ) -> Any:
        """Receive attention output from attention workers.

        Args:
            metadata: Optional metadata
            **kwargs: Additional arguments

        Returns:
            tuple: (hidden_states, metadata)
                - hidden_states: Concatenated attention outputs
                - metadata: Inferred AFD metadata containing
                            seq_lens and other info
        """
        raise NotImplementedError

    @abstractmethod
    def send_ffn_output(
        self,
        hidden_states: torch.Tensor,
        metadata: Optional["AFDConnectorMetadata"],
        **kwargs
    ) -> None:
        """Send FFN computation result back to attention workers.

        Args:
            hidden_states: Computed FFN result
            metadata: AFD metadata containing seq_lens
                      for splitting and routing info
            **kwargs: Additional arguments
        """
        raise NotImplementedError

    def compute_moe(
        self,
        experts: torch.nn.Module,
        hidden_states: torch.Tensor,
        **kwargs
    ) -> Any:
        """
        Perform MoE computation via the connector (or delegate to experts).
        Default implementation calls experts.afd_ffn_compute.
        Connectors can override to call different methods on experts.
        """
        return experts.afd_ffn_compute(
            layer=experts,
            hidden_states=hidden_states,
            **kwargs
        )

    def select_experts(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Any] = None,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select experts for MoE.

        Args:
            hidden_states: Input hidden states
            router_logits: Router logits
            top_k: Number of experts to select
            use_grouped_topk: Whether to use grouped topk
            renormalize: Whether to renormalize weights
            topk_group: Number of groups for topk
            num_expert_group: Number of expert groups
            custom_routing_function: Custom routing function
            e_score_correction_bias: Bias for score correction

        Returns:
             tuple: (topk_weights, topk_ids, row_idx)
        """
        raise NotImplementedError("select_experts not implemented for this connector")
