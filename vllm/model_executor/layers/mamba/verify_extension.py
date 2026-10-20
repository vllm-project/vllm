# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Extension point for verifying non-chain drafts on gated delta net layers.

GDN layers verify speculative tokens as a chain: row ``i`` of a request's query
continues from row ``i - 1``, each row writes its own recurrent / convolution
state to ``state_indices[req, i]``, and the next step resumes from row
``num_accepted_tokens - 1``. A draft *tree* needs row ``i`` to continue from
another earlier row instead, which the in-tree kernels cannot express.

A speculative method registered with ``register_speculative_method`` can set
``SpeculativeMethodSpec.linear_attn_verify_extension`` to a factory for a
``LinearAttnVerifyExtension``. Every ``GDNAttentionMetadataBuilder`` then
creates one and attaches it to the metadata it builds, and the GDN layer hands
the spec-decode rows of every forward to ``forward_spec_rows`` instead of
running its own convolution and recurrence on them. With no extension (the
default) nothing changes.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.spec_decode.registry import get_speculative_method

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class GDNSpecRows:
    """The spec-decode rows of one GDN layer's forward.

    ``T`` is the number of spec-decode tokens and ``B`` the number of their
    requests (a CUDA graph batch may pad ``B``; the in-tree kernels see the
    same padding). Heads and dims are per tensor-parallel rank.
    """

    layer_name: str
    """The layer's prefix, its key in the forward context."""
    mixed_qkv: torch.Tensor
    """``[T, conv_dim]`` convolution input (the q, k and v columns). May be a
    view whose row stride is larger than ``conv_dim``."""
    a: torch.Tensor
    """``[T, num_v_heads]`` decay gate input. May be a strided view."""
    b: torch.Tensor
    """``[T, num_v_heads]`` beta gate input. May be a strided view."""
    conv_state: torch.Tensor
    """``[num_slots, conv_dim, kernel_size - 1]`` convolution states."""
    conv_weight: torch.Tensor
    """``[conv_dim, kernel_size]``."""
    conv_bias: torch.Tensor | None
    activation: str | None
    ssm_state: torch.Tensor
    """``[num_slots, num_v_heads, head_v_dim, head_k_dim]`` recurrent states."""
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    state_indices: torch.Tensor
    """``[B, max_query_len]``: the state slot of each row of each request."""
    num_accepted_tokens: torch.Tensor
    """``[B]``: request ``b`` starts from ``state_indices[b, n - 1]``."""
    query_start_loc: torch.Tensor
    """``[B + 1]``: the rows of each request in ``mixed_qkv``."""
    out: torch.Tensor
    """``[T, num_v_heads, head_v_dim]``, written by the extension."""


class LinearAttnVerifyExtension:
    """Runs the convolution and the recurrence of a GDN layer's spec-decode
    rows in place of the in-tree chain kernels.

    State layout contract (the in-tree chain one): row ``i`` of request ``b``
    stores its convolution and recurrent state at ``state_indices[b, i]``; the
    initial state of request ``b`` is at ``state_indices[b,
    num_accepted_tokens[b] - 1]``. A tree verifier keeps that layout and
    computes row ``i`` from the row it continues; before the next step it moves
    the accepted path's states into chain order.
    """

    def forward_spec_rows(self, rows: GDNSpecRows) -> None:
        """Write the attention output of every spec-decode row to ``rows.out``
        and every row's states as described above. Called for every forward
        with spec-decode rows, including CUDA graph captures."""
        raise NotImplementedError


def create_linear_attn_verify_extension(
    vllm_config: "VllmConfig",
) -> LinearAttnVerifyExtension | None:
    """The extension of the registered speculative method, if it has one."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None:
        return None
    registered = get_speculative_method(speculative_config.method)
    if registered is None or registered.linear_attn_verify_extension is None:
        return None
    factory = resolve_obj_by_qualname(
        registered.linear_attn_verify_extension.replace(":", ".")
    )
    return factory(vllm_config)
