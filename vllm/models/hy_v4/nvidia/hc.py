# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""iHC (independent Hyper-Connections) layers for HY V4 (NVIDIA).

iHC replaces the single residual stream of a standard transformer with
``hc_mult`` parallel residual channels. Each decoder sub-block reduces the
channels to one hidden state (``HYV4HCPreLayer``), runs the sub-block, then
scatters the result back over the channels (``HYV4HCPostLayer``). The final
``HYV4HCHeadLayer`` merges the channels before the model's output norm.

NOTE: Each of the three steps has an optional single-kernel HPC replacement
(``HpcIHCPre`` / ``HpcIHCPost`` / ``HpcIHCHead``). They are only constructed
when the hpc package is installed, ``VLLM_ENABLE_HPC_OPS=1`` and the shape /
device constraints hold; otherwise the eager path below runs unchanged.
TODO: port the cross-layer post+pre fusion (``HpcIHCPostPre``) as well; it
requires restructuring the decoder-layer forward scheduling.
"""

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.model_executor.layers.hpc import HpcIHCHead, HpcIHCPost, HpcIHCPre
from vllm.model_executor.layers.linear import ReplicatedLinear


class HYV4HCPreLayer(nn.Module):
    """iHC pre-processing layer (2D-activation adaptation).

    Steps:
        1. RMS-normalize the flattened ``[num_tokens, hc * d]`` input.
        2. Project to the pre/post gating logits.
        3. Turn the logits into sigmoid gates.
        4. Reduce over the channel dim with the pre gates.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_dim: int,
        hc_mult: int = 4,
        magnitude: float = 2.0,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        hc_eps: float = 1e-6,
        layernorm_epsilon: float = 1e-5,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim
        self.hc_mult = hc_mult
        self.magnitude = magnitude
        self.hc_eps = hc_eps
        self.layernorm_epsilon = layernorm_epsilon
        hc_in_dim = hc_mult * hidden_dim
        mix_hc = 2 * hc_mult  # pre + post only, no comb

        self.hc_fn = ReplicatedLinear(
            input_size=hc_in_dim,
            output_size=mix_hc,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_fn",
        )
        self.hc_scale = nn.Parameter(torch.empty(2, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))

        self.reset_parameters(init_std, base_noise_std)

        # Optional single-kernel HPC replacement for the whole forward below.
        # ``norm_owner`` is left unset: the RMSNorm that follows the pre block
        # lives in the decoder layer and is not folded in yet.
        self.hpc_op: HpcIHCPre | None = None
        if HpcIHCPre.support(hc_mult, hidden_dim):
            self.hpc_op = HpcIHCPre(
                hc_mult=hc_mult,
                hidden_size=hidden_dim,
                magnitude=magnitude,
                hc_eps=hc_eps,
                norm_eps=layernorm_epsilon,
                fallback_op=self,
            )

    def reset_parameters(self, init_std: float, base_noise_std: float = 0.0) -> None:
        """Initialize the gate scale and per-channel gate bias."""
        del init_std  # hc_fn is initialized by ReplicatedLinear
        nn.init.constant_(self.hc_scale, 0.01)
        with torch.no_grad():
            self.hc_base[: self.hc_mult].fill_(
                -torch.log(torch.tensor(self.hc_mult - 1.0, dtype=self.hc_base.dtype))
            )
            self.hc_base[self.hc_mult : 2 * self.hc_mult].fill_(0.0)
            if base_noise_std > 0.0:
                self.hc_base.add_(torch.randn_like(self.hc_base) * base_noise_std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce the iHC channels and emit the post gates.

        Args:
            x: Input of shape ``[num_tokens, hc, d]``.

        Returns:
            A tuple of the pre-gated reduction ``[num_tokens, d]`` and the post
            gates ``[num_tokens, hc]`` consumed by `HYV4HCPostLayer`.
        """
        if self.hpc_op is not None:
            return self.hpc_op(x)

        shape = x.size()  # [num_tokens, hc, d]
        hc = self.hc_mult
        hc_eps = self.hc_eps

        x_flat = x.flatten(1).float()  # [num_tokens, hc*d]

        rsqrt = torch.rsqrt(
            x_flat.square().mean(-1, keepdim=True) + self.layernorm_epsilon
        )
        mixes = self.hc_fn(x_flat)[0] * rsqrt  # [num_tokens, 2*hc]

        pre_raw = mixes[..., :hc]
        post_raw = mixes[..., hc : 2 * hc]

        pre = (
            torch.sigmoid(
                pre_raw * self.hc_scale[0].float() + self.hc_base[:hc].float()
            )
            + hc_eps
        )
        post = (
            self.magnitude
            * torch.sigmoid(
                post_raw * self.hc_scale[1].float() + self.hc_base[hc : 2 * hc].float()
            )
            + hc_eps
        )

        y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
        return y.to(x.dtype), post


class HYV4HCPostLayer(nn.Module):
    """iHC post-processing layer (2D-activation adaptation).

    Applies post-gating to the sub-block output and adds the multi-channel
    residual (no comb mixing)::

        y[n, i, d] = post[n, i] * x[n, d] + residual[n, i, d]
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.config = config

        # Optional single-kernel HPC replacement for the whole forward below.
        # Only constructed under enable_ihc (see HYV4HCLayer), so hc_mult is
        # present; getattr keeps this robust if that ever changes.
        self.hpc_op: HpcIHCPost | None = None
        hc_mult = getattr(config, "hc_mult", 0)
        if HpcIHCPost.support(hc_mult, config.hidden_size):
            self.hpc_op = HpcIHCPost(hc_mult=hc_mult, hidden_size=config.hidden_size)

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor
    ) -> torch.Tensor:
        """Scatter the sub-block output back onto the iHC channels.

        Args:
            x: Attention/MLP output of shape ``[num_tokens, d]``.
            residual: Multi-channel residual ``[num_tokens, hc, d]``.
            post: Post gates ``[num_tokens, hc]`` from `HYV4HCPreLayer`.

        Returns:
            The updated residual channels ``[num_tokens, hc, d]``.
        """
        if self.hpc_op is not None:
            return self.hpc_op(x, residual, post)

        dtype = x.dtype
        x = x.float()
        residual = residual.float()
        post = post.float()

        post_gated = post.unsqueeze(-1) * x.unsqueeze(-2)  # [num_tokens, hc, d]
        y = post_gated + residual
        return y.to(dtype)


class HYV4HCHeadLayer(nn.Module):
    """iHC head layer (2D-activation adaptation).

    Merges the iHC channels back into a single hidden state before the final
    layer norm, using an RMS-normed projection plus sigmoid-gated reduction.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_head_fn = ReplicatedLinear(
            input_size=hc_mult * hidden_size,
            output_size=hc_mult,
            params_dtype=torch.float32,
            bias=False,
            prefix=f"{prefix}.hc_head_fn",
        )
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

        self.reset_parameters(init_std, base_noise_std)

        # Optional single-kernel HPC replacement for the whole forward below.
        self.hpc_op: HpcIHCHead | None = None
        if HpcIHCHead.support(hc_mult, hidden_size):
            self.hpc_op = HpcIHCHead(
                hc_mult=hc_mult,
                hidden_size=hidden_size,
                hc_eps=hc_eps,
                norm_eps=config.rms_norm_eps,
                fallback_op=self,
            )

    def reset_parameters(
        self, init_std: float = 6e-3, base_noise_std: float = 0.0
    ) -> None:
        """Initialize the head gate scale and per-channel gate bias."""
        del init_std  # hc_head_fn is initialized by ReplicatedLinear
        nn.init.constant_(self.hc_head_scale, 0.01)
        with torch.no_grad():
            self.hc_head_base.fill_(
                -torch.log(
                    torch.tensor(self.hc_mult - 1.0, dtype=self.hc_head_base.dtype)
                )
            )
            if base_noise_std > 0.0:
                self.hc_head_base.add_(
                    torch.randn_like(self.hc_head_base) * base_noise_std
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge the iHC channels into a single hidden state.

        Args:
            x: Input of shape ``[num_tokens, hc, d]``.

        Returns:
            The merged hidden state ``[num_tokens, d]``.
        """
        if self.hpc_op is not None:
            return self.hpc_op(x)

        shape, x_dtype = x.size(), x.dtype

        x = x.flatten(1).float()  # [num_tokens, hc*d]
        rsqrt = torch.rsqrt(
            x.square().mean(-1, keepdim=True) + self.config.rms_norm_eps
        )
        mixes = self.hc_head_fn(x)[0] * rsqrt  # [num_tokens, hc]
        pre = (
            torch.sigmoid(
                mixes * self.hc_head_scale.float() + self.hc_head_base.float()
            )
            + self.hc_eps
        )
        y = torch.sum(pre.unsqueeze(-1) * x.reshape(shape), dim=1)  # [num_tokens, d]
        return y.to(x_dtype)


class HYV4HCLayer(nn.Module):
    """Wrapper owning one iHC boundary (pre + post) of a decoder sub-block."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        init_std: float = 6e-3,
        base_noise_std: float = 0.0,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.enable_ihc = getattr(config, "enable_ihc", False)
        if self.enable_ihc:
            self.hc_pre = HYV4HCPreLayer(
                config,
                config.hidden_size,
                config.hc_mult,
                config.hc_magnitude,
                init_std,
                base_noise_std,
                config.hc_eps,
                config.rms_norm_eps,
                prefix=f"{prefix}.hc_pre",
            )
            self.hc_post = HYV4HCPostLayer(config)

    def prepare_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the sub-block input to 3D when iHC is enabled."""
        if not self.enable_ihc:
            return hidden_states
        return self._prepare_input_to_3d(hidden_states)

    def _prepare_input_to_3d(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Reshape the iHC input to ``[num_tokens, hc, h]``.

        Accepted inputs are ``[num_tokens, hc, h]`` (no-op), ``[num_tokens, h]``
        (broadcast over the channels) and ``[num_tokens, hc * h]`` (reshape).
        """
        if hidden_states.dim() == 3:
            return hidden_states
        if hidden_states.dim() != 2:
            raise RuntimeError(
                f"HC expects a 2D/3D tensor, got shape={tuple(hidden_states.shape)}"
            )

        d0, d1 = hidden_states.shape
        h = self.config.hidden_size
        hc = self.config.hc_mult

        if d1 == h:
            return hidden_states.unsqueeze(1).repeat(1, hc, 1)

        expected = hc * h
        if d1 == expected:
            return hidden_states.reshape(d0, hc, h)

        raise RuntimeError(
            f"HC expects last dim to be hidden_size ({h})"
            f"or hc_mult*hidden_size ({expected}), got {d1}. "
            f"hc_mult={hc}, hidden_size={h}."
        )

    def pre(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Reduce the iHC channels and produce the post gates.

        Returns:
            A tuple of the reduced hidden states ``[num_tokens, d]``, the post
            gates ``[num_tokens, hc]`` (``None`` when iHC is disabled) and the
            residual (the untouched input).
        """
        if not self.enable_ihc:
            return hidden_states, None, hidden_states
        reduced, post_gates = self.hc_pre(hidden_states)
        return reduced, post_gates, hidden_states

    def post(
        self,
        output_with_bias: torch.Tensor,
        residual: torch.Tensor,
        post_gates: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply post-gating and add the residual."""
        if not self.enable_ihc:
            return output_with_bias + residual
        assert post_gates is not None
        return self.hc_post(output_with_bias, residual, post_gates)
