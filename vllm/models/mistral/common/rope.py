# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from transformers import PreTrainedConfig

from vllm.model_executor.layers.rotary_embedding.common import (
    yarn_find_correction_range,
    yarn_linear_ramp_mask,
)
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)


class MistralRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """Interleaved (GPT-J style) YaRN RoPE with optional LeRoPE.

    LeRoPE multiplies the extrapolated and interpolated inverse frequencies by
    ``exp(alpha)``, where ``alpha`` is a checkpoint parameter shared by all
    layers. The YaRN ramp keeps using the unscaled frequencies. The cos/sin
    cache is rebuilt by :meth:`set_lerope_alpha` once the weights are loaded.
    """

    def __init__(self, config: PreTrainedConfig, rotary_dim: int) -> None:
        rope_parameters = config.rope_parameters
        is_yarn = rope_parameters.get("rope_type", "default") == "yarn"
        # YaRN magnitude scaling (`apply_scale`) is not used by Mistral models.
        assert not is_yarn or rope_parameters.get("attention_factor") == 1.0
        self.lerope_alpha: torch.Tensor | None = None
        super().__init__(
            head_size=rotary_dim,
            rotary_dim=rotary_dim,
            max_position_embeddings=(
                rope_parameters["original_max_position_embeddings"]
                if is_yarn
                else config.max_position_embeddings
            ),
            base=rope_parameters["rope_theta"],
            is_neox_style=False,
            scaling_factor=rope_parameters["factor"] if is_yarn else 1.0,
            dtype=torch.get_default_dtype(),
            beta_fast=rope_parameters.get("beta_fast", 32),
            beta_slow=rope_parameters.get("beta_slow", 1),
            mscale=rope_parameters.get("mscale", 1),
            mscale_all_dim=rope_parameters.get("mscale_all_dim", 0),
        )
        # `apply_scale: false` checkpoints must not rescale cos/sin.
        assert not is_yarn or self.mscale == 1.0

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        if self.lerope_alpha is not None:
            lerope_scale = torch.exp(self.lerope_alpha.to(pos_freqs))
            inv_freq_extrapolation = inv_freq_extrapolation * lerope_scale
            inv_freq_interpolation = inv_freq_interpolation * lerope_scale
        # The YaRN ramp is defined on the frequencies without LeRoPE.
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        extrapolation_mask = 1 - yarn_linear_ramp_mask(
            low, high, self.rotary_dim // 2, dtype=torch.float
        )
        return (
            inv_freq_interpolation * (1 - extrapolation_mask)
            + inv_freq_extrapolation * extrapolation_mask
        )

    @torch.no_grad()
    def set_lerope_alpha(self, lerope_alpha: torch.Tensor) -> None:
        assert lerope_alpha.shape == (self.rotary_dim // 2,)
        self.lerope_alpha = lerope_alpha.detach().float()
        cache = self._compute_cos_sin_cache()
        self.cos_sin_cache.copy_(cache.to(self.cos_sin_cache.dtype))
