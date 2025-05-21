# SPDX-License-Identifier: Apache-2.0

# Copied from
# https://huggingface.co/mosaicml/mpt-7b/blob/main/configuration_mpt.py
"""A HuggingFace-style model configuration."""
import warnings
from typing import Any, Optional, Union

from transformers import PretrainedConfig

attn_config_defaults: dict = {
    'attn_type': 'multihead_attention',
    'attn_pdrop': 0.0,
    'attn_impl': 'triton',
    'qk_ln': False,
    'clip_qkv': None,
    'softmax_scale': None,
    'prefix_lm': False,
    'attn_uses_sequence_id': False,
    'alibi': False,
    'alibi_bias_max': 8
}
ffn_config_defaults: dict = {'ffn_type': 'mptmlp'}
init_config_defaults: dict = {
    'name': 'kaiming_normal_',
    'fan_mode': 'fan_in',
    'init_nonlinearity': 'relu',
    'init_div_is_residual': True,
    'emb_init_std': None,
    'emb_init_uniform_lim': None,
    'init_std': None,
    'init_gain': 0.0
}


class MPTConfig(PretrainedConfig):
    model_type = 'mpt'
    attribute_map = {
        'num_attention_heads': 'n_heads',
        'hidden_size': 'd_model',
        'num_hidden_layers': 'n_layers',
    }

    # pylint: disable=dangerous-default-value
    def __init__(self,
                 d_model: int = 2048,
                 n_heads: int = 16,
                 n_layers: int = 24,
                 expansion_ratio: int = 4,
                 max_seq_len: int = 2048,
                 vocab_size: int = 50368,
                 resid_pdrop: float = 0.0,
                 emb_pdrop: float = 0.0,
                 learned_pos_emb: bool = True,
                 attn_config: dict = attn_config_defaults,
                 ffn_config: dict = ffn_config_defaults,
                 init_device: str = 'cpu',
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = False,
                 embedding_fraction: float = 1.0,
                 norm_type: str = 'low_precision_layernorm',
                 use_cache: bool = False,
                 init_config: dict = init_config_defaults,
                 fc_type: str = 'torch',
                 verbose: Optional[int] = None,
                 **kwargs: Any):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.expansion_ratio = expansion_ratio
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.resid_pdrop = resid_pdrop
        self.emb_pdrop = emb_pdrop
        self.learned_pos_emb = learned_pos_emb
        self.attn_config = attn_config
        self.ffn_config = ffn_config
        self.init_device = init_device
        self.logit_scale = logit_scale
        self.no_bias = no_bias
        self.embedding_fraction = embedding_fraction
        self.norm_type = norm_type
        self.use_cache = use_cache
        self.init_config = init_config
        self.fc_type = fc_type
        if verbose is not None:
            warnings.warn(DeprecationWarning(
                'verbose argument for MPTConfig is now ignored and '
                'will be removed. Use python_log_level instead.'),
                          stacklevel=2)
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        if self.attn_config.get('alibi', False):
            self.learned_pos_emb = False
            warnings.warn(
                f'alibi is turned on, setting `learned_pos_emb` '
                f'to {self.learned_pos_emb}`',
                stacklevel=2)
        super().__init__(**kwargs)
        self._validate_config()

    def _set_config_defaults(
            self, config: dict[str, Any],
            config_defaults: dict[str, Any]) -> dict[str, Any]:
        for (k, v) in config_defaults.items():
            if k not in config:
                config[k] = v
        return config

    def _validate_config(self) -> None:
        self.attn_config = self._set_config_defaults(self.attn_config,
                                                     attn_config_defaults)
        self.ffn_config = self._set_config_defaults(self.ffn_config,
                                                    ffn_config_defaults)
        self.init_config = self._set_config_defaults(self.init_config,
                                                     init_config_defaults)
        if self.d_model % self.n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        if any(
                prob < 0 or prob > 1 for prob in
            [self.attn_config['attn_pdrop'], self.resid_pdrop, self.emb_pdrop
             ]):
            raise ValueError(
                "self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are "
                "probabilities and must be between 0 and 1")
        if self.attn_config['attn_impl'] not in ['torch', 'flash', 'triton']:
            raise ValueError(
                f"Unknown attn_impl={self.attn_config['attn_impl']}")
        if self.attn_config['prefix_lm'] and self.attn_config[
                'attn_impl'] not in ['torch', 'triton']:
            raise NotImplementedError(
                'prefix_lm only implemented with torch and triton attention.')
        if self.attn_config['alibi'] and self.attn_config['attn_impl'] not in [
                'torch', 'triton'
        ]:
            raise NotImplementedError(
                'alibi only implemented with torch and triton attention.')
        if self.attn_config['attn_uses_sequence_id'] and self.attn_config[
                'attn_impl'] not in ['torch', 'triton']:
            raise NotImplementedError(
                'attn_uses_sequence_id only implemented with torch '
                'and triton attention.')
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError(
                'model.embedding_fraction must be between 0 (exclusive) '
                'and 1 (inclusive)!')
        if isinstance(self.logit_scale,
                      str) and self.logit_scale != 'inv_sqrt_d_model':
            raise ValueError(
                f"self.logit_scale={self.logit_scale!r} is not recognized as "
                "an option; use numeric value or 'inv_sqrt_d_model'.")
        if self.init_config.get('name', None) is None:
            raise ValueError(
                f"self.init_config={self.init_config!r} 'name' needs to be set."
            )
        if not self.learned_pos_emb and (not self.attn_config['alibi']):
            warnings.warn(
                'Positional information not being provided to the model.',
                stacklevel=2)
        if self.fc_type == 'te' or self.ffn_config['ffn_type'] == 'te_ln_mlp':
            try:
                # pylint: disable=import-outside-toplevel
                import transformer_engine.pytorch as te
                del te
            except Exception as exc:
                raise ImportError(
                    'TransformerEngine import fail. `fc_type: te` requires '
                    'TransformerEngine be installed. '
                    'The required version of transformer_engine also requires '
                    'FlashAttention v1.0.6 is installed:\n'
                    'pip install flash-attn==1.0.6 --no-build-isolation \n'
                    'pip install git+https://github.com/NVIDIA/TransformerEngine.git@144e4888b2cdd60bd52e706d5b7a79cb9c1a7156'
                ) from exc
        if self.ffn_config['ffn_type'] == 'mptmlp':
            self.ffn_config['fc_type'] = self.fc_type
        elif self.ffn_config['ffn_type'] == 'te_ln_mlp':
            self.ffn_config['bias'] = not self.no_bias
