# coding=utf-8
# Copied from
# https://huggingface.co/mosaicml/mpt-7b/blob/main/configuration_mpt.py
"""A HuggingFace-style model configuration."""
import warnings
from typing import Any, Dict, Optional, Union
from transformers import PretrainedConfig

attn_config_defaults: Dict = {
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
ffn_config_defaults: Dict = {'ffn_type': 'mptmlp'}
init_config_defaults: Dict = {
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
                 attn_config: Dict = attn_config_defaults,
                 ffn_config: Dict = ffn_config_defaults,
                 init_device: str = 'cpu',
                 logit_scale: Optional[Union[float, str]] = None,
                 no_bias: bool = False,
                 embedding_fraction: float = 1.0,
                 norm_type: str = 'low_precision_layernorm',
                 use_cache: bool = False,
                 init_config: Dict = init_config_defaults,
                 fc_type: str = 'torch',
                 verbose: Optional[int] = None,
                 **kwargs: Any):
        """The MPT configuration class.
        Args:
            d_model (int): The size of the embedding dimension of the model.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of layers in the model.
            expansion_ratio (int): The ratio of the up/down scale in the ffn.
            max_seq_len (int): The maximum sequence length of the model.
            vocab_size (int): The size of the vocabulary.
            resid_pdrop (float): The dropout probability applied to the attention output before combining with residual.
            emb_pdrop (float): The dropout probability for the embedding layer.
            learned_pos_emb (bool): Whether to use learned positional embeddings
            attn_config (Dict): A dictionary used to configure the model's attention module:
                attn_type (str): type of attention to use. Options: multihead_attention, multiquery_attention, grouped_query_attention
                attn_pdrop (float): The dropout probability for the attention layers.
                attn_impl (str): The attention implementation to use. One of 'torch', 'flash', or 'triton'.
                qk_ln (bool): Whether to apply layer normalization to the queries and keys in the attention layer.
                clip_qkv (Optional[float]): If not None, clip the queries, keys, and values in the attention layer to
                    this value.
                softmax_scale (Optional[float]): If not None, scale the softmax in the attention layer by this value. If None,
                    use the default scale of ``1/sqrt(d_keys)``.
                prefix_lm (Optional[bool]): Whether the model should operate as a Prefix LM. This requires passing an
                    extra `prefix_mask` argument which indicates which tokens belong to the prefix. Tokens in the prefix
                    can attend to one another bi-directionally. Tokens outside the prefix use causal attention.
                attn_uses_sequence_id (Optional[bool]): Whether to restrict attention to tokens that have the same sequence_id.
                    When the model is in `train` mode, this requires passing an extra `sequence_id` argument which indicates
                    which sub-sequence each token belongs to.
                    Defaults to ``False`` meaning any provided `sequence_id` will be ignored.
                alibi (bool): Whether to use the alibi bias instead of position embeddings.
                alibi_bias_max (int): The maximum value of the alibi bias.
                kv_n_heads (Optional[int]): For grouped_query_attention only, allow user to specify number of kv heads.
            ffn_config (Dict): A dictionary used to configure the model's ffn module:
                ffn_type (str): type of ffn to use. Options: mptmlp, te_ln_mlp
            init_device (str): The device to use for parameter initialization.
            logit_scale (Optional[Union[float, str]]): If not None, scale the logits by this value.
            no_bias (bool): Whether to use bias in all layers.
            verbose (int): The verbosity level. 0 is silent.
            embedding_fraction (float): The fraction to scale the gradients of the embedding layer by.
            norm_type (str): choose type of norm to use
            use_cache (bool): Whether or not the model should return the last key/values attentions
            init_config (Dict): A dictionary used to configure the model initialization:
                init_config.name: The parameter initialization scheme to use. Options: 'default_', 'baseline_',
                    'kaiming_uniform_', 'kaiming_normal_', 'neox_init_', 'small_init_', 'xavier_uniform_', or
                    'xavier_normal_'. These mimic the parameter initialization methods in PyTorch.
                init_div_is_residual (Union[int, float, str, bool]): Value to divide initial weights by if ``module._is_residual`` is True.
                emb_init_std (Optional[float]): The standard deviation of the normal distribution used to initialize the embedding layer.
                emb_init_uniform_lim (Optional[Union[Tuple[float, float], float]]): The lower and upper limits of the uniform distribution
                    used to initialize the embedding layer. Mutually exclusive with ``emb_init_std``.
                init_std (float): The standard deviation of the normal distribution used to initialize the model,
                    if using the baseline_ parameter initialization scheme.
                init_gain (float): The gain to use for parameter initialization with kaiming or xavier initialization schemes.
                fan_mode (str): The fan mode to use for parameter initialization with kaiming initialization schemes.
                init_nonlinearity (str): The nonlinearity to use for parameter initialization with kaiming initialization schemes.
                ---
                See llmfoundry.models.utils.param_init_fns.py for info on other param init config options
            fc_type (str): choose fc layer implementation. Options: torch and te. te layers support fp8 when using H100 GPUs.
        """
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
                'verbose argument for MPTConfig is now ignored and will be removed. Use python_log_level instead.'
            ),
                          stacklevel=2)
        if 'name' in kwargs:
            del kwargs['name']
        if 'loss_fn' in kwargs:
            del kwargs['loss_fn']
        if self.attn_config.get('alibi', False):
            self.learned_pos_emb = False
            warnings.warn(
                f'alibi is turned on, setting `learned_pos_emb` to {self.learned_pos_emb}`',
                stacklevel=2)
        super().__init__(**kwargs)
        self._validate_config()

    def _set_config_defaults(
            self, config: Dict[str, Any],
            config_defaults: Dict[str, Any]) -> Dict[str, Any]:
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
        if any((
                prob < 0 or prob > 1 for prob in
            [self.attn_config['attn_pdrop'], self.resid_pdrop, self.emb_pdrop]
        )):
            raise ValueError(
                "self.attn_config['attn_pdrop'], resid_pdrop, emb_pdrop are probabilities and must be between 0 and 1"  # pylint: disable=line-too-long
            )
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
                'attn_uses_sequence_id only implemented with torch and triton attention.'  # pylint: disable=line-too-long
            )
        if self.embedding_fraction > 1 or self.embedding_fraction <= 0:
            raise ValueError(
                'model.embedding_fraction must be between 0 (exclusive) and 1 (inclusive)!'  # pylint: disable=line-too-long
            )
        if isinstance(self.logit_scale,
                      str) and self.logit_scale != 'inv_sqrt_d_model':
            raise ValueError(
                f"self.logit_scale={self.logit_scale!r} is not recognized as an option; use numeric value or 'inv_sqrt_d_model'."  # pylint: disable=line-too-long
            )
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
                    # pylint: disable=line-too-long
                    'TransformerEngine import fail. `fc_type: te` requires TransformerEngine be installed. '
                    +
                    'The required version of transformer_engine also requires FlashAttention v1.0.6 is installed:\n'
                    + 'pip install flash-attn==1.0.6 --no-build-isolation \n' +
                    'pip install git+https://github.com/NVIDIA/TransformerEngine.git@144e4888b2cdd60bd52e706d5b7a79cb9c1a7156'
                ) from exc
        if self.ffn_config['ffn_type'] == 'mptmlp':
            self.ffn_config['fc_type'] = self.fc_type
        elif self.ffn_config['ffn_type'] == 'te_ln_mlp':
            self.ffn_config['bias'] = not self.no_bias
