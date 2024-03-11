# Copyright (c) 2023, Tri Dao.

import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from einops import rearrange
from transformers import FalconConfig, GPT2Config


def remap_state_dict_hf_falcon(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^transformer.h.", "transformer.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())
    # Word embedding
    def key_mapping_emb(key):
        return re.sub(
            r"^transformer.word_embeddings.", "transformer.embeddings.word_embeddings.", key
        )

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]
    else:
        output_embeddings = state_dict.pop("lm_head.weight")
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )
        output_embeddings_bias = state_dict.pop("lm_head.bias")
        state_dict["lm_head.bias"] = F.pad(
            output_embeddings_bias, (0, vocab_size - output_embeddings_bias.shape[0])
        )

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(
            r"^transformer.layers.(\d+).input_layernorm.", r"transformer.layers.\1.norm1.", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).post_attention_layernorm.",
            r"transformer.layers.\1.norm2.",
            key,
        )
        key = re.sub(r"^transformer.layers.(\d+).ln_attn.", r"transformer.layers.\1.norm1.", key)
        key = re.sub(r"^transformer.layers.(\d+).ln_mlp.", r"transformer.layers.\1.norm2.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_h_to_4h.", r"transformer.layers.\1.mlp.fc1.", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.dense_4h_to_h.", r"transformer.layers.\1.mlp.fc2.", key
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.query_key_value.",
            r"transformer.layers.\1.mixer.Wqkv.",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attention.dense.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    n_head = config.n_head
    n_head_kv = getattr(config, "n_head_kv", 1)
    headdim = config.hidden_size // n_head
    for l in range(config.n_layer):
        # The weights are stored in a different layout compared to our implementation
        Wqkv = rearrange(
            state_dict.pop(f"transformer.layers.{l}.mixer.Wqkv.weight"),
            "(group ratio headdim) ... -> group ratio headdim ...",
            ratio=n_head // n_head_kv + 2,
            headdim=headdim,
        )
        Wq = rearrange(Wqkv[:, :-2], "group ratio headdim ... -> (group ratio headdim) ...")
        Wk = rearrange(Wqkv[:, [-2]], "group ratio headdim ... -> (group ratio headdim) ...")
        Wv = rearrange(Wqkv[:, [-1]], "group ratio headdim ... -> (group ratio headdim) ...")
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)

    return state_dict


def falcon_config_to_gpt2_config(falcon_config: FalconConfig) -> GPT2Config:
    # The 40b config uses "n_head_kv" instead of "num_kv_heads"
    n_head_kv = getattr(
        falcon_config,
        "n_head_kv",
        1 if getattr(falcon_config, "multi_query", False) else falcon_config.n_head,
    )
    # HACK: the 40b config has 2 LN per layer instead of 1, but that's not reflected in the config.
    # So we have to infer it from the number of heads in the key/value block
    parallel_block_tied_norm = n_head_kv == 1
    return GPT2Config(
        vocab_size=falcon_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=falcon_config.hidden_size,
        n_layer=falcon_config.n_layer,
        n_head=falcon_config.n_head,
        n_inner=falcon_config.hidden_size * 4,
        activation_function="gelu",
        resid_pdrop=falcon_config.hidden_dropout,
        embd_pdrop=0.0,  # There doesn't seem to be any embedding dropout
        attn_pdrop=falcon_config.attention_dropout,
        layer_norm_epsilon=falcon_config.layer_norm_epsilon,
        initializer_range=falcon_config.initializer_range,
        bos_token_id=falcon_config.bos_token_id,
        eos_token_id=falcon_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        parallel_block=falcon_config.parallel_attn,
        n_head_kv=n_head_kv,
        parallel_block_tied_norm=parallel_block_tied_norm,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=False,
        tie_word_embeddings=True,
        qkv_proj_bias=falcon_config.bias,
        out_proj_bias=falcon_config.bias,
        mlp_fc1_bias=falcon_config.bias,
        mlp_fc2_bias=falcon_config.bias,
        lm_head_bias=False,
    )
