# Copyright (c) 2023, GGGGGGXY, Tri Dao.

import math
import json
import re
from pathlib import Path

from collections import OrderedDict

import torch
import torch.nn.functional as F

from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig


def remap_state_dict_hf_baichuan(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^model.", "transformer.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        return re.sub(
            r"^transformer.embed_tokens.",
            "transformer.embeddings.word_embeddings.",
            key,
        )

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple)
        * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict[
            "transformer.embeddings.word_embeddings.weight"
        ]
    else:
        output_embeddings = state_dict.pop("lm_head.weight")
        # Need to recompute vocab_size since Baichuan shards the word embeddings and output embeddings
        # differently.
        vocab_size = (
            math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.norm.", r"transformer.ln_f.", key)
        key = re.sub(
            r"^transformer.layers.(\d+).input_layernorm.",
            r"transformer.layers.\1.norm1.",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).post_attention_layernorm.",
            r"transformer.layers.\1.norm2.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.n_layer):
        w1 = state_dict.pop(f"transformer.layers.{l}.mlp.gate_proj.weight")
        w3 = state_dict.pop(f"transformer.layers.{l}.mlp.up_proj.weight")
        # Our ordering is different
        state_dict[f"transformer.layers.{l}.mlp.fc1.weight"] = torch.cat(
            [w3, w1], dim=0
        )

    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).mlp.down_proj.",
            r"transformer.layers.\1.mlp.fc2.",
            key,
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).self_attn.W_pack.",
            r"transformer.layers.\1.mixer.Wqkv.",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).self_attn.o_proj.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    for l in range(config.n_layer):
        # pop rotary_emb.inv_freq from state dict
        state_dict.pop(f"transformer.layers.{l}.self_attn.rotary_emb.inv_freq", None)
    return state_dict


def baichuan_config_to_gpt2_config(baichuan_config: PretrainedConfig) -> GPT2Config:
    # HACK: the config doesn't have say whether it's rotary or alibi.
    # So we have to infer from the hidden size (7B -> rotary, 13B -> alibi).
    # HACK: the config doesn't have say whether it uses norm head.
    # So we have to infer from the vocab size
    # (v1, vocab size 64k, no norm head; v2, vocab size 128k, norm head).
    use_rotary = baichuan_config.hidden_size < 5000
    return GPT2Config(
        vocab_size=baichuan_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=baichuan_config.hidden_size,
        n_layer=baichuan_config.num_hidden_layers,
        n_head=baichuan_config.num_attention_heads,
        n_inner=baichuan_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # baichuan doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=baichuan_config.rms_norm_eps,
        initializer_range=baichuan_config.initializer_range,
        bos_token_id=baichuan_config.bos_token_id,
        eos_token_id=baichuan_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=baichuan_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0 if use_rotary else 0.0,
        rotary_emb_interleaved=False,
        use_alibi=not use_rotary,
        use_flash_attn=not use_rotary,  # Alibi code path requires flash_attn
        tie_word_embeddings=False,
        norm_head=baichuan_config.vocab_size > 70000,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
    )
