# Copyright (c) 2023, Tri Dao.

import math
import json
import re
from pathlib import Path

from collections import OrderedDict

import torch
import torch.nn.functional as F

from einops import rearrange
from transformers import GPT2Config, AutoConfig, PretrainedConfig


def remap_state_dict_hf_btlm(state_dict, config):
    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^transformer.wpe.", "transformer.embeddings.position_embeddings.", key)

    if "transformer.wpe.weight" in state_dict:
        state_dict = OrderedDict((key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.wte.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.ln_f.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(r"^transformer.h.(\d+).ln_(1|2).(weight|bias)", r"transformer.layers.\1.norm\2.\3", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for d in range(config.num_hidden_layers):
        W1 = state_dict.pop(f"transformer.h.{d}.mlp.c_fc.weight")
        W3 = state_dict.pop(f"transformer.h.{d}.mlp.c_fc2.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc1.weight"] = torch.cat([W1.t(), W3.t()], dim=0)
        b1 = state_dict.pop(f"transformer.h.{d}.mlp.c_fc.bias")
        b3 = state_dict.pop(f"transformer.h.{d}.mlp.c_fc2.bias")
        state_dict[f"transformer.layers.{d}.mlp.fc1.bias"] = torch.cat([b1, b3], dim=0)
        W2 = state_dict.pop(f"transformer.h.{d}.mlp.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mlp.fc2.weight"] = W2.t()

    def key_mapping_mlp(key):
        key = re.sub(r"^transformer.h.(\d+).mlp.c_proj.bias", r"transformer.layers.\1.mlp.fc2.bias", key)
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        Wqkv = state_dict.pop(f"transformer.h.{d}.attn.c_attn.weight")
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = Wqkv.t()
        Wout = state_dict.pop(f"transformer.h.{d}.attn.c_proj.weight")
        state_dict[f"transformer.layers.{d}.mixer.out_proj.weight"] = Wout.t()
    state_dict.pop(f"transformer.relative_pe.slopes")  # We don't store the Alibi slopes

    def key_mapping_attn(key):
        key = re.sub(r"^transformer.h.(\d+).attn.c_attn.bias", r"transformer.layers.\1.mixer.Wqkv.bias", key)
        key = re.sub(
            r"^transformer.h.(\d+).attn.c_proj.bias", r"transformer.layers.\1.mixer.out_proj.bias", key
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def btlm_config_to_gpt2_config(btlm_config: PretrainedConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=btlm_config.vocab_size,
        n_positions=0 if btlm_config.position_embedding_type == "alibi" else btlm_config.n_positions,
        n_embd=btlm_config.hidden_size,
        n_layer=btlm_config.num_hidden_layers,
        n_head=btlm_config.num_attention_heads,
        n_inner=btlm_config.n_inner,
        activation_function=btlm_config.activation_function,
        resid_pdrop=btlm_config.resid_pdrop,
        embd_pdrop=btlm_config.embd_pdrop,
        attn_pdrop=btlm_config.attn_pdrop,
        layer_norm_epsilon=btlm_config.layer_norm_epsilon,
        initializer_range=btlm_config.initializer_range,
        bos_token_id=btlm_config.bos_token_id,
        eos_token_id=btlm_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        use_alibi=btlm_config.position_embedding_type == "alibi",
        use_flash_attn=btlm_config.position_embedding_type == "alibi",  # Alibi code path requires flash_attn
        mup_width_scale=btlm_config.mup_width_scale,
        mup_embeddings_multiplier=btlm_config.mup_embeddings_scale,
        mup_output_multiplier=btlm_config.mup_output_alpha,
        mup_scale_qk_dot_by_d=btlm_config.mup_scale_qk_dot_by_d,
        mlp_multiple_of=1,
    )
