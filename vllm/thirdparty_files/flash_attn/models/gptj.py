# Copyright (c) 2023, Tri Dao.

import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTJConfig


def remap_state_dict_hf_gptj(state_dict, config):
    def key_mapping_layers(key):
        return re.sub(r"^transformer.h.", "transformer.layers.", key)

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())
    # Word embedding
    def key_mapping_emb(key):
        return re.sub(r"^transformer.wte.", "transformer.embeddings.word_embeddings.", key)

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
        return re.sub(r"^transformer.layers.(\d+).ln_1.", r"transformer.layers.\1.norm1.", key)

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.fc_in.", r"transformer.layers.\1.mlp.fc1.", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.fc_out.", r"transformer.layers.\1.mlp.fc2.", key
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f"transformer.layers.{l}.attn.q_proj.weight")
        Wk = state_dict.pop(f"transformer.layers.{l}.attn.k_proj.weight")
        Wv = state_dict.pop(f"transformer.layers.{l}.attn.v_proj.weight")
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)
        # We don't store these biases
        state_dict.pop(f"transformer.layers.{l}.attn.bias")
        state_dict.pop(f"transformer.layers.{l}.attn.masked_bias")

    def key_mapping_attn(key):
        return re.sub(
            r"^transformer.layers.(\d+).attn.out_proj.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def gptj_config_to_gpt2_config(gptj_config: GPTJConfig) -> GPT2Config:
    headdim = gptj_config.n_embd // gptj_config.n_head
    return GPT2Config(
        vocab_size=gptj_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=gptj_config.n_embd,
        n_layer=gptj_config.n_layer,
        n_head=gptj_config.n_head,
        n_inner=gptj_config.n_inner,
        activation_function=gptj_config.activation_function,
        resid_pdrop=gptj_config.resid_pdrop,
        embd_pdrop=gptj_config.embd_pdrop,
        attn_pdrop=gptj_config.attn_pdrop,
        layer_norm_epsilon=gptj_config.layer_norm_epsilon,
        initializer_range=gptj_config.initializer_range,
        bos_token_id=gptj_config.bos_token_id,
        eos_token_id=gptj_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        prenorm=True,
        parallel_block=True,
        parallel_block_tied_norm=True,
        rotary_emb_fraction=gptj_config.rotary_dim / headdim,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        lm_head_bias=True,
    )
