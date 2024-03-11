# Copyright (c) 2023, Tri Dao.

import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import GPT2Config, OPTConfig


def remap_state_dict_hf_opt(state_dict, config):
    def key_mapping_model(key):
        key = re.sub(r"^model.decoder.", "transformer.", key)
        # The OPT-350m model uses '^decoder' instead of '^model.decoder'
        key = re.sub(r"^decoder.", "transformer.", key)
        return key

    state_dict = OrderedDict((key_mapping_model(k), v) for k, v in state_dict.items())
    # Word embedding and position embedding
    def key_mapping_emb(key):
        key = re.sub(r"^transformer.embed_tokens.", "transformer.embeddings.word_embeddings.", key)
        # The OPT-350m model uses has project_in and project_out
        key = re.sub(r"^transformer.project_in.", "transformer.embeddings.project_in.", key)
        key = re.sub(r"^transformer.project_out.", "project_out.", key)
        key = re.sub(
            r"^transformer.embed_positions.", "transformer.embeddings.position_embeddings.", key
        )
        return key

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    # OPT uses the first 2 indices of pos_emb for padding tokens
    pos_embeddings = state_dict.pop("transformer.embeddings.position_embeddings.weight")
    state_dict["transformer.embeddings.position_embeddings.weight"] = pos_embeddings[2:]
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = math.ceil(config.vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.final_layer_norm.", r"transformer.ln_f.", key)
        # The OPT-175B checkpoint calls this 'decoder.layer_norm' instead of 'decoder.final_layer_norm'
        key = re.sub(r"^transformer.layer_norm.", r"transformer.ln_f.", key)
        key = re.sub(
            r"^transformer.layers.(\d+).self_attn_layer_norm.", r"transformer.layers.\1.norm1.", key
        )
        key = re.sub(
            r"^transformer.layers.(\d+).final_layer_norm.", r"transformer.layers.\1.norm2.", key
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).fc(1|2).", r"transformer.layers.\1.mlp.fc\2.", key
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f"transformer.layers.{l}.self_attn.q_proj.weight")
        Wk = state_dict.pop(f"transformer.layers.{l}.self_attn.k_proj.weight")
        Wv = state_dict.pop(f"transformer.layers.{l}.self_attn.v_proj.weight")
        bq = state_dict.pop(f"transformer.layers.{l}.self_attn.q_proj.bias")
        bk = state_dict.pop(f"transformer.layers.{l}.self_attn.k_proj.bias")
        bv = state_dict.pop(f"transformer.layers.{l}.self_attn.v_proj.bias")
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.bias"] = torch.cat([bq, bk, bv], dim=0)

    def key_mapping_attn(key):
        return re.sub(
            r"^transformer.layers.(\d+).self_attn.out_proj.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def opt_config_to_gpt2_config(opt_config: OPTConfig) -> GPT2Config:
    assert opt_config.layerdrop == 0.0
    assert opt_config.layer_norm_elementwise_affine
    word_embed_proj_dim = (
        None
        if opt_config.word_embed_proj_dim == opt_config.hidden_size
        else opt_config.word_embed_proj_dim
    )
    return GPT2Config(
        vocab_size=opt_config.vocab_size,
        n_positions=opt_config.max_position_embeddings,
        n_embd=opt_config.hidden_size,
        n_layer=opt_config.num_hidden_layers,
        n_head=opt_config.num_attention_heads,
        n_inner=opt_config.ffn_dim,
        activation_function=opt_config.activation_function,
        resid_pdrop=opt_config.dropout,
        # HF's implementation of OPT doesn't seem to have embedding dropout
        embd_pdrop=opt_config.dropout,
        attn_pdrop=opt_config.attention_dropout,
        initializer_range=opt_config.init_std,
        bos_token_id=opt_config.bos_token_id,
        eos_token_id=opt_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        prenorm=opt_config.do_layer_norm_before,
        word_embed_proj_dim=word_embed_proj_dim,
    )
