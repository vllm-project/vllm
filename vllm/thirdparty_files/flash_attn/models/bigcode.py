import math
import re
from collections import OrderedDict

import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPTBigCodeConfig, PretrainedConfig


def remap_state_dict_hf_bigcode(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a Huggingface BigCode model to be flash_attn compatible.
    """

    # Word embedding and position embedding
    def key_mapping_pos_emb(key):
        return re.sub(r"^transformer.wpe.", "transformer.embeddings.position_embeddings.", key)

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
        key = re.sub(
            r"^transformer.h.(\d+).ln_(1|2).(weight|bias)",
            r"transformer.layers.\1.norm\2.\3",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.h.(\d+).mlp.c_fc.weight",
            r"transformer.layers.\1.mlp.fc1.weight",
            key,
        )
        key = re.sub(
            r"^transformer.h.(\d+).mlp.c_proj.weight",
            r"transformer.layers.\1.mlp.fc2.weight",
            key,
        )
        key = re.sub(
            r"^transformer.h.(\d+).mlp.c_fc.bias",
            r"transformer.layers.\1.mlp.fc1.bias",
            key,
        )
        key = re.sub(
            r"^transformer.h.(\d+).mlp.c_proj.bias",
            r"transformer.layers.\1.mlp.fc2.bias",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # TODO: add support for multi-head attention
    assert config.multi_query, "Only multi-query attention is supported"

    # Attention
    for d in range(config.num_hidden_layers):
        embed_dim = config.n_embd
        head_dim = embed_dim // config.n_head

        c_attn_weight = state_dict.pop(f"transformer.h.{d}.attn.c_attn.weight")
        # with multi-query attention, the weights have shape (embed_dim, embed_dim + head_dim + head_dim)
        # see https://github.com/huggingface/transformers/blob/95b374952dc27d8511541d6f5a4e22c9ec11fb24/src/transformers/models/gpt_bigcode/modeling_gpt_bigcode.py#L112
        # see also https://github.com/ggerganov/ggml/blob/dd1d575956e54c5bdc07632f25506b3b1884dbd2/examples/starcoder/convert-hf-to-ggml.py#L183
        # ((n_head + 2) * head_dim, embed_dim) -> (3 * n_heads * head_dim, hidden_dim)
        q, k, v = torch.split(c_attn_weight, [embed_dim, head_dim, head_dim], dim=0)
        # duplicate k, v along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
        k = torch.tile(k, (config.n_head, 1))
        v = torch.tile(v, (config.n_head, 1))
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.weight"] = torch.cat((q, k, v), dim=0)

        # same deal with the bias
        c_attn_bias = state_dict.pop(f"transformer.h.{d}.attn.c_attn.bias")
        # ((n_head + 2) * head_dim, embed_dim) -> (3 * n_heads * head_dim, hidden_dim)
        q, k, v = torch.split(c_attn_bias, [embed_dim, head_dim, head_dim], dim=0)
        # duplicate k, v along the first axis (head_dim, hidden_dim) -> (n_heads * head_dim, hidden_dim)
        k = torch.tile(k, (config.n_head,))
        v = torch.tile(v, (config.n_head,))
        state_dict[f"transformer.layers.{d}.mixer.Wqkv.bias"] = torch.cat((q, k, v), dim=0)

    def key_mapping_attn(key):
        key = re.sub(
            r"^transformer.h.(\d+).attn.c_proj.weight",
            r"transformer.layers.\1.mixer.out_proj.weight",
            key,
        )
        key = re.sub(
            r"^transformer.h.(\d+).attn.c_proj.bias",
            r"transformer.layers.\1.mixer.out_proj.bias",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def inv_remap_state_dict_hf_bigcode(state_dict, config: PretrainedConfig):
    """
    Map the state_dict of a flash_attn model to be Huggingface BigCode compatible.

    This function is meant to be the inverse of remap_state_dict_hf_bigcode.
    """

    # Word embedding and position embeddings
    def inv_key_mapping_pos_emb(key):
        return re.sub(r"^transformer.embeddings.position_embeddings.", "transformer.wpe.", key)

    state_dict = OrderedDict((inv_key_mapping_pos_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")

    word_embeddings = word_embeddings[:, : config.vocab_size]
    state_dict["transformer.wte.weight"] = word_embeddings
    state_dict["lm_head.weight"] = word_embeddings

    # LayerNorm
    def inv_key_mapping_ln(key):
        key = re.sub(r"^transformer.ln_f.(weight|bias)", r"transformer.ln_f.\1", key)
        key = re.sub(
            r"^transformer.layers.(\d+).norm(1|2).(weight|bias)",
            r"transformer.h.\1.ln_\2.\3",
            key,
        )
        return key

    state_dict = OrderedDict((inv_key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLPs
    def inv_key_mapping_mlp(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.fc1.weight",
            r"transformer.h.\1.mlp.c_fc.weight",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.fc2.weight",
            r"transformer.h.\1.mlp.c_proj.weight",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.fc1.bias",
            r"transformer.h.\1.mlp.c_fc.bias",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mlp.fc2.bias",
            r"transformer.h.\1.mlp.c_proj.bias",
            key,
        )
        return key

    state_dict = OrderedDict((inv_key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for d in range(config.num_hidden_layers):
        embed_dim = config.n_embd
        head_dim = embed_dim // config.n_head

        Wqkv_weight = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.weight")
        q, k, v = torch.split(
            Wqkv_weight, [embed_dim, head_dim * config.n_head, head_dim * config.n_head], dim=0
        )
        c_attn_weight = torch.cat((q, k[:head_dim], v[:head_dim]), dim=0)
        state_dict[f"transformer.h.{d}.attn.c_attn.weight"] = c_attn_weight

        # Same deal with the bias
        Wqkv_bias = state_dict.pop(f"transformer.layers.{d}.mixer.Wqkv.bias")
        q, k, v = torch.split(
            Wqkv_bias, [embed_dim, head_dim * config.n_head, head_dim * config.n_head], dim=0
        )
        c_attn_bias = torch.cat((q, k[:head_dim], v[:head_dim]), dim=0)
        state_dict[f"transformer.h.{d}.attn.c_attn.bias"] = c_attn_bias

    def inv_key_mapping_attn(key):
        key = re.sub(
            r"^transformer.layers.(\d+).mixer.out_proj.weight",
            r"transformer.h.\1.attn.c_proj.weight",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).mixer.out_proj.bias",
            r"transformer.h.\1.attn.c_proj.bias",
            key,
        )
        return key

    state_dict = OrderedDict((inv_key_mapping_attn(k), v) for k, v in state_dict.items())

    return state_dict


def bigcode_config_to_gpt2_config(bigcode_config: GPTBigCodeConfig) -> GPT2Config:
    return GPT2Config(
        activation_function=bigcode_config.activation_function,
        attn_pdrop=bigcode_config.attn_pdrop,
        bos_token_id=bigcode_config.bos_token_id,
        embd_pdrop=bigcode_config.embd_pdrop,
        eos_token_id=bigcode_config.eos_token_id,
        initializer_range=bigcode_config.initializer_range,
        layer_norm_epsilon=bigcode_config.layer_norm_epsilon,
        max_batch_size=bigcode_config.max_batch_size,
        max_sequence_length=bigcode_config.max_sequence_length,
        model_type=bigcode_config.model_type,
        multi_query=bigcode_config.multi_query,
        n_embd=bigcode_config.n_embd,
        n_head=bigcode_config.n_head,
        n_inner=bigcode_config.n_inner,
        n_layer=bigcode_config.n_layer,
        n_positions=bigcode_config.n_positions,
        resid_pdrop=bigcode_config.resid_pdrop,
        scale_attn_weights=bigcode_config.scale_attn_weights,
        summary_activation=bigcode_config.summary_activation,
        summary_first_dropout=bigcode_config.summary_first_dropout,
        summary_proj_to_labels=bigcode_config.summary_proj_to_labels,
        summary_type=bigcode_config.summary_type,
        summary_use_proj=bigcode_config.summary_use_proj,
        use_cache=bigcode_config.use_cache,
        vocab_size=bigcode_config.vocab_size,
    )
