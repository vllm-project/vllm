# Copyright (c) 2023, Tri Dao.

import json
import math
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor
from transformers import GPT2Config, LlamaConfig

from einops import rearrange


def remap_state_dict_meta_llama(
    state_dict: Dict[str, torch.Tensor], config: GPT2Config
) -> Dict[str, torch.Tensor]:
    """Convert the state_dict in Meta format to standard GPT format.

    This function modifies state_dict in place.
    """

    def key_mapping_layers(key):
        return f"transformer.{key}" if not key.startswith("output.") else key

    state_dict = OrderedDict((key_mapping_layers(k), v) for k, v in state_dict.items())

    # Word embedding
    def key_mapping_emb(key):
        return re.sub(
            r"^transformer.tok_embeddings.", "transformer.embeddings.word_embeddings.", key
        )

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]
    else:
        output_embeddings = state_dict.pop("output.weight")
        # Need to recompute vocab_size since LLaMa shards the word embeddings and output embeddings
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
            r"^transformer.layers.(\d+).attention_norm.",
            r"transformer.layers.\1.norm1.",
            key,
        )
        key = re.sub(r"^transformer.layers.(\d+).ffn_norm.", r"transformer.layers.\1.norm2.", key)
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    # MLP
    for l in range(config.n_layer):
        w1 = state_dict.pop(f"transformer.layers.{l}.feed_forward.w1.weight")
        w3 = state_dict.pop(f"transformer.layers.{l}.feed_forward.w3.weight")
        # Our ordering is different
        state_dict[f"transformer.layers.{l}.mlp.fc1.weight"] = torch.cat([w3, w1], dim=0)

    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).feed_forward.w2.",
            r"transformer.layers.\1.mlp.fc2.",
            key,
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f"transformer.layers.{l}.attention.wq.weight")
        Wk = state_dict.pop(f"transformer.layers.{l}.attention.wk.weight")
        Wv = state_dict.pop(f"transformer.layers.{l}.attention.wv.weight")
        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat([Wq, Wk, Wv], dim=0)
        # We don't store these
        state_dict.pop(f"transformer.layers.{l}.attention.inner_attention.rope.freqs", None)

    def key_mapping_attn(key):
        return re.sub(
            r"^transformer.layers.(\d+).attention.wo.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())

    state_dict.pop("transformer.rope.freqs", None)

    return state_dict


def remap_state_dict_hf_llama(
    state_dict: Dict[str, torch.Tensor], config: GPT2Config
) -> Dict[str, torch.Tensor]:
    """Convert the state_dict in Hugging Face format to standard GPT format.

    This function modifies state_dict in place.
    """

    # Embedding
    def key_mapping_emb(key):
        return re.sub(r"^model.embed_tokens.", "transformer.embeddings.word_embeddings.", key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("transformer.embeddings.word_embeddings.weight")
    # It's possible that vocab_size is padded to be a multiple of 8, for example.
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    state_dict["transformer.embeddings.word_embeddings.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )

    # LM head
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["transformer.embeddings.word_embeddings.weight"]
    else:
        output_embeddings = state_dict.pop("lm_head.weight")
        # Need to recompute vocab_size since LLaMa shards the word embeddings and output embeddings
        # differently.
        vocab_size = (
            math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        # It's possible that vocab_size is padded to be a multiple of 8, for example.
        state_dict["lm_head.weight"] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )

    # MLP
    for l in range(config.n_layer):
        # Fusing weights this way based on difference in the following:
        # https://github.com/huggingface/transformers/blob/b42010bb1d3cbf262d27e0a328661885be46dfdb/src/transformers/models/llama/modeling_llama.py#L220
        # https://github.com/Dao-AILab/flash-attention/blob/c60851a8253257eb970e06a022c82517a8033e8c/flash_attn/modules/mlp.py#L115
        w1 = state_dict.pop(f"model.layers.{l}.mlp.gate_proj.weight")
        w3 = state_dict.pop(f"model.layers.{l}.mlp.up_proj.weight")
        state_dict[f"transformer.layers.{l}.mlp.fc1.weight"] = torch.cat([w3, w1], dim=0)

    def key_mapping_mlp(key):
        return re.sub(
            r"^model.layers.(\d+).mlp.down_proj.",
            r"transformer.layers.\1.mlp.fc2.",
            key,
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^model.norm.", r"transformer.ln_f.", key)
        key = re.sub(
            r"^model.layers.(\d+).input_layernorm.",
            r"transformer.layers.\1.norm1.",
            key,
        )
        key = re.sub(
            r"^model.layers.(\d+).post_attention_layernorm.",
            r"transformer.layers.\1.norm2.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def inv_permute(w):
        # Inverse of permute implemented in:
        # https://github.com/huggingface/transformers/blob/b42010bb1d3cbf262d27e0a328661885be46dfdb/src/transformers/models/llama/convert_llama_weights_to_hf.py#L114
        return rearrange(
            w, "(h two d) n -> (h d two) n", d=config.n_embd // config.n_head // 2, two=2
        )

    # Attention
    for l in range(config.n_layer):
        Wq = state_dict.pop(f"model.layers.{l}.self_attn.q_proj.weight")
        Wk = state_dict.pop(f"model.layers.{l}.self_attn.k_proj.weight")
        Wv = state_dict.pop(f"model.layers.{l}.self_attn.v_proj.weight")

        state_dict[f"transformer.layers.{l}.mixer.Wqkv.weight"] = torch.cat(
            [inv_permute(Wq), inv_permute(Wk), Wv], dim=0
        )
        # We don't store these
        state_dict.pop(f"model.layers.{l}.self_attn.rotary_emb.inv_freq", None)

    def key_mapping_attn(key):
        return re.sub(
            r"^model.layers.(\d+).self_attn.o_proj.",
            r"transformer.layers.\1.mixer.out_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    return state_dict


def inv_remap_state_dict_hf_llama(
    state_dict: Dict[str, torch.Tensor], config: GPT2Config
) -> Dict[str, torch.Tensor]:
    """Convert the state_dict in standard GPT format to Hugging Face format.

    This function is meant to be the inverse of remap_state_dict_hf_llama, up to a
    multiplier pad in the embedding and lm_head. That is if the original embedding
    isn't a multiple of pad_vocab_size_multiple, then
    inv_remap_state_dict_hf_llama(remap_state_dict_hf_llama(state_dict)) != state_dict.

    This function modifies state_dict in place.
    """

    # Embedding
    def key_mapping_emb(key):
        return re.sub(r"^transformer.embeddings.word_embeddings.", "model.embed_tokens.", key)

    state_dict = OrderedDict((key_mapping_emb(k), v) for k, v in state_dict.items())
    word_embeddings = state_dict.pop("model.embed_tokens.weight")
    pad_vocab_size_multiple = getattr(config, "pad_vocab_size_multiple", 1)
    vocab_size = (
        math.ceil(word_embeddings.shape[0] / pad_vocab_size_multiple) * pad_vocab_size_multiple
    )
    state_dict["model.embed_tokens.weight"] = F.pad(
        word_embeddings, (0, 0, 0, vocab_size - word_embeddings.shape[0])
    )

    # LM head
    if getattr(config, "tie_word_embeddings"):
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]
    else:
        output_embeddings = state_dict.pop("lm_head.weight")
        vocab_size = (
            math.ceil(output_embeddings.shape[0] / pad_vocab_size_multiple)
            * pad_vocab_size_multiple
        )
        state_dict["lm_head.weight"] = F.pad(
            output_embeddings, (0, 0, 0, vocab_size - output_embeddings.shape[0])
        )

    # MLP
    for l in range(config.n_layer):
        w3, w1 = torch.chunk(
            state_dict.pop(f"transformer.layers.{l}.mlp.fc1.weight"), chunks=2, dim=0
        )
        state_dict[f"model.layers.{l}.mlp.gate_proj.weight"] = w1
        state_dict[f"model.layers.{l}.mlp.up_proj.weight"] = w3

    def key_mapping_mlp(key):
        return re.sub(
            r"^transformer.layers.(\d+).mlp.fc2.",
            r"model.layers.\1.mlp.down_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_mlp(k), v) for k, v in state_dict.items())

    # LayerNorm
    def key_mapping_ln(key):
        key = re.sub(r"^transformer.ln_f.", r"model.norm.", key)
        key = re.sub(
            r"^transformer.layers.(\d+).norm1.",
            r"model.layers.\1.input_layernorm.",
            key,
        )
        key = re.sub(
            r"^transformer.layers.(\d+).norm2.",
            r"model.layers.\1.post_attention_layernorm.",
            key,
        )
        return key

    state_dict = OrderedDict((key_mapping_ln(k), v) for k, v in state_dict.items())

    def permute(w):
        return rearrange(
            w, "(h d two) n -> (h two d) n", d=config.n_embd // config.n_head // 2, two=2
        )

    n_head = config.n_head
    n_head_kv = getattr(config, "n_head_kv", n_head)

    embed_dim = config.hidden_size
    head_dim = embed_dim // n_head

    q_dim = n_head * head_dim
    k_dim = v_dim = n_head_kv * head_dim

    # Attention
    for l in range(config.n_layer):
        Wqkv = state_dict.pop(f"transformer.layers.{l}.mixer.Wqkv.weight")
        Wq = Wqkv[:q_dim]
        Wk = Wqkv[q_dim : q_dim + k_dim]
        Wv = Wqkv[q_dim + k_dim : q_dim + k_dim + v_dim]
        state_dict[f"model.layers.{l}.self_attn.q_proj.weight"] = permute(Wq)
        state_dict[f"model.layers.{l}.self_attn.k_proj.weight"] = permute(Wk)
        state_dict[f"model.layers.{l}.self_attn.v_proj.weight"] = Wv
        state_dict.pop(f"transformer.layers.{l}.attention.inner_attention.rope.freqs", None)

    def key_mapping_attn(key):
        return re.sub(
            r"^transformer.layers.(\d+).mixer.out_proj.",
            r"model.layers.\1.self_attn.o_proj.",
            key,
        )

    state_dict = OrderedDict((key_mapping_attn(k), v) for k, v in state_dict.items())
    return state_dict


def config_from_meta_checkpoint(
    checkpoint_path: Union[str, os.PathLike], model_name: str
) -> LlamaConfig:
    """Load a LlamaConfig from a checkpoint path."""
    with open(Path(checkpoint_path) / model_name / "params.json") as f:
        params = json.load(f)
    config = LlamaConfig(
        hidden_size=params["dim"],
        intermediate_size=None,
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        rms_norm_eps=params["norm_eps"],
        num_key_value_heads=params.get("n_kv_heads", None),
    )
    multiple_of = params.get("multiple_of", 1)
    ffn_dim_multiplier = params.get("ffn_dim_multiplier", None)

    # Compute the hidden dimension of the MLP
    # https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/model.py#L224
    intermediate_size = 4 * config.hidden_size
    # https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/model.py#L195-L199
    intermediate_size = int(2 * intermediate_size / 3)
    # custom dim factor multiplier
    if ffn_dim_multiplier is not None:
        intermediate_size = int(ffn_dim_multiplier * intermediate_size)
    intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)

    config.intermediate_size = intermediate_size
    if "rope_theta" in params:
        config.rotary_emb_base = params["rope_theta"]
    config.vocab_size = 32000
    # some CodeLLaMa have vocab_size 32000, some 32016
    # Sadly it's not specified in the `params.json` file :(
    tokenizer = Path(checkpoint_path) / model_name / "tokenizer.model"
    if tokenizer.is_file():
        config.vocab_size = SentencePieceProcessor(str(tokenizer)).vocab_size()
    return config


def config_from_hf_checkpoint(
    checkpoint_path: Union[str, os.PathLike], model_name: str
) -> LlamaConfig:
    return LlamaConfig.from_pretrained(Path(checkpoint_path) / f"{model_name}-hf" / "config.json")


def config_from_checkpoint(
    checkpoint_path: Union[str, os.PathLike], model_name: str, checkpoint_format="meta"
) -> LlamaConfig:
    if checkpoint_format == "meta":
        return config_from_meta_checkpoint(checkpoint_path, model_name)
    else:
        return config_from_hf_checkpoint(checkpoint_path, model_name)


def state_dicts_from_checkpoint(
    checkpoint_path: Union[str, os.PathLike], model_name: str
) -> List[dict]:
    # Need to sort, otherwise we mess up the ordering and the weights are wrong
    return [
        torch.load(path, map_location="cpu")
        for path in sorted((Path(checkpoint_path) / model_name).glob("consolidated.*.pth"))
    ]


def llama_config_to_gpt2_config(llama_config: LlamaConfig) -> GPT2Config:
    return GPT2Config(
        vocab_size=llama_config.vocab_size,
        n_positions=0,  # No absolute position embedding
        n_embd=llama_config.hidden_size,
        n_layer=llama_config.num_hidden_layers,
        n_head=llama_config.num_attention_heads,
        n_inner=llama_config.intermediate_size,
        activation_function="swiglu",  # Hardcode since HF calls it 'silu'
        # Llama doesn't have dropout, idk if it's because they only release the inference code
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=llama_config.rms_norm_eps,
        initializer_range=llama_config.initializer_range,
        bos_token_id=llama_config.bos_token_id,
        eos_token_id=llama_config.eos_token_id,
        # These are new arguments not in the original GPT2Config
        pad_token_id=llama_config.pad_token_id,  # Idk if this does anything
        rms_norm=True,
        rotary_emb_fraction=1.0,
        rotary_emb_interleaved=True,
        tie_word_embeddings=False,
        qkv_proj_bias=False,
        out_proj_bias=False,
        mlp_fc1_bias=False,
        mlp_fc2_bias=False,
        rotary_emb_base=getattr(llama_config, "rotary_emb_base", 10000.0),
        n_head_kv=llama_config.num_key_value_heads,
    )
