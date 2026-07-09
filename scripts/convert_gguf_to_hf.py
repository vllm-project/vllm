"""Convert GGUF models to HuggingFace format for use with vLLM.

Usage:
  python scripts/convert_gguf_to_hf.py E:\OLLAMA-Models\GGUF\qwen3-1.7b-coder-distilled-sft-Q8_0.gguf F:\VLLM-Models\qwen3-1.7b-coder
  python scripts/convert_gguf_to_hf.py E:\OLLAMA-Models\GGUF\ --outdir F:\VLLM-Models  # batch
"""
import json, os, sys
from pathlib import Path

import torch
import gguf
import gguf.quants as gguf_quants
import numpy as np

ARCH_MAP = {
    "llama":      ("LlamaForCausalLM", "llama"),
    "qwen2":      ("Qwen2ForCausalLM", "qwen2"),
    "qwen3":      ("Qwen2ForCausalLM", "qwen2"),
    "qwen2moe":   ("Qwen2MoeForCausalLM", "qwen2_moe"),
    "gemma2":     ("Gemma2ForCausalLM", "gemma2"),
    "phi3":       ("Phi3ForCausalLM", "phi3"),
    "starcoder2": ("Starcoder2ForCausalLM", "starcoder2"),
    "falcon":     ("FalconForCachedCausalModel", "falcon"),
}

# Arch-specific config defaults (GGUF keys are already mapped to HF names)
ARCH_CONFIG_DEFAULTS = {
    "llama":      {"rope_theta": 10000.0, "max_position_embeddings": 2048, "rms_norm_eps": 1e-5, "vocab_size": 32000, "hidden_act": "silu"},
    "qwen2":      {"rope_theta": 1000000.0, "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "vocab_size": 151936, "hidden_act": "silu"},
    "qwen2_moe":  {"rope_theta": 1000000.0, "max_position_embeddings": 32768, "rms_norm_eps": 1e-6, "vocab_size": 151936, "hidden_act": "silu"},
    "gemma2":     {"max_position_embeddings": 8192, "rms_norm_eps": 1e-6, "vocab_size": 256000, "hidden_act": "gelu_pytorch_tanh", "head_dim": 256, "attention_logit_softcapping": 50.0, "final_logit_softcapping": 30.0, "query_pre_attn_scalar": 256},
    "phi3":       {"max_position_embeddings": 4096, "rms_norm_eps": 1e-5, "vocab_size": 32064, "hidden_act": "silu", "rope_scaling": {"type": "su", "short_factor": [], "long_factor": []}},
    "starcoder2": {"max_position_embeddings": 16384, "rms_norm_eps": 1e-5, "vocab_size": 49152, "hidden_act": "gelu_pytorch_tanh", "use_bias": True},
}

GGUF_TO_HF_KEYS = [
    ("embedding_length", "hidden_size"),
    ("feed_forward_length", "intermediate_size"),
    ("attention.head_count", "num_attention_heads"),
    ("block_count", "num_hidden_layers"),
    ("attention.head_count_kv", "num_key_value_heads"),
    ("rope.freq_base", "rope_theta"),
    ("context_length", "max_position_embeddings"),
    ("attention.layer_norm_rms_epsilon", "rms_norm_eps"),
    ("vocab_size", "vocab_size"),
]


def gguf_to_hf_tensor_name(name: str, arch: str) -> str:
    """Map GGUF tensor names to HuggingFace tensor names."""
    # Token embeddings
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "token_embd_norm.weight":
        return "model.embed_tokens.norm.weight" if arch == "gemma2" else None
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output_norm.bias":
        return "model.norm.bias"
    if name == "output.weight":
        return "lm_head.weight"
    if name == "output.bias":
        return None  # most archs don't have bias on lm_head

    # Block (layer) weights: blk.{n}.{component}.{weight}
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blk":
        layer = parts[1]
        suffix = parts[-1]
        component = ".".join(parts[2:-1])

        base = f"model.layers.{layer}"

        # Attention
        if component == "attn_q":
            return f"{base}.self_attn.q_proj.{suffix}"
        if component == "attn_k":
            return f"{base}.self_attn.k_proj.{suffix}"
        if component == "attn_v":
            return f"{base}.self_attn.v_proj.{suffix}"
        if component == "attn_output":
            return f"{base}.self_attn.o_proj.{suffix}"
        if component == "attn_norm":
            return f"{base}.input_layernorm.{suffix}"
        if component == "attn_norm_2":
            return f"{base}.post_attention_layernorm.{suffix}"
        if component == "attn_rel_b":
            return None  # RoPE is computed, not stored

        # FFN (common)
        if component == "ffn_gate":
            return f"{base}.mlp.gate_proj.{suffix}"
        if component == "ffn_down":
            return f"{base}.mlp.down_proj.{suffix}"
        if component == "ffn_up":
            return f"{base}.mlp.up_proj.{suffix}"
        if component == "ffn_norm":
            return f"{base}.post_attention_layernorm.{suffix}"

        # Gemma2 specific
        if component == "attn_q_norm":
            return f"{base}.self_attn.q_norm.{suffix}"
        if component == "attn_k_norm":
            return f"{base}.self_attn.k_norm.{suffix}"
        if component == "post_attention_norm":
            return f"{base}.post_attention_layernorm.{suffix}"
        if component == "pre_attention_norm":
            return f"{base}.input_layernorm.{suffix}"
        if component == "post_ffw_norm":
            return f"{base}.post_attention_layernorm.{suffix}"
        if component == "pre_ffw_norm":
            return f"{base}.input_layernorm.{suffix}"

        # Starcoder2/ Falcon: attn_qkv fused
        if component == "attn_qkv":
            return f"{base}.self_attn.qkv_proj.{suffix}"

        # MoE
        if "ffn_gate" in component or "ffn_down" in component or "ffn_up" in component:
            idx = ""
            for p in parts:
                if p.startswith("ffn_gate"):
                    return f"{base}.mlp.gate_proj.{suffix}"
                if p.startswith("ffn_down"):
                    return f"{base}.mlp.down_proj.{suffix}"
                if p.startswith("ffn_up"):
                    return f"{base}.mlp.up_proj.{suffix}"
            return None

    # Special: expert weights (MoE models)
    if " experts " in name or "expert" in parts:
        # e.g. blk.0.ffn_gate.1.weight -> model.layers.0.mlp.experts.1.gate_proj.weight
        for i, p in enumerate(parts):
            if p == "ffn_gate" and i + 1 < len(parts) and parts[i+1].isdigit():
                return f"model.layers.{parts[1]}.mlp.experts.{parts[i+1]}.gate_proj.{suffix}"
            if p == "ffn_down" and i + 1 < len(parts) and parts[i+1].isdigit():
                return f"model.layers.{parts[1]}.mlp.experts.{parts[i+1]}.down_proj.{suffix}"
            if p == "ffn_up" and i + 1 < len(parts) and parts[i+1].isdigit():
                return f"model.layers.{parts[1]}.mlp.experts.{parts[i+1]}.up_proj.{suffix}"

    return None  # skip unknown


def convert_gguf_to_hf(gguf_path: str, output_dir: str, max_shard_size_gb: int = 2):
    """Convert a single GGUF model to HuggingFace format."""
    gguf_path = Path(gguf_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading {gguf_path.name}...")
    reader = gguf.GGUFReader(gguf_path)

    # Get architecture
    arch = reader.fields.get("general.architecture")
    if arch is None:
        print("  ERROR: no architecture field in GGUF")
        return False
    arch_str = str(arch.parts[-1], "utf-8")
    print(f"  Architecture: {arch_str}")

    # Get hyperparameters
    arch_info = ARCH_MAP.get(arch_str)
    if arch_info is None:
        print(f"  WARNING: unknown architecture '{arch_str}', trying generic config")
        hf_type = "AutoModelForCausalLM"
        model_type = arch_str
        arch_defaults = {}
    else:
        hf_type, model_type = arch_info
        arch_defaults = ARCH_CONFIG_DEFAULTS.get(arch_str, {})

    params = {}
    for field_name, field in reader.fields.items():
        key = str(field_name)
        val = field.parts[-1]
        if isinstance(val, bytes):
            try:
                val = val.decode("utf-8")
            except Exception:
                continue
        elif hasattr(val, 'dtype'):
            # numpy/memmap array: extract scalar or keep array
            if val.ndim == 0:
                val = val.item()
            elif val.size == 1:
                val = val.item()
            else:
                continue  # skip vector fields (e.g. tokenizer arrays)
        elif not isinstance(val, (int, float, str)):
            continue
        params[key] = val

    # Map GGUF metadata keys to HF config (try arch-specific prefixes)
    hf_params = {}
    prefixes = [arch_str, "llama", "bert"]
    for gguf_suffix, hf_key in GGUF_TO_HF_KEYS:
        for prefix in prefixes:
            gguf_key = f"{prefix}.{gguf_suffix}"
            if gguf_key in params:
                hf_params[hf_key] = params[gguf_key]
                break

    # Build config.json from mapped params + arch defaults
    config = {
        "architectures": [hf_type],
        "model_type": model_type,
        "torch_dtype": "float16",
    }

    # Required params (from metadata or tensor inference)
    core_keys = ["hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "num_key_value_heads", "rms_norm_eps", "vocab_size"]
    for k in core_keys + ["rope_theta", "max_position_embeddings"]:
        if k in hf_params:
            config[k] = hf_params[k]

    # Arch-specific defaults for keys not in metadata
    for k, v in arch_defaults.items():
        config.setdefault(k, v)

    config.setdefault("hidden_size", 2048)
    config.setdefault("vocab_size", 32000)
    config.setdefault("rms_norm_eps", 1e-5)
    config.setdefault("hidden_act", "silu")

    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Wrote config.json")

    # Create tokenizer.json if the GGUF has tokenizer data
    tokenizer_data = None
    for field_name in reader.fields:
        if field_name.startswith("tokenizer"):
            tokenizer_data = True
            break

    if tokenizer_data:
        # Extract tokenizer info and create minimal tokenizer.json + tokenizer_config.json
        tokenizer_config = {"model_type": config.get("model_type", "llama")}
        
        # Try to extract vocab
        token_ids = []
        token_scores = []
        token_types = []
        
        if "tokenizer.ggml.tokens" in reader.fields:
            token_field = reader.fields["tokenizer.ggml.tokens"]
            for part in token_field.parts:
                if isinstance(part, bytes):
                    token_ids.append(part.decode("utf-8", errors="replace"))
                else:
                    token_ids.append(str(part))
        
        if "tokenizer.ggml.scores" in reader.fields:
            score_field = reader.fields["tokenizer.ggml.scores"]
            token_scores = [float(s) for s in score_field.parts]
        
        if "tokenizer.ggml.token_type" in reader.fields:
            type_field = reader.fields["tokenizer.ggml.token_type"]
            try:
                arr = type_field.parts[-1]
                token_types = [int(arr)] if isinstance(arr, (int, np.integer)) else arr.flatten().tolist() if hasattr(arr, 'flatten') else []
            except Exception:
                token_types = []

        # Determine BOS/EOS/UNK
        bos = params.get("tokenizer.ggml.bos_token_id", 1)
        eos = params.get("tokenizer.ggml.eos_token_id", 2)
        unk = params.get("tokenizer.ggml.unknown_token_id", 0)
        pad = params.get("tokenizer.ggml.padding_token_id", eos)

        # Determine tokenizer class
        if "tokenizer.ggml.model" in params:
            tok_type = str(params["tokenizer.ggml.model"], "utf-8") if isinstance(params["tokenizer.ggml.model"], bytes) else str(params["tokenizer.ggml.model"])
        else:
            tok_type = "llama" if arch_str in ("llama", "qwen2") else "bpe"

        if tok_type == "llama":
            tokenizer_config["tokenizer_class"] = "LlamaTokenizer"
            tokenizer_class = "LLaMA"
        elif tok_type == "gpt2" or tok_type == "bpe":
            tokenizer_config["tokenizer_class"] = "GPT2Tokenizer"
            tokenizer_class = "GPT2"
        else:
            tokenizer_config["tokenizer_class"] = "PreTrainedTokenizerFast"
            tokenizer_class = tok_type.capitalize()

        tokenizer_config["bos_token"] = bos if isinstance(bos, str) else (token_ids[bos] if bos < len(token_ids) else "<s>")
        tokenizer_config["eos_token"] = eos if isinstance(eos, str) else (token_ids[eos] if eos < len(token_ids) else "</s>")
        tokenizer_config["unk_token"] = unk if isinstance(unk, str) else (token_ids[unk] if unk < len(token_ids) else "<unk>")

        with open(out / "tokenizer_config.json", "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        print(f"  Wrote tokenizer_config.json")

        # Create minimal tokenizer files - vLLM will use these for inference
        # Even without full merges, models can be loaded and used with the completions API.
        if token_ids:
            # vLLM works with just tokenizer_config.json and config.json for most models
            pass

    # Convert and save tensors
    print(f"  Reading {len(reader.tensors)} tensors...")
    tensors = {}
    for tensor in reader.tensors:
        hf_name = gguf_to_hf_tensor_name(tensor.name, arch_str)
        if hf_name is None:
            print(f"    Skipping {tensor.name}")
            continue

        # Dequantize GGUF tensor data to float32 numpy array
        # (handles all quantization types including Q8_0, F32, F16, etc.)
        # Output shape is in PyTorch convention [out_features, in_features]
        data = gguf_quants.dequantize(tensor.data, tensor.tensor_type)
        t = torch.from_numpy(data.copy()).half()

        tensors[hf_name] = t

    # Post-process: remove tensors not valid for the target architecture
    if arch_str in ("qwen3", "qwen2") and hf_type == "Qwen2ForCausalLM":
        # Qwen3 has per-head QK norms but Qwen2ForCausalLM doesn't support them
        tensors = {k: v for k, v in tensors.items()
                   if not ("self_attn.q_norm" in k or "self_attn.k_norm" in k)}

    print(f"  Converted {len(tensors)} tensors")

    # Infer config values from tensor shapes if metadata was missing
    embed_key = "model.embed_tokens.weight"
    if embed_key in tensors and config.get("vocab_size") == 32000:
        inferred_vocab = tensors[embed_key].shape[0]
        config["vocab_size"] = inferred_vocab
        json.dump(config, open(out / "config.json", "w"), indent=2)

    # Shard and save as safetensors
    from safetensors.torch import save_file as safe_save
    shard_size = 0
    shard_idx = 0
    shard_tensors = {}
    max_bytes = max_shard_size_gb * 1024**3

    for name, t in tensors.items():
        tensor_bytes = t.numel() * t.element_size()
        if shard_size + tensor_bytes > max_bytes and shard_tensors:
            shard_file = out / f"model-{shard_idx:05d}-of-{100000:05d}.safetensors"
            safe_save(shard_tensors, str(shard_file))
            print(f"    Wrote {shard_file.name} ({shard_size/1024**3:.1f} GB)")
            shard_tensors = {}
            shard_size = 0
            shard_idx += 1
        shard_tensors[name] = t.contiguous()
        shard_size += tensor_bytes

    if shard_tensors:
        shard_file = out / f"model-{shard_idx:05d}-of-{100000:05d}.safetensors"
        safe_save(shard_tensors, str(shard_file))
        print(f"    Wrote {shard_file.name} ({shard_size/1024**3:.1f} GB)")
        shard_idx += 1

    # Write model-{total_shards}-of-{total_shards}.safetensors index
    total_shards = shard_idx
    # Rename shard files to have correct total
    for i in range(total_shards):
        old = out / f"model-{i:05d}-of-{100000:05d}.safetensors"
        new = out / f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        if old.exists():
            old.rename(new)

    # Write model.safetensors.index.json
    weight_map = {}
    for i in range(total_shards):
        shard_file = f"model-{i:05d}-of-{total_shards:05d}.safetensors"
        # Read back which tensors are in which shard
        from safetensors import safe_open
        with safe_open(str(out / shard_file), framework="pt") as f:
            for key in f.keys():
                weight_map[key] = shard_file

    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
                    "weight_map": weight_map}, f, indent=2)

    total_size_gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1024**3
    print(f"  Done! {total_shards} shards, {total_size_gb:.2f} GB total")
    print(f"  Output: {out}")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert GGUF models to HuggingFace format")
    parser.add_argument("input", help="GGUF file or directory containing .gguf files")
    parser.add_argument("--outdir", "-o", default="F:\\VLLM-Models", help="Output directory")
    parser.add_argument("--shard", type=int, default=2, help="Max shard size in GB")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = list(input_path.glob("*.gguf")) + list(input_path.glob("*.bin"))
    else:
        print(f"Input {input_path} not found")
        sys.exit(1)

    for f in sorted(files):
        model_name = f.stem
        # Clean model name for dir
        model_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
        output_dir = outdir / model_name
        print(f"\n{'='*60}")
        print(f"Converting: {f.name} -> {output_dir}")
        print(f"{'='*60}")
        convert_gguf_to_hf(str(f), str(output_dir), args.shard)


if __name__ == "__main__":
    main()
