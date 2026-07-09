"""Fix config.json for a converted GGUF model by inferring params from tensor shapes."""
import json, sys
from pathlib import Path
import torch

def fix_config(model_dir):
    model_dir = Path(model_dir)
    config_file = model_dir / "config.json"
    if not config_file.exists():
        print(f"No config.json in {model_dir}")
        return False

    config = json.loads(config_file.read_text())

    # Check if critical params are missing
    need_fix = any(config.get(k) is None for k in
                   ["hidden_size", "intermediate_size", "num_attention_heads",
                    "num_hidden_layers", "num_key_value_heads"])

    if not need_fix:
        print("Config looks complete")
        return True

    # Infer from safetensors
    safetensors = list(model_dir.glob("*.safetensors"))
    if not safetensors:
        print("No safetensors found")
        return False

    from safetensors import safe_open
    shapes = {}
    for st in safetensors:
        with safe_open(str(st), framework="pt") as f:
            for k in f.keys():
                shapes[k] = f.get_tensor(k).shape

    # Infer model params from tensor shapes
    # embed_tokens -> hidden_size
    embed = shapes.get("model.embed_tokens.weight")
    if embed and config.get("hidden_size") is None:
        config["hidden_size"] = embed[1]
        config["vocab_size"] = embed[0]
        print(f"  hidden_size = {embed[1]}, vocab_size = {embed[0]}")

    # Count layers
    layer_keys = [k for k in shapes if k.startswith("model.layers.")]
    layer_nums = set()
    for k in layer_keys:
        parts = k.split(".")
        for p in parts:
            if p.isdigit():
                layer_nums.add(int(p))
    if layer_nums and config.get("num_hidden_layers") is None:
        config["num_hidden_layers"] = max(layer_nums) + 1
        print(f"  num_hidden_layers = {config['num_hidden_layers']}")

    # Attention heads from q_proj
    q_proj = shapes.get("model.layers.0.self_attn.q_proj.weight")
    if q_proj and config.get("num_attention_heads") is None:
        # q_proj shape: [hidden_size * num_heads, hidden_size] or [num_heads * head_dim, hidden_size]
        hs = config.get("hidden_size", q_proj[1])
        for hd in [256, 128, 64, 32]:
            if q_proj[0] * hs % (hd * hs) == 0:
                nh = q_proj[0] // hd
                if nh > 0:
                    config["num_attention_heads"] = nh
                    config.setdefault("head_dim", hd)
                    print(f"  num_attention_heads = {nh}, head_dim = {hd}")
                    break

    k_proj = shapes.get("model.layers.0.self_attn.k_proj.weight")
    if k_proj and config.get("num_key_value_heads") is None:
        hs = config.get("hidden_size", k_proj[1])
        nh = config.get("num_attention_heads", 1)
        hd = config.get("head_dim", k_proj[0] // nh if nh else 128)
        nkv = k_proj[0] // hd
        if nkv > 0:
            config["num_key_value_heads"] = nkv
            print(f"  num_key_value_heads = {nkv}")

    # Intermediate size from gate_proj
    gate = shapes.get("model.layers.0.mlp.gate_proj.weight")
    if gate and config.get("intermediate_size") is None:
        config["intermediate_size"] = gate[0]
        print(f"  intermediate_size = {gate[0]}")

    # Set defaults for remaining nulls
    config.setdefault("torch_dtype", "float16")
    config.setdefault("rms_norm_eps", 1e-6)
    config.setdefault("max_position_embeddings", 32768)
    config.setdefault("hidden_act", "silu")

    config_file.write_text(json.dumps(config, indent=2))
    print(f"  Updated {config_file}")
    return True


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    fix_config(model_dir)
