from safetensors import safe_open
from safetensors.torch import save_file
import torch

SMOOTH_STRENGHT = 0.5

activation_scales = torch.load("/home/ray/default/mixtral_scales.pth")
smoothquant_scales = {}
for layer_idx in range(32):
    key_prefix = f"model.layers.{layer_idx}.block_sparse_moe.experts"
    target_prefix = f"model.layers.{layer_idx}.block_sparse_moe.scales"
    for weight_name in ["w1", "w2", "w3"]:
        tensors = [activation_scales[key_prefix + f".{expert_idx}.{weight_name}"] for expert_idx in range(8)]
        smoothquant_scales[target_prefix + f".{weight_name}"] = torch.mean(torch.stack(tensors), dim=0)**SMOOTH_STRENGHT

def rewrite_safetensors(name):
    tensors = {}
    with safe_open(name, framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
            if "w1" in k or "w2" in k or "w3" in k:
                name_parts = k.split(".")
                scale_name = "model.layers." + name_parts[2] + ".block_sparse_moe.scales." + name_parts[-2]
                print(f"scaling {k} with {scale_name}")
                tensors[scale_name] = smoothquant_scales[scale_name]
                tensors[k] *= smoothquant_scales[scale_name]
                # Convert tensor to fp8
                tensors[k] = tensors[k].to(torch.float8_e4m3fn)
    save_file(tensors, name)

for i in range(1, 20):
    filename = f"model-{i:05}-of-00019.safetensors"
    print(f"rewriting {filename}")
    rewrite_safetensors(filename)