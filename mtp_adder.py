import glob
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

NANO_MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/smor/models/nemotron-nano-9B-v2"
ERNIE_MODEL = "/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/smor/models/ernie-4.5-21B"

def mtp_adder():
    last_safetensors_file_path = sorted(glob.glob(NANO_MODEL + "/*.safetensors"))[-1]
    last_safetensors_filename = last_safetensors_file_path.split("/")[-1]
    index_file_path = NANO_MODEL + "/model.safetensors.index.json"
    
    tensors = {}
    with safe_open(last_safetensors_file_path, framework="pt") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    # Load config to get model dimensions
    config = AutoConfig.from_pretrained(NANO_MODEL, trust_remote_code=True)
    hidden_size = config.hidden_size  # 4480
    num_attention_heads = config.num_attention_heads  # 40
    num_key_value_heads = config.num_key_value_heads  # 8
    head_dim = 128  # hidden_size // num_attention_heads
    layer_dtype = torch.bfloat16
    
    # MTP layer index = last layer + 1
    mtp_layer_idx = 56
    
    # Calculate projection sizes (based on NemotronHAttention)
    # For TP compatibility, we use total heads here
    q_size = num_attention_heads * head_dim  # 40 * 128 = 5120
    kv_size = num_key_value_heads * head_dim  # 8 * 128 = 1024
    
    print(f"Adding MTP layer {mtp_layer_idx} weights:")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_attention_heads: {num_attention_heads}")
    print(f"  num_key_value_heads: {num_key_value_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  q_size: {q_size}")
    print(f"  kv_size: {kv_size}")
    
    # Track new weights for index update
    new_weights = []
    
    # 1. RMSNorm weights for embedding normalization (shape: [hidden_size])
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_emb_norm.weight"
    tensors[weight_name] = torch.ones(hidden_size, dtype=layer_dtype)
    new_weights.append(weight_name)
    
    # 2. RMSNorm weights for hidden state normalization (shape: [hidden_size])
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_hidden_norm.weight"
    tensors[weight_name] = torch.ones(hidden_size, dtype=layer_dtype)
    new_weights.append(weight_name)
    
    # 3. Linear projection for fusion (shape: [hidden_size, hidden_size * 2])
    # Initialize as identity-like projection (average of concatenated inputs)
    fusion_weight = torch.cat([
        torch.eye(hidden_size, dtype=layer_dtype) * 0.5,  # Weight for embeddings
        torch.eye(hidden_size, dtype=layer_dtype) * 0.5,  # Weight for hidden states
    ], dim=1)
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_linear_proj.weight"
    tensors[weight_name] = fusion_weight
    new_weights.append(weight_name)
    
    # 4. Attention QKV projection weights - SEPARATE q_proj, k_proj, v_proj
    # These will be stacked together by the weight loader using shard_id
    
    # Q projection (shape: [q_size, hidden_size])
    std = (2.0 / (hidden_size + q_size)) ** 0.5
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_block.mixer.q_proj.weight"
    tensors[weight_name] = torch.randn(q_size, hidden_size, dtype=layer_dtype) * std
    new_weights.append(weight_name)
    
    # K projection (shape: [kv_size, hidden_size])
    std = (2.0 / (hidden_size + kv_size)) ** 0.5
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_block.mixer.k_proj.weight"
    tensors[weight_name] = torch.randn(kv_size, hidden_size, dtype=layer_dtype) * std
    new_weights.append(weight_name)
    
    # V projection (shape: [kv_size, hidden_size])
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_block.mixer.v_proj.weight"
    tensors[weight_name] = torch.randn(kv_size, hidden_size, dtype=layer_dtype) * std
    new_weights.append(weight_name)
    
    # 5. O projection weights (shape: [hidden_size, q_size])
    std = (2.0 / (q_size + hidden_size)) ** 0.5
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_block.mixer.o_proj.weight"
    tensors[weight_name] = torch.randn(hidden_size, q_size, dtype=layer_dtype) * std
    new_weights.append(weight_name)
    
    # 6. MTP block normalization (shape: [hidden_size])
    weight_name = f"backbone.layers.{mtp_layer_idx}.mtp_block.norm.weight"
    tensors[weight_name] = torch.ones(hidden_size, dtype=layer_dtype)
    new_weights.append(weight_name)
    
    # 7. Final normalization for NemotronHMultiTokenPredictor (shape: [hidden_size])
    # This is the norm in the MTP predictor (backbone.norm.weight, not backbone.norm_f.weight)
    weight_name = "backbone.norm.weight"
    tensors[weight_name] = torch.ones(hidden_size, dtype=layer_dtype)
    new_weights.append(weight_name)
    
    # 8. Embeddings should be shared - copy from existing backbone.embeddings.weight
    # The model loader should handle the mapping from backbone.embeddings -> model.embed_tokens
    # So we don't need to add it here, it already exists
    
    # Remove old weights if they exist
    old_keys = [
        f"backbone.layers.{mtp_layer_idx}.identity_transform.weight",
        f"backbone.layers.{mtp_layer_idx}.mtp_block.mixer.qkv_proj.weight",  # Old combined weight
    ]
    for old_key in old_keys:
        if old_key in tensors:
            print(f"Removing old weight: {old_key}")
            del tensors[old_key]
    
    # Save updated checkpoint
    print(f"Saving {len(new_weights)} new weights to {last_safetensors_file_path}")
    save_file(tensors, last_safetensors_file_path)
    
    # Update safetensors index file
    print(f"Updating index file: {index_file_path}")
    with open(index_file_path, 'r') as f:
        index_data = json.load(f)
    
    # Add new weights to the weight_map
    for weight_name in new_weights:
        index_data["weight_map"][weight_name] = last_safetensors_filename
    
    # Update total size (approximate - add size of new tensors)
    new_size = sum(tensors[w].numel() * tensors[w].element_size() for w in new_weights)
    index_data["metadata"]["total_size"] += new_size
    
    # Save updated index
    with open(index_file_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"Successfully added {len(new_weights)} MTP weights to layer {mtp_layer_idx}")
    print(f"New weights: {new_weights}")

def main():
    mtp_adder()
    
if __name__ == "__main__":
    main()
