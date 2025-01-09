import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file

# Script to convert an EAGLE checkpoint available at
# https://huggingface.co/yuhuili into vLLM compatible checkpoint.
# Borrowed from
# https://gist.github.com/abhigoyal1997/1e7a4109ccb7704fbc67f625e86b2d6d

# Example Usage

# python3 vllm/spec_decode/scripts/vLLM_compatible_EAGLE_ckpt.py
#     --eagle_dir </path/to/eagle/checkpoint/dir>
#     --baseline_ckpt_file_lm_head \
#         </path/to/base_model/ckpt_file_with_lm_head_weight>


def update_model(eagle_dir, baseline_ckpt_path):
    # Load the Eagle model checkpoint
    eagle_ckpt_path = os.path.join(eagle_dir, "pytorch_model.bin")
    ckpt = torch.load(eagle_ckpt_path)

    # Load the baseline model checkpoint
    ref_ckpt = load_file(baseline_ckpt_path)

    # Update the EAGLE model with the lm_head.weight from the reference model
    ckpt['lm_head.weight'] = ref_ckpt['lm_head.weight']

    # Save the modified checkpoint as safetensors
    save_file(ckpt, os.path.join(eagle_dir, "model.safetensors"))

    # Load and modify the configuration
    config_path = os.path.join(eagle_dir, "config.json")
    with open(config_path) as rf:
        cfg = json.load(rf)

    if "fc.bias" in ckpt:
        cfg["eagle_fc_bias"] = True

    cfg = {"model_type": "eagle", "model": cfg}

    # Save the new configuration
    with open(config_path, "w") as wf:
        json.dump(cfg, wf)

    # Delete the original pytorch model checkpoint file
    os.remove(eagle_ckpt_path)

    print(f"Model updated and saved to {eagle_dir}/model.safetensors. "
          "Use this new checkpoint with vLLM")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Make EAGLE checkpoints available at "
        "https://huggingface.co/yuhuili vLLM compatible.")
    parser.add_argument('--eagle_dir',
                        type=str,
                        help="Directory for the EAGLE checkpoint")
    parser.add_argument(
        '--baseline_ckpt_file_lm_head',
        type=str,
        help="Path to the baseline model checkpoint file containing the "
        "weights for lm_head. The checkpoint needs to be in safetensor "
        "format.")

    args = parser.parse_args()

    # Update the model
    update_model(args.eagle_dir, args.baseline_ckpt_file_lm_head)


if __name__ == "__main__":
    main()
