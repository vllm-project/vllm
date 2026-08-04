# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import json

# Third Party
from transformers import AutoConfig


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Fetch model configuration using AutoConfig."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="The name of the model to fetch configuration for.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Load model configuration using AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model)

        # Prepare configuration data in a dictionary format
        config_data = {
            "hidden_size": getattr(config, "hidden_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
        }

        # DeepSeek MLA models (V3, V3.1, V3.2, … and R1) store
        # KV in latent space
        if (
            args.model.lower().startswith("deepseek-ai/deepseek-v3")
            or args.model == "deepseek-ai/DeepSeek-R1"
        ):
            config_data["kv_lora_rank"] = getattr(config, "kv_lora_rank", None)
            config_data["qk_rope_head_dim"] = getattr(config, "qk_rope_head_dim", None)

        # Models whose head_dim is explicit in config and may
        # differ from hidden_size / num_heads:
        # Qwen3, GLM4, and Hunyuan dense variants.
        if (
            "qwen/qwen3-" in args.model.lower()
            or "zai-org/glm-4." in args.model.lower()
            or (
                args.model.lower().startswith("tencent/hunyuan-")
                and args.model.lower() != "tencent/hunyuan-large"
            )
        ):
            config_data["head_dim"] = getattr(config, "head_dim", None)

        # Hunyuan-Large uses CLA (Cross-Layer Attention):
        # KV layers = num_hidden_layers / cla_share_factor
        if args.model.lower() == "tencent/hunyuan-large":
            config_data["cla_share_factor"] = getattr(config, "cla_share_factor", None)

        # Convert to JSON and print
        string = json.dumps(config_data, indent=4)

        print("\033[32m" + "Model configuration for " + args.model + ":\n" + "\033[0m")

        print(f'"{args.model}": {string}\n')

        print(
            "\033[32mPlease copy the above JSON to the 'modelconfig.json'"
            "and create a new PR\033[0m"
        )

    except Exception as e:
        # Print error message in JSON format
        error_data = {"error": str(e)}
        print(json.dumps(error_data, indent=4))


if __name__ == "__main__":
    main()
