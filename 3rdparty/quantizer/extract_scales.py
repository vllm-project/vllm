import argparse
import json
import os
from vllm.model_executor.weight_utils import (
    hf_model_weights_iterator,
    prepare_hf_model_weights
)

default_output_name = "kv_cache_scales.json"


def main(args):
    layer_scale_factors_map = {}
    if args.output is None:
        hf_folder, _, _ = prepare_hf_model_weights(args.model,
                                                   args.cache_dir,
                                                   args.load_format,
                                                   revision=args.revision,
                                                   fall_back_to_pt=False)
        output_file = os.path.join(hf_folder, default_output_name)
    else:
        output_file = os.path.join(args.output, default_output_name)
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)

    for name, param in hf_model_weights_iterator(args.model,
                                                 args.cache_dir,
                                                 args.load_format,
                                                 args.revision,
                                                 fall_back_to_pt=False):
        if "kv_cache_scaling_factor" in name:
            nums = [int(s) for s in name.split('.') if s.isdigit()]
            assert len(nums) == 1, f"Could not determine layer idx for {name}!"
            layer_idx = nums[0]
            assert layer_idx not in layer_scale_factors_map, f"Duplicate scaling " \
                f"factor corresponding to layer {layer_idx}!"
            try:
                layer_scale_factors_map[layer_idx] = param.item()
            except RuntimeError:
                print("This utility supports only per-tensor scalar scale factors "
                        f"for now. The tensor\n {name} = {param} is an invalid "
                        "scale factor!")
                raise
    if len(layer_scale_factors_map) == 0:
        print("WARNING: No KV cache scale factors found! No output saved.")
    else:
        with open(output_file, 'w') as f:
            json.dump(layer_scale_factors_map, f, sort_keys=True)
            print(f"Completed! KV cache scaling factors saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This simple utility extracts the "
                                     "KV cache scaling factors from a quantized HF model "
                                     "and saves them to a JSON file compatible with later "
                                     "use by vLLM (pass this file to the appropriate "
                                     "runtime typically using the argument "
                                     "--kv-cache-scales-path <filename>). This is only used "
                                     "if the KV cache dtype is FP8 and on ROCm (AMD GPU).")
    parser.add_argument("--model",
                        help="Specify either a directory or name of a HF model. If the model "
                        "does not exist, this utility will attempt to download said model "
                        "from the HF repo.",
                        required=True)
    parser.add_argument("--cache_dir",
                        help="Optionally specify a cache directory to use for a HF model "
                        "download.",
                        default=None)
    parser.add_argument("--load_format",
                        help="Optionally specify the format of the model's tensor files "
                        "containing the KV cache scaling factors.",
                        choices=["auto", "safetensors", "npcache"],
                        default="auto")
    parser.add_argument("--revision",
                        help="Optionally specify the model's revision number.",
                        default=None)
    parser.add_argument("--output",
                        help="Specify the output directory. By default it will be saved in "
                        f"the model directory with the filename {default_output_name}, "
                        "however you can override this behavior here.",
                        default=None)
    args = parser.parse_args()

    main(args)
