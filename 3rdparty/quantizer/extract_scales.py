import argparse
import fnmatch
import glob
from huggingface_hub import snapshot_download, HfFileSystem
import json
import numpy as np
import os
from safetensors.torch import safe_open
import torch
from typing import List, Optional, Tuple


# Adapted from vllm/model_executor/weight_utils.py
# The main differences are that we add the NPZ format and that there's no
# need for a file lock when downloading model weights because this tool is
# not intended to be run on multiple processes simultaneously.
# Since our use case is sufficiently different, we define our own function
# here.
def _prepare_hf_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
    revision: Optional[str] = None,
) -> Tuple[str, List[str], bool]:
    # Download model weights from huggingface.
    is_local = os.path.isdir(model_name_or_path)
    use_safetensors = False
    # Some quantized models use .pt files for storing the weights.
    if load_format == "auto":
        allow_patterns = ["*.safetensors", "*.bin"]
    elif load_format == "safetensors":
        use_safetensors = True
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    elif load_format == "npz":
        allow_patterns = ["*.npz"]
    else:
        raise ValueError(f"Unknown load_format: {load_format}")

    if fall_back_to_pt:
        allow_patterns += ["*.pt"]

    if not is_local:
        # Before we download we look at that is available:
        fs = HfFileSystem()
        file_list = fs.ls(model_name_or_path, detail=False, revision=revision)
        # depending on what is available we download different things
        for pattern in allow_patterns:
            matching = fnmatch.filter(file_list, pattern)
            if len(matching) > 0:
                allow_patterns = [pattern]
                break
        print(f"Downloading model... Using model weights format {allow_patterns}")
        hf_folder = snapshot_download(model_name_or_path,
                                      allow_patterns=allow_patterns,
                                      cache_dir=cache_dir,
                                      revision=revision)
    else:
        hf_folder = model_name_or_path
    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
        if len(hf_weights_files) > 0:
            if pattern == "*.safetensors":
                use_safetensors = True
            break
    if not use_safetensors:
        # Exclude files that are not needed for inference.
        # https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
        blacklist = [
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt",
        ]
        hf_weights_files = [
            f for f in hf_weights_files
            if not any(f.endswith(x) for x in blacklist)
        ]

    if len(hf_weights_files) == 0:
        raise RuntimeError(
            f"Cannot find any model weights with `{model_name_or_path}`")

    return hf_folder, hf_weights_files, use_safetensors


# Adapted from vllm/model_executor/weight_utils.py
def _hf_tensorfile_iterator(filename: str, load_format: str, 
                            use_safetensors: bool):
    if load_format == "npz":
        assert not use_safetensors
        with np.load(filename) as data:
            for name in data.files:
                param = torch.from_numpy(data[name])
                yield name, param
    elif use_safetensors:
        with safe_open(filename, framework="pt") as f:
            for name in f.keys():
                param = f.get_tensor(name)
                yield name, param
    else:
        state = torch.load(filename, map_location="cpu")
        for name, param in state.items():
            yield name, param
        del state
        torch.cuda.empty_cache()


def main(args):
    rank_tensors_map = {}
    hf_folder, hf_tensor_files, use_safetensors = _prepare_hf_weights(
                                                    args.quantized_model,
                                                    args.cache_dir,
                                                    args.load_format,
                                                    revision=args.revision,
                                                    fall_back_to_pt=True)
    # Matches the number immediately after this keyword in the tensor filename to
    # determine the TP rank corresponding to said tensor file
    rank_keyword = "rank"
    for tensor_file in hf_tensor_files:
        try:
            rank_idx = tensor_file.find(rank_keyword)
            if rank_idx != -1:
                start_idx = rank_idx + len(rank_keyword)
                stop_idx = start_idx
                while stop_idx < len(tensor_file) and tensor_file[stop_idx].isdecimal():
                    stop_idx += 1
                if stop_idx == start_idx:
                    raise RuntimeError("Did not find rank # in filename.")
                rank = int(tensor_file[start_idx:stop_idx])
            elif len(hf_tensor_files) == 1:
                # Since there is only one tensor file, we can assume
                # that it's intended for TP rank 0
                rank = 0
            else:
                raise RuntimeError(f"Filename does not contain '{rank_keyword}'.")
        except RuntimeError:
            print("Unable to determine TP rank "
                  f"corresponding to file '{tensor_file}'")
            raise
        
        if rank not in rank_tensors_map:
            layer_scales_map = {}
            rank_tensors_map[rank] = layer_scales_map
        else:
            raise RuntimeError(f"Tensor file '{tensor_file}' shares TP rank {rank} "
                               "with another tensor file.")
        
        module_delimiter = ":" if args.load_format == "npz" else "."
        for name, param in _hf_tensorfile_iterator(tensor_file, args.load_format,
                                                  use_safetensors):
            if "kv_cache_scaling_factor" in name:
                nums = [int(s) for s in name.split(module_delimiter) if s.isdecimal()]
                assert len(nums) == 1, f"Could not determine layer idx for {name}"
                layer_idx = nums[0]
                assert layer_idx not in layer_scales_map, f"Duplicate scaling " \
                    f"factor corresponding to layer {layer_idx}"
                try:
                    layer_scales_map[layer_idx] = param.item()
                except RuntimeError:
                    print("This utility supports only per-tensor scalar scale factors "
                            f"for now. The tensor\n {name} = {param} is an invalid "
                            "scale factor.")
                    raise

    if args.output_path is None:
        output_file = os.path.join(hf_folder, args.output_name)
    else:
        output_file = os.path.join(args.output_path, args.output_name)
        if not os.path.isdir(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

    if all(len(layer_scales_map) == 0 for layer_scales_map in rank_tensors_map.values()):
        # Note: this is true even if the rank_tensors_map is empty
        print("WARNING: No KV cache scale factors found. No output saved.")
    else:
        empirical_tp_world_size = max(rank_tensors_map.keys()) + 1
        if args.tp_size is not None:
            assert args.tp_size == empirical_tp_world_size, "User expected TP world size = " \
                f"{args.tp_size} from model but tool is expecting TP world size = " \
                f"{empirical_tp_world_size} from model instead."
        for i in range(empirical_tp_world_size):
            assert i in rank_tensors_map, f"Expected TP world size = {empirical_tp_world_size} " \
                                          "but did not find KV cache scaling factors " \
                                          f"for TP rank {i}"
        with open(output_file, 'w') as f:
            json.dump(rank_tensors_map, f, sort_keys=True, indent=4)
            print(f"Completed! Found TP world size = {empirical_tp_world_size}.",
                  f"KV cache scaling factors saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This simple utility extracts the "
                                     "KV cache scaling factors from a quantized HF model "
                                     "and saves them to a JSON file compatible with later "
                                     "use by vLLM (pass this file to the appropriate "
                                     "runtime typically using the argument "
                                     "--kv_cache_scales_path <filename>). This is only used "
                                     "if the KV cache dtype is FP8 and on ROCm (AMD GPU).")
    parser.add_argument("--quantized_model",
                        help="Specify either the local path to, or name of, a quantized HF model. "
                        "It is expected that the quantization format is FP8_E4M3, for use on ROCm "
                        "(AMD GPU).",
                        required=True)
    parser.add_argument("--cache_dir",
                        help="Optionally specify a cache directory to use in the event of a HF "
                        "model download.",
                        default=None)
    parser.add_argument("--load_format",
                        help="Optionally specify the format of the model's tensor files "
                        "containing the KV cache scaling factors.",
                        choices=["auto", "safetensors", "npz", "pt"],
                        default="auto")
    parser.add_argument("--revision",
                        help="Optionally specify the model's revision number.",
                        default=None)
    parser.add_argument("--output_dir",
                        help="Optionally specify the output directory. By default the "
                        "KV cache scaling factors will be saved in the model directory, "
                        "however you can override this behavior here.",
                        default=None)
    parser.add_argument("--output_name",
                        help="Optionally specify the output filename.",
                        default="kv_cache_scales.json")
    parser.add_argument("--tp_size",
                        help="Optionally specify the tensor-parallel (TP) size that the "
                        "quantized model should correspond to. If specified, during KV "
                        "cache scaling factor extraction the observed TP size will be "
                        "checked against this and an error will be raised if there is "
                        "a mismatch. If not specified, the quantized model's expected "
                        "TP size is instead inferred from the largest TP rank observed. "
                        "The expected TP size is cross-checked against the TP ranks "
                        "observed in the quantized model and an error is raised if any "
                        "discrepancies are found.",
                        default=None, type=int)
    args = parser.parse_args()

    main(args)
