import argparse
import fnmatch
import glob
from huggingface_hub import snapshot_download, HfFileSystem
import json
import numpy as np
import os
from safetensors.torch import safe_open
import torch
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple


# Adapted from vllm/model_executor/weight_utils.py
# The main differences are that we add the NPZ format and simplify
# its functionality drastically for our purposes (e.g. we assume that
# the quantized model exists locally and there is no need to download it)
def _prepare_hf_weights(
    quantized_model_dir: str,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
) -> Tuple[str, List[str], bool]:
    if not os.path.isdir(quantized_model_dir):
        raise FileNotFoundError(f"The quantized model directory `{quantized_model_dir}` "
                                "does not exist.")
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

    hf_weights_files: List[str] = []
    for pattern in allow_patterns:
        hf_weights_files += glob.glob(os.path.join(quantized_model_dir, pattern))
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
            f"Cannot find any model weights with `{quantized_model_dir}`")

    return hf_weights_files, use_safetensors


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


def _kv_scales_extractor(hf_tensor_files: Iterator[str],
                        use_safetensors: bool,
                        rank_keyword: str = "rank",
                        expected_tp_size: Optional[int] = None,) -> Dict[int, Dict[int, float]]:
    rank_scales_map = {}
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
        
        if rank not in rank_scales_map:
            layer_scales_map = {}
            rank_scales_map[rank] = layer_scales_map
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
                            f"for now. The tensor\n {name} = {param} \nis an invalid "
                            "scale factor.")
                    raise

    if all(len(layer_scales_map) == 0 for layer_scales_map in rank_scales_map.values()):
        # Note: this is true even if the rank_scales_map is empty
        print("WARNING: No KV cache scale factors found. No output saved.")
        return None
    empirical_tp_world_size = max(rank_scales_map.keys()) + 1
    if expected_tp_size is not None:
        assert expected_tp_size == empirical_tp_world_size, "User expected TP world size = " \
            f"{expected_tp_size} from model but tool is expecting TP world size = " \
            f"{empirical_tp_world_size} from model instead."
    for i in range(empirical_tp_world_size):
        assert i in rank_scales_map, f"Expected TP world size = {empirical_tp_world_size} " \
                                        "but did not find KV cache scaling factors " \
                                        f"for TP rank {i}"
    print(f"Found TP world size = {empirical_tp_world_size} when extracting KV cache scales!")
    return rank_scales_map


def _metadata_extractor(quantized_model_dir: str,
                        metadata_from_schema: Dict[str, Callable[[Dict[str, Any]], Any]]) -> Dict[str, Any]:
    if not os.path.isdir(quantized_model_dir):
        raise FileNotFoundError(f"The quantized model directory `{quantized_model_dir}` "
                                "does not exist.")
    allow_patterns = [ "*.json" ]

    metadata_files: List[str] = []
    for pattern in allow_patterns:
        metadata_files += glob.glob(os.path.join(quantized_model_dir, pattern))
    
    result = {}
    for file in metadata_files:
        with open(file) as f:
            try:
                schema = json.load(f)
                for metadata, from_schema_fn in metadata_from_schema.items():
                    if metadata not in result:
                        result[metadata] = from_schema_fn(schema)
                    
            except json.JSONDecodeError:
                pass
            except ValueError:
                pass
    return result


def main(args):
    metadata_from_schema = {
        "model_type": lambda schema: schema["layers"][0]["decoder_type"],
        "tp_size": lambda schema: int(schema["tensor_parallel"]),
        "model_dtype": lambda schema: schema["dtype"]
    }
    metadata_dict = _metadata_extractor(args.quantized_model, metadata_from_schema)
    model_dtype = metadata_dict["model_dtype"]

    hf_tensor_files, use_safetensors = _prepare_hf_weights(args.quantized_model, args.load_format)
    # Matches the number immediately after this keyword in the tensor filename to
    # determine the TP rank corresponding to said tensor file
    rank_keyword = "rank"
    rank_scales_map = _kv_scales_extractor(hf_tensor_files, use_safetensors,
                                           rank_keyword, args.tp_size)
    # Postprocess: formatting to the current schema. Consider pulling it out into a dedicated
    # function should it ever become more complicated.
    rank_scales_map = { rank_keyword + str(rank) : scale
                        for rank, scale in rank_scales_map.items() }

    # Consider unifying and formalizing this into its own class (and other necessary subclasses) in
    # the future
    schema = { "model_type": metadata_dict["model_type"],
               "kv_cache": {
                   "dtype": "fp8" if rank_scales_map else model_dtype,
                   "scaling_factor": rank_scales_map
               },
               # The fields below this comment are not used or checked for now
               # but will be in the future
               "activation": {
                   "dtype": model_dtype,
                   "scaling_factor": None,
               },
               "weight": {
                   "dtype": model_dtype,
                   "scaling_factor": None
               }
             }

    if args.output_dir is None:
        output_file = os.path.join(args.quantized_model, args.output_name)
    else:
        output_file = os.path.join(args.output_dir, args.output_name)
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)

   
    with open(output_file, 'w') as f:
        pass
        #json.dump(rank_scales_map, f, sort_keys=True, indent=4)
        #print(f"Completed! Found TP world size = {empirical_tp_world_size}.",
        #        f"KV cache scaling factors saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This simple utility extracts the "
                                     "KV cache scaling factors from a quantized HF model "
                                     "and saves them to a JSON file compatible with later "
                                     "use by vLLM (pass this file to the appropriate "
                                     "runtime typically using the argument "
                                     "--kv_cache_scales_path <filename>). This is only used "
                                     "if the KV cache dtype is FP8 and on ROCm (AMD GPU).")
    parser.add_argument("--quantized_model",
                        help="Specify the directory containing a single quantized HF model. "
                        "It is expected that the quantization format is FP8_E4M3, for use on ROCm "
                        "(AMD GPU).",
                        required=True)
    parser.add_argument("--load_format",
                        help="Optionally specify the format of the model's tensor files "
                        "containing the KV cache scaling factors.",
                        choices=["auto", "safetensors", "npz", "pt"],
                        default="auto")
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
