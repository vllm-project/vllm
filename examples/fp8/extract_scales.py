import argparse
import glob
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import safe_open

from vllm.model_executor.layers.quantization.schema import QuantParamSchema


# Adapted from vllm/model_executor/model_loader/weight_utils.py
# The main differences are that we add the NPZ format and simplify
# its functionality drastically for our purposes (e.g. we assume that
# the quantized model exists locally and there is no need to download it)
def _prepare_hf_weights(
    quantized_model_dir: str,
    load_format: str = "auto",
    fall_back_to_pt: bool = True,
) -> Tuple[List[str], bool]:
    if not os.path.isdir(quantized_model_dir):
        raise FileNotFoundError(
            f"The quantized model directory `{quantized_model_dir}` "
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
        hf_weights_files += glob.glob(
            os.path.join(quantized_model_dir, pattern))
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


# Adapted from vllm/model_executor/model_loader/weight_utils.py
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
            for name in f.keys():  # NOQA: SIM118
                param = f.get_tensor(name)
                yield name, param
    else:
        state = torch.load(filename, map_location="cpu")
        for name, param in state.items():
            yield name, param
        del state
        torch.cuda.empty_cache()


def _kv_scales_extractor(
        hf_tensor_files: List[str],
        use_safetensors: bool,
        rank_keyword: str = "rank",
        expected_tp_size: Optional[int] = None) -> Dict[int, Dict[int, float]]:
    """
    Given a list of files containing tensor data, attempt to extract KV cache
    scales from these files. Intended as a helper function taking in the output
    from _prepare_hf_weights.
    Args:
    rank_keyword        Matches the number immediately after this keyword in the
                        tensor filename to determine the TP rank corresponding
                        to said tensor file
    expected_tp_size    If specified, the TP size of the tensor files is checked
                        against this and an error is raised if they don't match.
    Returns a dictionary mapping TP ranks to their relevant KV cache scales.
    The per-rank scales are themselves represented as a dictionary of layer
    indices to the respective per-layer scale.
    """
    for char in rank_keyword:
        assert not char.isdecimal(
        ), f"Rank keyword {rank_keyword} contains a numeric character!"
    rank_scales_map: Dict[int, Dict[int, float]] = {}
    for tensor_file in hf_tensor_files:
        try:
            rank_idx = tensor_file.find(rank_keyword)
            if rank_idx != -1:
                start_idx = rank_idx + len(rank_keyword)
                stop_idx = start_idx
                while stop_idx < len(
                        tensor_file) and tensor_file[stop_idx].isdecimal():
                    stop_idx += 1
                if stop_idx == start_idx:
                    raise RuntimeError("Did not find rank # in filename.")
                rank = int(tensor_file[start_idx:stop_idx])
            elif len(hf_tensor_files) == 1:
                # Since there is only one tensor file, we can assume
                # that it's intended for TP rank 0
                rank = 0
            else:
                raise RuntimeError(
                    f"Filename does not contain '{rank_keyword}'.")
        except RuntimeError:
            print("Unable to determine TP rank "
                  f"corresponding to file '{tensor_file}'")
            raise

        if rank not in rank_scales_map:
            layer_scales_map: Dict[int, float] = {}
            rank_scales_map[rank] = layer_scales_map
        else:
            raise RuntimeError(
                f"Tensor file '{tensor_file}' shares TP rank {rank} "
                "with another tensor file.")

        module_delimiter = ":" if args.load_format == "npz" else "."
        for name, param in _hf_tensorfile_iterator(tensor_file,
                                                   args.load_format,
                                                   use_safetensors):
            if "kv_cache_scaling_factor" in name:
                nums = [
                    int(s) for s in name.split(module_delimiter)
                    if s.isdecimal()
                ]
                assert len(
                    nums) == 1, f"Could not determine layer idx for {name}"
                layer_idx = nums[0]
                assert layer_idx not in layer_scales_map, f"Duplicate scaling"\
                    f" factor corresponding to layer {layer_idx}"
                try:
                    layer_scales_map[layer_idx] = param.item()
                except RuntimeError:
                    print(
                        "This utility supports only per-tensor scalar scales "
                        f"for now. The tensor\n {name} = {param} \nis an "
                        "invalid scale factor.")
                    raise

    if all(
            len(layer_scales_map) == 0
            for layer_scales_map in rank_scales_map.values()):
        # Note: this is true even if the rank_scales_map is empty
        print("WARNING: No KV cache scale factors found. No output saved.")
        return None
    empirical_tp_world_size = max(rank_scales_map.keys()) + 1
    if expected_tp_size is not None:
        assert expected_tp_size == empirical_tp_world_size, \
            f"User expected TP world size = {expected_tp_size} " \
            "from model but tool is expecting TP world size = " \
            f"{empirical_tp_world_size} from model instead."
    for i in range(empirical_tp_world_size):
        assert i in rank_scales_map, "Expected TP world size = "\
            f"{empirical_tp_world_size} but did not find KV " \
            f"cache scaling factors for TP rank {i}"
    print(f"Found TP world size = {empirical_tp_world_size} "
          "when extracting KV cache scales!")
    return rank_scales_map


def _metadata_extractor(quantized_model_dir: str,
                        metadata_extract_fns: \
                        Dict[str, Callable[[Dict[str, Any]], Any]]) \
                        -> Dict[str, Any]:
    """
    Given a directory containing quantized model files, this function
    aims to extract metadata from the JSON files within this directory.
    Each JSON file is expected to represent a dictionary in JSON
    format (referred to as a "JSON-dictionary"). Metadata extraction is
    defined by a dictionary called metadata_extract_fns, where each
    metadata field name is mapped to an extraction function.

    These extraction functions are designed to take a JSON-dictionary
    as their only argument  and return the corresponding metadata.
    While extraction functions are permitted to raise  exceptions, they
    should only raise a KeyError or ValueError if the metadata field
    cannot  be extracted from the current JSON-dictionary, yet there's
    a possibility of finding it in another JSON-dictionary.

    The function returns a dictionary that maps metadata fields to
    their extracted data. The keys of this dictionary correspond exactly
    to those in metadata_extract_fns. If any fields fail to be extracted,
    their corresponding values are set to None, and a warning is printed.
    """
    if not os.path.isdir(quantized_model_dir):
        raise FileNotFoundError(
            f"The quantized model directory `{quantized_model_dir}` "
            "does not exist.")
    metadata_files = glob.glob(os.path.join(quantized_model_dir, "*.json"))

    result: Dict[str, Any] = {}
    for file in metadata_files:
        with open(file) as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                print(f"Could not parse `{file}` as a valid metadata file,"
                      " skipping it.")
                continue
            if not isinstance(metadata, dict):
                print(f"The file `{file}` does not correspond to a "
                      "JSON-serialized dictionary, skipping it.")
                continue
            for metadata_name, extract_fn in metadata_extract_fns.items():
                try:
                    metadata_info = extract_fn(metadata)
                    if metadata_name not in result:
                        result[metadata_name] = metadata_info
                    elif metadata_info != result[metadata_name]:
                        raise RuntimeError(
                            "Metadata mismatch! Originally found "
                            f"{metadata_name} = {result[metadata_name]} but "
                            f"now found {metadata_name} = {metadata_info} in "
                            f"`{file}`")
                except KeyError:
                    # It is possible that a given file does not contain some
                    # of our selected metadata as it could be located in some
                    # other metadata file.
                    # 'EFINAE': extract_fn failure is not an error.
                    pass
                except ValueError:
                    # See above.
                    pass

    # Warn if we cannot find any of the requested metadata
    for metadata_name in metadata_extract_fns:
        if metadata_name not in result:
            print("WARNING: Unable to find requested metadata field "
                  f"`{metadata_name}`, setting it to None.")
            result[metadata_name] = None

    return result


def main(args):
    metadata_extract_fns = {
        "model_type": lambda json_dict: json_dict["layers"][0]["decoder_type"],
        "tp_size": lambda json_dict: int(json_dict["tensor_parallel"]),
        "model_dtype": lambda json_dict: json_dict["dtype"]
    }
    recovered_metadata = _metadata_extractor(args.quantized_model,
                                             metadata_extract_fns)
    if args.tp_size is not None:
        metadata_tp_size = recovered_metadata["tp_size"]
        if metadata_tp_size is not None:
            assert args.tp_size == metadata_tp_size, \
              f"User expected TP world size = {args.tp_size} " \
              f"but found TP world size = {metadata_tp_size} from metadata!"
    expected_tp_size = args.tp_size or recovered_metadata["tp_size"]
    rank_keyword = "rank"
    hf_tensor_files, use_safetensors = _prepare_hf_weights(
        args.quantized_model, args.load_format)
    rank_scales_map = _kv_scales_extractor(hf_tensor_files, use_safetensors,
                                           rank_keyword, expected_tp_size)
    # Postprocess: formatting to the current schema. Consider pulling it
    # out into a dedicated function should it ever become more complicated.
    rank_scales_map = {
        rank: {k: scale[k]
               for k in sorted(scale.keys())}
        for rank, scale in rank_scales_map.items()
    }
    # TODO: Expand this with activation and weights scaling factors when
    # they are used in the future
    schema = QuantParamSchema(
        model_type=recovered_metadata["model_type"],
        kv_cache={
            "dtype": ("float8_e4m3fn" if len(rank_scales_map) > 0 else
                      recovered_metadata["model_dtype"]),
            "scaling_factor":
            rank_scales_map
        },
    )

    if args.output_dir is None:
        output_file = os.path.join(args.quantized_model, args.output_name)
    else:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        output_file = os.path.join(args.output_dir, args.output_name)

    with open(output_file, 'w') as f:
        f.write(schema.model_dump_json(indent=4))
        print(f"Completed! KV cache scaling factors saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This simple utility extracts the "
        "KV cache scaling factors from a quantized HF model "
        "and saves them to a JSON file compatible with later "
        "use by vLLM (pass this file to the appropriate "
        "runtime typically using the argument "
        "--quantization-param-path <filename>). This is only used "
        "if the KV cache dtype is FP8 and on ROCm (AMD GPU).")
    parser.add_argument(
        "--quantized-model",
        help="Specify the directory containing a single quantized HF model. "
        "It is expected that the quantization format is FP8_E4M3, for use "
        "on ROCm (AMD GPU).",
        required=True)
    parser.add_argument(
        "--load_format",
        help="Optionally specify the format of the model's tensor files "
        "containing the KV cache scaling factors.",
        choices=["auto", "safetensors", "npz", "pt"],
        default="auto")
    parser.add_argument(
        "--output-dir",
        help="Optionally specify the output directory. By default the "
        "KV cache scaling factors will be saved in the model directory, "
        "however you can override this behavior here.",
        default=None)
    parser.add_argument(
        "--output-name",
        help="Optionally specify the output filename.",
        # TODO: Change this once additional scaling factors are enabled
        default="kv_cache_scales.json")
    parser.add_argument(
        "--tp-size",
        help="Optionally specify the tensor-parallel (TP) size that the "
        "quantized model should correspond to. If specified, during KV "
        "cache scaling factor extraction the observed TP size will be "
        "checked against this and an error will be raised if there is "
        "a mismatch. If not specified, the quantized model's expected "
        "TP size is instead inferred from the largest TP rank observed. "
        "The expected TP size is cross-checked against the TP ranks "
        "observed in the quantized model and an error is raised if any "
        "discrepancies are found.",
        default=None,
        type=int)
    args = parser.parse_args()

    main(args)
