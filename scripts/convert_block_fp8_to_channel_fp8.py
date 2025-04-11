import os
import json
import torch
from compress_pickle import load


from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONSTANTS
SAFETENSORS = "safetensors"
WEIGHT_SCALE_NAME = "weight_scale_inv"
MODEL_STATE_DICT_MAPPING_FILENAME = "model.safetensors.index.json"
# end constants

def get_device_and_range():
    device = os.popen("hl-smi -Q name -f csv | tail -n 1").read().strip()
    if 'HL-225' in device:
        return device, 240.0
    elif 'HL-328' in device:
        return device, 448.0
    elif 'HL-325' in device:
        return device, 448.0
    else:
        raise ValueError(f"Unknown device: {device}")

def get_input_scales(pkl_path):
    input_scales = {}
    if pkl_path is not None:
        with open(pkl_path, 'rb') as file:
            input_scales = load(file)
    return input_scales

def maybe_create_dir(qmodel_path):
    os.makedirs(qmodel_path, exist_ok=True)


def get_all_weight_filename(model_path):
    return [file for file in os.listdir(model_path) if file.endswith(f".{SAFETENSORS}")]


def pad_weight(weight, block_size):
    """Pads a matrix to make its dimensions multiples of block_size."""
    M, N = weight.shape[-2:]
    block_size_m, block_size_n = block_size
    pad_M = (block_size_m - M % block_size_m) % block_size_m
    pad_N = (block_size_n - N % block_size_n) % block_size_n

    if pad_M == 0 and pad_N == 0:
        return weight, M, N  # No padding needed
    padded_weight = torch.nn.functional.pad(weight, (0, pad_N, 0, pad_M), mode='constant', value=0)
    padded_weight = torch.nn.Parameter(padded_weight, requires_grad=False)
    return padded_weight, M, N  # Return original dimensions for unpadding


def unpad_weight(weight, original_M, original_N, keep_first_dim=False):
    """Removes padding from the matrix to restore its original shape."""
    if weight.size(-2) == 640:
        return weight[:original_M, :]
    else:
        return weight


def pad_block_fp8_weight_naive(weight, weight_scale, block_size):
    assert len(block_size) == 2
    block_size_m, block_size_n = block_size
    weight_scale_m, weight_scale_n = weight_scale.shape[-2:]

    weight, orig_M, orig_N = pad_weight(weight, block_size)
    M, N = weight.shape[-2:]

    assert weight_scale_m == M // block_size_m
    assert weight_scale_n == N // block_size_n
    return weight, orig_M, orig_N


def dequant_block_fp8_weight_naive(weight, weight_scale, block_size, dtype, original_M, original_N):
    assert len(block_size) == 2
    weight_shape_len = len(weight.shape)
    block_size_m, block_size_n = block_size
    logger.debug(f"weight shape is {weight.shape} and weight_scale shape is {weight_scale.shape}")
    # mul scale
    if weight_shape_len == 2:
        weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = False
    elif weight_shape_len == 3:
        fd, weight_scale_m, weight_scale_n = weight_scale.shape
        weight_scale = weight_scale.view(fd, weight_scale_m, 1, weight_scale_n, 1)
        weight = weight.view(fd, weight_scale_m, block_size_m, weight_scale_n, block_size_n)
        dequant_weight = weight.to(dtype) * weight_scale.to(dtype)
        dequant_weight = dequant_weight.view(fd, weight_scale_m*block_size_m, weight_scale_n*block_size_n)
        keep_first_dim = True
    else:
        raise ValueError("Only support original weight shape is either 2 or 3")

    dequant_weight = unpad_weight(dequant_weight, original_M, original_N, keep_first_dim=keep_first_dim)
    return dequant_weight


def dynamic_quant(data):
    scale = (torch.abs(data)).max(dim=1).values / FULL_RANGE
    scale = scale.unsqueeze(-1)
    data = data / scale
    data_fp8 = data.to(torch.float8_e4m3fn)
    return data_fp8, scale.float()


def main(model_path: str, qmodel_path: str, input_scales_path: str) -> None:
    torch.set_grad_enabled(False)
    maybe_create_dir(qmodel_path)
    # copy all files start with config* and tokenizer* from model_path to qmodel_path
    for file in os.listdir(model_path):
        if file.startswith("config") or file.startswith("tokenizer"):
            logger.info(f"Copying {file} from {model_path} to {qmodel_path}")
            file_path = os.path.join(model_path, file)
            os.system(f"cp {file_path} {qmodel_path}")
    if os.path.exists(os.path.join(qmodel_path, "config.json")):
        with open(os.path.join(qmodel_path, "config.json"), "r") as f:
            config = json.load(f)
        config["quantization_config"] = {
            "activation_scheme": "static" if input_scales_path is not None else "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8"
        }
        with open(os.path.join(qmodel_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)
    all_weight_filename = sorted(get_all_weight_filename(model_path))
    logger.info(f"Got {len(all_weight_filename)} weight files")
    qtensor_mapping = {}
    weight_map = json.load(open(f"{model_path}/{MODEL_STATE_DICT_MAPPING_FILENAME}"))["weight_map"]

    input_scales = get_input_scales(input_scales_path)

    for i, filename in tqdm(enumerate(all_weight_filename), total=len(all_weight_filename)):
        logger.debug(f"Processing {i + 1}/{len(all_weight_filename)}: {filename}")
        file_path = os.path.join(model_path, filename)
        qmodel_file_path = os.path.join(qmodel_path, filename)
        qtensors = {}

        with safe_open(file_path, framework="pt", device="cpu") as f:
            for name in f.keys():
                logger.debug(f"[{i+1}/{len(all_weight_filename)}] Processing {name}")
                if "model.layers.61" in name:
                    logger.debug(f"Ignoring {name}")
                    continue
                elif "proj" in name and "scale_inv" in name:
                    weight_scale_name = name
                    weight_name = name[: -len("_scale_inv")]
                    logger.debug(f"Begin quantizing weight: {weight_name} with scale: {weight_scale_name}")

                    scale = f.get_tensor(name)
                    scale_file = weight_map.get(weight_scale_name)
                    weight_file = weight_map.get(weight_name)
                    if scale_file != weight_file:
                        with safe_open(os.path.join(model_path, weight_file), framework="pt", device="cpu") as wf:
                            weight = wf.get_tensor(weight_name)
                    else:
                        weight = f.get_tensor(weight_name)

                    block_size = [128, 128]
                    weight, orig_M, orig_N = pad_block_fp8_weight_naive(weight, scale, block_size)
                    deq_weights = dequant_block_fp8_weight_naive(weight, scale, block_size, torch.float32, orig_M, orig_N)
                    qtensor, scale = dynamic_quant(deq_weights)
                    qtensors[weight_scale_name] = scale.squeeze(1)
                    qtensors[weight_name] = qtensor
                    qtensor_mapping[weight_scale_name] = filename
                    qtensor_mapping[weight_name] = filename

                    input_scale_name = weight_scale_name.replace("weight_scale_inv", "input_scale_inv")
                    if input_scale_name in input_scales.keys():
                        input_scale = input_scales.pop(input_scale_name)
                        input_scale = input_scale * 448.0 / FULL_RANGE
                        input_scale_name = input_scale_name.replace("input_scale_inv", "input_scale")
                        qtensors[input_scale_name] = input_scale
                        qtensor_mapping[input_scale_name] = filename
                    
                    logger.debug(f"Completed quantizing weight: {weight_name} with scale: {weight_scale_name}")
                elif "proj" in name and not ("scale_inv" in name) and not ("eh_" in name):
                    logger.debug(f"Ignoring {name}")
                    continue
                else:
                    logger.debug(f"Skipping quantization for {name}")
                    weight_name = name
                    weight = f.get_tensor(weight_name)
                    qtensors[weight_name] = weight
                    qtensor_mapping[weight_name] = filename
        if bool(qtensors):
            logger.info(f"[{i+1}/{len(all_weight_filename)}] Saving {len(qtensors)} tensors to {qmodel_file_path}")
            save_file(qtensors, qmodel_file_path)
    if input_scales.keys():
        logger.warning(f"warning: the following input_scales are unused:")
        for k in input_scales.keys():
            logger.warning(k)

    model_state_dict_mapping_file_path = os.path.join(qmodel_path, MODEL_STATE_DICT_MAPPING_FILENAME)
    print(f"Saving tensor mapping to {model_state_dict_mapping_file_path}")
    state_dict_mapping = {
        "metadata": {},
        "weight_map": qtensor_mapping,
    }
    with open(model_state_dict_mapping_file_path, "w") as f:
        json.dump(state_dict_mapping, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--qmodel_path", type=str, required=True)
    parser.add_argument("--input_scales_path", type=str, default = None)
    args = parser.parse_args()

    if os.path.exists(args.qmodel_path):
        raise ValueError(f"{args.qmodel_path} already exists, please remove or backup.")
    
    DEVICE, FULL_RANGE = get_device_and_range()
    logger.info(f"Using device: {DEVICE} with full range: {FULL_RANGE}")

    main(args.model_path, args.qmodel_path, args.input_scales_path)

    print("Conversion is completed.")
