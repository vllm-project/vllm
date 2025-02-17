import os
import torch
import tqdm
from loguru import logger
import logging
import safetensors
from safetensors import safe_open
from safetensors.torch import save_file
import json

logging.basicConfig(level=logging.DEBUG)
torch.set_grad_enabled(False)

# CONSTANTS
SAFETENSORS = "safetensors"
WEIGHT_SCALE_NAME = "weight_scale_inv" #"scale_weight"
INPUT_SCALE_NAME = "scale_input"
SCALE_DTYPE = torch.bfloat16
SCALE_FILE_NAME = f"scales.{SAFETENSORS}"
FULL_RANGE = torch.finfo(torch.float8_e4m3fn).max
WEIGHT_BACKOFF = 0.5
QUANT_MODULE_TYPES = (torch.nn.Linear,)
SKIP_WEIGHT_LST = {
    "enorm.weight",
    "hnorm.weight",
    "eh_proj.weight",
    "shared_head.norm.weight",
    "shared_head.head.weight",
    "model.norm",
    "layernorm",
    "e_score_correction_bias",
    "lm_head.weight",
    "embed_tokens",
    "mlp.gate.weight",  # mlp.gate is not linear
}
"""
# https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Quantization/Inference_Using_FP8.html?highlight=backoff#supported-json-config-file-options
Similarly, the maxabs value of a weight is scaled to weight_backoff*FP8_143_FULLSCALE. The default values are input_backoff=0.25 and weight_backoff=0.5.
"""
MODEL_STATE_DICT_MAPPING_FILENAME = "model.safetensors.index.json"


def skip_weight(weight_name):
    return any([skip_name in weight_name for skip_name in SKIP_WEIGHT_LST])


def get_cpu_mem_size_in_gb():
    import psutil

    mem = psutil.virtual_memory()
    return mem.available


def get_all_weight_filename(model_path):
    all_files = os.listdir(model_path)
    all_weight_filename = []
    for file in all_files:
        if file.endswith(f".{SAFETENSORS}"):
            all_weight_filename.append(file)
    return all_weight_filename


# from _fp8_quant/_core/fp_utils.py
def calc_maxabs_scale(xmaxabs, fullscale, backoff=1):
    scale = xmaxabs / (fullscale * backoff)
    return scale


def quant_tensor(tensor):
    # Note:
    #  1. Check the scale dtype
    #  2. Check the scale shape
    amax = tensor.abs().max(dim=1).values + 1e-8
    scale = calc_maxabs_scale(amax, FULL_RANGE, WEIGHT_BACKOFF)
    scale = scale.to(SCALE_DTYPE)
    qtensor = tensor / scale.unsqueeze(1)
    cliped_qtensor = torch.clamp(qtensor, -FULL_RANGE, FULL_RANGE)
    cliped_qtensor_fp8 = cliped_qtensor.to(torch.float8_e4m3fn)
    return scale.float(), cliped_qtensor_fp8


def _maybe_create_dir(qmodel_path):
    if not os.path.exists(qmodel_path):
        os.makedirs(qmodel_path)


def quant_model_weight_with_low_cpu_usage(model_path, qmodel_path):
    _maybe_create_dir(qmodel_path)
    all_weight_filename = get_all_weight_filename(model_path)
    files_cnt = len(all_weight_filename)
    logger.info(f"Got {len(all_weight_filename)} weight files")
    qtensor_mappping = {}
    for i, filename in enumerate(all_weight_filename):
        logger.info(f"Processing {i + 1}/{len(all_weight_filename)}: {filename}")
        file_path = os.path.join(model_path, filename)
        qmodel_file_name = filename
        qmodel_file_path = os.path.join(qmodel_path, qmodel_file_name)
        qtensors = {}
        with safe_open(file_path, framework="pt", device="cpu") as f:
            for weight_name in f.keys():
                weight = f.get_tensor(weight_name)
                if skip_weight(weight_name):
                    logger.debug(f"Skipping quantize {weight_name}")
                    qtensors[weight_name] = weight
                    qtensor_mappping[weight_name] = qmodel_file_name
                    continue
                logger.debug(f"[{i+1}/{files_cnt}] Processing {weight_name}")
                scale, qtensor = quant_tensor(weight)
                preifx_name = weight_name[: -len(".weight")]
                scale_name = f"{preifx_name}.{WEIGHT_SCALE_NAME}"
                qtensors[scale_name] = scale
                qtensors[weight_name] = qtensor
                qtensor_mappping[scale_name] = qmodel_file_name
                qtensor_mappping[weight_name] = qmodel_file_name
        logger.debug(f"[{i+1}/{files_cnt}] Saving {len(qtensors)} tensors to {qmodel_file_path}")
        save_file(qtensors, os.path.join(qmodel_path, qmodel_file_path))
    # Dump tensor mapping into json file
    model_state_dict_mapping_file_path = os.path.join(qmodel_path, MODEL_STATE_DICT_MAPPING_FILENAME)
    logger.info(f"Saving tensor mapping to {model_state_dict_mapping_file_path}")
    state_dict_mapping = {
        "metadata":{},
        "weight_map": qtensor_mappping,
    }
    with open(model_state_dict_mapping_file_path, "w") as f:
        json.dump(state_dict_mapping, f, indent=4)


def _import_oh():
    import transformers
    from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

    orig_check_support_param_buffer_assignment = transformers.modeling_utils.check_support_param_buffer_assignment
    adapt_transformers_to_gaudi()
    transformers.modeling_utils.check_support_param_buffer_assignment = orig_check_support_param_buffer_assignment


@torch.no_grad()
def static_quant_model_tran(model_path, qmodel_path):
    # assert get_cpu_mem_size_in_gb(800), "Not enough memory, please use quant_model_weight_with_low_cpu_usage"
    import transformers
    from patch_for_ds import patch_transformers

    # import_oh()
    patch_transformers()
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    for name, module in model.named_modules():
        if not isinstance(module, QUANT_MODULE_TYPES) or skip_weight(name):
            logger.debug(f"Skipping quantize {name}")
            continue
        logger.debug(f"Processing {name}")
        weight = module.weight
        scale, qtensor = quant_tensor(weight)
        module.weight.data = qtensor
        setattr(module, "scale_weight", torch.nn.Parameter(scale, requires_grad=False))
    logger.info(f"Saving quantized model to {qmodel_path}")
    model.save_pretrained(qmodel_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--qmodel_path", type=str, required=True)
    parser.add_argument("--low_cpu_mem", action="store_true", help="Load weight file one by one to reduce memory usage")
    args = parser.parse_args()
    if args.low_cpu_mem:
        quant_model_weight_with_low_cpu_usage(args.model_path, args.qmodel_path)
    else:
        static_quant_model_tran(args.model_path, args.qmodel_path)

