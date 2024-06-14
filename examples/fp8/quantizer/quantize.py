# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Adapted from examples/quantization/hf_ptq.py
"""

import argparse
import copy
import json
import random
import time

import ammo.torch.quantization as atq
import numpy as np
import torch
from ammo.torch.export import export_model_config
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

RAND_SEED = 1234
MAX_SEQ_LEN = 2048

EMPTY_CFG = {
    "quant_cfg": {
        "*weight_quantizer": {
            "enable": False,
        },
        "*input_quantizer": {
            "enable": False
        },
        "*lm_head*": {
            "enable": False
        },
        "*output_layer*": {
            "enable": False
        },
        "default": {
            "enable": False
        },
    },
    "algorithm": "max",
}

KV_CACHE_CFG = {
    "*.query_key_value.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.Wqkv.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.W_pack.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.c_attn.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.k_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
    "*.v_proj.output_quantizer": {
        "num_bits": 8,
        "axis": None,
        "enable": True
    },
}

QUANT_CFG_CHOICES = {
    "int8_sq": atq.INT8_SMOOTHQUANT_CFG,
    "fp8": atq.FP8_DEFAULT_CFG,
    "int4_awq": atq.INT4_AWQ_CFG,
    "w4a8_awq": atq.W4A8_AWQ_BETA_CFG,
    "int8_wo": EMPTY_CFG,
    "int4_wo": EMPTY_CFG,
    "full_prec": EMPTY_CFG,
}

MODEL_NAME_PATTERN_MAP = {
    "GPT2": "gpt2",
    "Xverse": "llama",
    "Llama": "llama",
    "Mistral": "llama",
    "GPTJ": "gptj",
    "FalconForCausalLM": "falcon",
    "RWForCausalLM": "falcon",
    "baichuan": "baichuan",
    "MPT": "mpt",
    "Bloom": "bloom",
    "ChatGLM": "chatglm",
    "QWen": "qwen",
}


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN, model_type=None):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )
    if model_type and model_type == "qwen":
        # qwen use token id 151643 as pad and eos tokens
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(151643)
        tokenizer.eos_token = tokenizer.convert_ids_to_tokens(151643)

    # can't set attribute 'pad_token' for "<unk>"
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    assert (tokenizer.pad_token
            is not None), f"Pad token for {model_type} cannot be set!"

    return tokenizer


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16" or dtype == "bfloat16":
        dtype = torch.bfloat16
    elif dtype == "fp16" or dtype == "float16":
        dtype = torch.float16
    elif dtype == "fp32" or dtype == "float32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")

    # model_kwargs = {"torch_dtype": dtype}
    model_kwargs = {"torch_dtype": "auto"}

    model = AutoModelForCausalLM.from_pretrained(ckpt_path,
                                                 device_map="auto",
                                                 **model_kwargs,
                                                 trust_remote_code=True)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    if dtype != model_dtype:
        print("[TensorRT-LLM][WARNING] The manually set model data type is "
              f"{dtype}, but the data type of the HuggingFace model is "
              f"{model_dtype}.")

    return model


def get_model_type(model):
    for k, v in MODEL_NAME_PATTERN_MAP.items():
        if k.lower() in type(model).__name__.lower():
            return v
    return None


def get_calib_dataloader(data="cnn_dailymail",
                         tokenizer=None,
                         batch_size=1,
                         calib_size=512,
                         block_size=512,
                         device=None):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset(
            "json",
            data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst",
            split="train")
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError

    batch_encoded = tokenizer.batch_encode_plus(dataset,
                                                return_tensors="pt",
                                                padding="max_length",
                                                truncation=True,
                                                max_length=block_size)
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]

    calib_dataloader = DataLoader(batch_encoded,
                                  batch_size=batch_size,
                                  shuffle=False)

    return calib_dataloader


def quantize_model(model, quant_cfg, calib_dataloader=None):

    def calibrate_loop():
        if calib_dataloader is None:
            return
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating batch {idx}")
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print("Quantization done. Total time used: {:.2f} s.".format(end_time -
                                                                 start_time))

    return model


def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    model = get_model(args.model_dir, args.dtype, args.device)
    model_type = get_model_type(model)
    tokenizer = get_tokenizer(args.model_dir, model_type=model_type)

    if args.qformat in ["full_prec", "int8_wo", "int4_wo"
                        ] and args.kv_cache_dtype is None:
        print(f"No quantization applied, export {args.dtype} model")
    else:
        if "awq" in args.qformat:
            if args.calib_size > 32:
                print("AWQ calibration could take longer with calib_size = "
                      f"{args.calib_size}, Using calib_size=32 instead")
                args.calib_size = 32
            print("\nAWQ calibration could take longer than other calibration "
                  "methods. Please increase the batch size to speed up the "
                  "calibration process. Batch size can be set by adding the "
                  "argument --batch_size <batch_size> to the command line.\n")

        calib_dataloader = get_calib_dataloader(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            calib_size=args.calib_size,
            device=args.device,
        )

        if args.qformat in QUANT_CFG_CHOICES:
            quant_cfg = QUANT_CFG_CHOICES[args.qformat]
        else:
            raise ValueError(
                f"Unsupported quantization format: {args.qformat}")

        if "awq" in args.qformat:
            quant_cfg = copy.deepcopy(QUANT_CFG_CHOICES[args.qformat])
            weight_quantizer = quant_cfg["quant_cfg"][
                "*weight_quantizer"]  # type: ignore
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = args.awq_block_size

        if args.kv_cache_dtype is not None:
            if args.kv_cache_dtype == "fp8":
                for value in KV_CACHE_CFG.values():
                    value.update({"num_bits": (4, 3)})  # type: ignore
            quant_cfg["quant_cfg"].update(KV_CACHE_CFG)  # type: ignore

        print(quant_cfg)

        model = quantize_model(model, quant_cfg, calib_dataloader)

    with torch.inference_mode():
        if model_type is None:
            print(f"Unknown model type {type(model).__name__}. Continue "
                  "exporting...")
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.output_dir
        start_time = time.time()

        if args.qformat == "int4_awq" and model_type == "qwen":
            torch.save(model.state_dict(), export_path)
        else:
            export_npz = (model_type not in [
                'gptj', 'falcon', 'chatglm', 'mpt', 'llama', 'baichuan'
            ])

            # export safetensors
            export_model_config(
                model,
                model_type,
                getattr(torch, args.dtype),
                export_dir=export_path,
                inference_tensor_parallel=args.tp_size,
                inference_pipeline_parallel=args.pp_size,
                # export_tensorrt_llm_config=(not export_npz),
                export_tensorrt_llm_config=False,
                export_npz=export_npz)

            # Workaround for wo quantization
            if args.qformat in ["int8_wo", "int4_wo", "full_prec"]:
                with open(f"{export_path}/config.json", 'r') as f:
                    tensorrt_llm_config = json.load(f)
                if args.qformat == "int8_wo":
                    tensorrt_llm_config["quantization"]["quant_algo"] = 'W8A16'
                elif args.qformat == "int4_wo":
                    tensorrt_llm_config["quantization"]["quant_algo"] = 'W4A16'
                else:
                    tensorrt_llm_config["quantization"]["quant_algo"] = None
                with open(f"{export_path}/config.json", "w") as f:
                    json.dump(tensorrt_llm_config, f, indent=4)

        end_time = time.time()
        print("Quantized model exported to {} \nTotal time used {:.2f} s.".
              format(export_path, end_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",
                        help="Specify where the HuggingFace model is",
                        required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", help="Model data type.", default="float16")
    parser.add_argument(
        "--qformat",
        help="Quantization format.",
        default="full_prec",
        choices=[
            "fp8", "int8_sq", "int4_awq", "w4a8_awq", "int8_wo", "int4_wo",
            "full_prec"
        ],
    )
    parser.add_argument("--batch-size",
                        help="Batch size for calibration.",
                        type=int,
                        default=1)
    parser.add_argument("--calib-size",
                        help="Number of samples for calibration.",
                        type=int,
                        default=512)
    parser.add_argument("--output-dir", default="exported_model")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--awq-block-size", type=int, default=128)
    parser.add_argument("--kv-cache-dtype",
                        help="KV Cache dtype.",
                        default=None,
                        choices=["int8", "fp8", None])
    args = parser.parse_args()

    main(args)
