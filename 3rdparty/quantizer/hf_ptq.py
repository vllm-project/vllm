# with AMMO installed, do below:
# python hf_ptq.py --pyt_ckpt_path="./ll2-7b" --export_path=ll2_7b_ptq_fp8 --qformat=fp8 --calib_size=128 --inference_gpus=1
# python hf_ptq.py --pyt_ckpt_path=<huggingface checkpoint path> \
#                  --export_path=llama_ptq \
#                  --qformat=fp8 \
#                  --calib_size=128 \
#                  --inference_gpus=1
#
# with TensorRT-LLM is installed, similarly do below:
# /dockerx/TensorRT-LLM/examples/quantization# python quantize.py --model_dir /dockerx/ll2-7b --dtype float16 --qformat fp8 --export_path /dockerx/ll2_7b_quantized_fp8 --calib_size 256

import argparse
import copy
import random
import time

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

import ammo.torch.quantization as atq
from ammo.torch.export import export_model_config

RAND_SEED = 1234
MAX_SEQ_LEN = 2048

QUANT_CFG_CHOICES = {
    "int8_sq": atq.INT8_SMOOTHQUANT_CFG,
    "fp8": atq.FP8_DEFAULT_CFG,
    "int4_awq": atq.INT4_AWQ_CFG,
}

def get_calib_dataloader(data="cnn_dailymail", tokenizer=None, batch_size=1, calib_size=512, block_size=512, device=None):
    print("Loading calibration dataset")
    if data == "pileval":
        dataset = load_dataset("json", data_files="https://the-eye.eu/public/AI/pile/val.jsonl.zst", split="train")
        dataset = dataset["text"][:calib_size]
    elif data == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        dataset = dataset["article"][:calib_size]
    else:
        raise NotImplementedError
    batch_encoded = tokenizer.batch_encode_plus(
        dataset, return_tensors="pt", padding=True, truncation=True, max_length=block_size
    )
    if device:
        batch_encoded = batch_encoded.to(device)
    batch_encoded = batch_encoded["input_ids"]
    calib_dataloader = DataLoader(batch_encoded, batch_size=batch_size, shuffle=False)
    return calib_dataloader


def get_tokenizer(ckpt_path, max_seq_len=MAX_SEQ_LEN):
    print(f"Initializing tokenizer from {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=max_seq_len,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_model(ckpt_path, dtype="fp16", device="cuda"):
    print(f"Initializing model from {ckpt_path}")
    if dtype == "bf16":
        dtype = torch.bfloat16
    elif dtype == "fp16":
        dtype = torch.float16
    elif dtype == "fp32":
        dtype = torch.float32
    else:
        raise NotImplementedError(f"Unknown dtype {dtype}")
    model_kwargs = {"torch_dtype": dtype}
    model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", **model_kwargs, trust_remote_code=True)
    model.eval()
    return model


def quantize_model(model, quant_cfg, calib_dataloader=None):
    def calibrate_loop():
        """Adjusts weights and scaling factors based on selected algorithms."""
        for idx, data in enumerate(calib_dataloader):
            print(f"Calibrating batch {idx}")
            model(data)

    print("Starting quantization...")
    start_time = time.time()
    atq.quantize(model, quant_cfg, forward_loop=calibrate_loop)
    end_time = time.time()
    print(f"Quantization done. Total time used: {end_time - start_time}s")
    return model


def _register_falcon_linears(model):
    """Register Falcon linear modules as Quantiation.

    As falcon models could use remote code, which will be loaded dynamically, to build their model.
    Therefore, we need to register the linear on the fly before quantization.

    """
    if type(model).__name__ in ["RWForCausalLM", "FalconForCausalLM"]:
        from ammo.torch.quantization import tensor_quant
        from ammo.torch.quantization.nn.modules.quant_module import QuantLinearConvBase

        linear_type = type(model.transformer.h[0].self_attention.dense)

        class QuantFalconLinearRW1B(linear_type, QuantLinearConvBase):  # type: ignore
            default_quant_desc_weight = tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW

        atq.module_mapping.QUANT_MODULE_MAPPING[linear_type] = QuantFalconLinearRW1B.convert



def main(args):
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU is required for inference.")

    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)

    tokenizer = get_tokenizer(args.pyt_ckpt_path)
    model = get_model(args.pyt_ckpt_path, args.dtype, args.device)

    _register_falcon_linears(model)
    if args.qformat in ["fp8", "int8_sq", "int4_awq"]:
        if args.qformat == "int4_awq":
            if args.calib_size > 32:
                calib_size = 32
                print(
                    f"AWQ calibration could take longer with calib_size = {args.calib_size}, Using"
                    f" calib_size={calib_size} instead"
                )
            print(
                "\nAWQ calibration could take longer than other calibration methods. Please"
                " increase the batch size to speed up the calibration process. Batch size can be"
                " set by adding the argument --batch_size <batch_size> to the command line.\n"
            )
        else:
            calib_size = args.calib_size

        calib_dataloader = get_calib_dataloader(
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            calib_size=calib_size,
            device=args.device,
        )
        if args.qformat in QUANT_CFG_CHOICES:
            quant_cfg = QUANT_CFG_CHOICES[args.qformat]
        else:
            raise ValueError(f"Unsupported quantization format: {args.qformat}")

        if args.qformat == "int4_awq":
            quant_cfg = copy.deepcopy(atq.INT4_AWQ_CFG)
            quant_cfg["quant_cfg"]["*weight_quantizer"]["block_sizes"][-1] = args.awq_block_size  # type: ignore

        model = quantize_model(model, quant_cfg, calib_dataloader)
    else:
        print(f"No quantization applied, export {args.dtype} model")


    with torch.inference_mode():
        if any([k in type(model).__name__ for k in ["Llama", "Mistral"]]):
            model_type = "llama"
        elif "GPTJ" in type(model).__name__:
            model_type = "gptj"
        elif type(model).__name__ in ["FalconForCausalLM", "RWForCausalLM"]:
            model_type = "falcon"
        elif "baichuan" in type(model).__name__.lower():
            model_type = "baichuan"
        elif "MPT" in type(model).__name__:
            model_type = "mpt"
        else:
            print(f"Unknown model type {type(model).__name__}. Continue exporting...")
            model_type = f"unknown:{type(model).__name__}"

        export_path = args.export_path
        start_time = time.time()
        export_model_config(
            model,
            model_type,
            torch.float16,
            export_dir=export_path,
            inference_tensor_parallel=int(args.inference_gpus),
        )
        end_time = time.time()
        print(
            f"Quantized model exported to :{export_path}. Total time used {end_time - start_time}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyt_ckpt_path", help="Specify where the PyTorch checkpoint path is", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", help="Model data type.", default="fp16")
    parser.add_argument("--qformat", help="Quantization format.", default="fp8")
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--calib_size", help="Number of samples for calibration.", type=int, default=512)
    parser.add_argument("--export_path", default="exported_model")
    parser.add_argument("--inference_gpus", default=1)
    parser.add_argument("--awq_block_size", default=128)

    args = parser.parse_args()

    main(args)


