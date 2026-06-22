# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize a small MoE model to Quark W8A8-INT8 and (optionally) upload to HF.

This produces a checkpoint compatible with ``QuarkW8A8Int8MoEMethod`` (the
oracle/modular-kernel INT8 MoE path), i.e. per-channel INT8 weights with
dynamic per-token INT8 activations.

It is intended to (re)generate the small model used by
``tests/quantization/test_quark.py::test_quark_int8_w8a8_moe`` and the gsm8k
eval config.

Usage:
    python tests/quantization/quantize_int8_moe_quark.py \
        --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --output-dir /tmp/qwen-moe-w8a8-int8-quark \
        --push-to-hub <org>/<repo>   # optional; requires `huggingface-cli login`

Requires the `quark` package (bundled in the ROCm vLLM dev image):
    https://quark.docs.amd.com/
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="Qwen/Qwen1.5-MoE-A2.7B-Chat",
        help="Base (unquantized) MoE model to quantize.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Local directory to write the quantized checkpoint to.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional HF repo id (e.g. 'amd/<name>') to upload the result to.",
    )
    parser.add_argument(
        "--num-calib-samples",
        type=int,
        default=128,
        help="Number of calibration samples for activation observers.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from quark.torch import ModelQuantizer
    from quark.torch.export import ExporterConfig, JsonExporterConfig
    from quark.torch.quantization import (
        Config,
        QuantizationConfig,
        QuantizationSpec,
    )
    from quark.torch.quantization.config.type import Dtype, QSchemeType
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="cuda"
    )
    model.eval()

    # Per-channel INT8 weights (static) + per-token INT8 activations (dynamic),
    # symmetric — matches QuarkW8A8Int8MoEMethod's supported scheme.
    weight_spec = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        symmetric=True,
        is_dynamic=False,
    )
    act_spec = QuantizationSpec(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        symmetric=True,
        is_dynamic=True,
    )
    global_cfg = QuantizationConfig(weight=weight_spec, input_tensors=act_spec)
    quant_config = Config(
        global_quant_config=global_cfg,
        exclude=["lm_head"],
    )

    # Calibration data.
    ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    samples = [
        tokenizer(ds[i]["text"], return_tensors="pt").input_ids.to("cuda")
        for i in range(min(args.num_calib_samples, len(ds)))
    ]

    quantizer = ModelQuantizer(quant_config)

    def calib_iter():
        for input_ids in samples:
            yield {"input_ids": input_ids}

    with torch.no_grad():
        model = quantizer.quantize_model(model, calib_iter())
        model = quantizer.freeze(model)

    exporter_config = ExporterConfig(json_export_config=JsonExporterConfig())
    from quark.torch import ModelExporter

    exporter = ModelExporter(config=exporter_config, export_dir=args.output_dir)
    with torch.no_grad():
        exporter.export_safetensors_model(
            model, quant_config=quant_config, tokenizer=tokenizer
        )
    tokenizer.save_pretrained(args.output_dir)
    print(f"Quantized checkpoint written to {args.output_dir}")

    if args.push_to_hub:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub)
        print(f"Uploaded to https://huggingface.co/{args.push_to_hub}")


if __name__ == "__main__":
    main()
