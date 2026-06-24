# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantize a small MoE model to Quark W8A8-INT8 and (optionally) upload to HF.

This produces a checkpoint compatible with ``QuarkW8A8Int8MoEMethod`` (the
oracle/modular-kernel INT8 MoE path), i.e. per-channel INT8 weights with
dynamic per-token INT8 activations (symmetric).

IMPORTANT (transformers version): Quark's ``ModelQuantizer`` only quantizes
``nn.Linear`` modules. transformers >= 5 fuses MoE experts into a single
batched module (e.g. ``Qwen3MoeExperts`` with 3D weight tensors), which is
NOT ``nn.Linear``, so the routed experts are silently left unquantized. To
quantize the experts you must run this script in an environment with a
transformers version that keeps experts as individual ``nn.Linear`` layers
(~4.57, the version used to produce the reference test model
``nameistoken/tiny-qwen3-moe-w8a8-int8-quark``). The resulting per-expert
checkpoint then loads fine in current vLLM. Prefer a Qwen3-MoE model (no
shared experts) to avoid the qwen2_moe shared-expert loader path.

Usage:
    python tests/quantization/quantize_int8_moe_quark.py \
        --model Qwen/Qwen1.5-MoE-A2.7B-Chat \
        --output-dir /tmp/qwen-moe-w8a8-int8-quark \
        --push-to-hub <org>/<repo>   # optional; requires `hf auth login`

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
        default=32,
        help="Number of calibration samples (used to run observers).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["lm_head"],
        help=(
            "Module name patterns to leave unquantized. Routers/gates and "
            "(for architectures like qwen2_moe) the shared expert should be "
            "excluded, e.g. --exclude lm_head '*.gate' '*.shared_expert.*' "
            "'*.shared_expert_gate'."
        ),
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Max sequence length per calibration sample.",
    )
    args = parser.parse_args()

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from quark.torch import ModelQuantizer, export_safetensors
    from quark.torch.quantization import Int8PerChannelSpec
    from quark.torch.quantization.config.config import Config, QuantizationConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype="auto", device_map="cuda"
    )
    model.eval()

    # Per-channel INT8 weights (static, ch_axis=0 = output channels) and
    # per-token INT8 activations (dynamic, ch_axis=-1), both symmetric — this is
    # exactly the scheme QuarkW8A8Int8MoEMethod supports.
    weight_spec = Int8PerChannelSpec(
        ch_axis=0, symmetric=True, is_dynamic=False
    ).to_quantization_spec()
    act_spec = Int8PerChannelSpec(
        ch_axis=-1, symmetric=True, is_dynamic=True
    ).to_quantization_spec()

    global_cfg = QuantizationConfig(weight=weight_spec, input_tensors=act_spec)
    quant_config = Config(global_quant_config=global_cfg, exclude=list(args.exclude))

    # Calibration data (observers for the static per-channel weight scales).
    ds = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    samples = []
    for i in range(min(args.num_calib_samples, len(ds))):
        ids = tokenizer(
            ds[i]["text"],
            return_tensors="pt",
            truncation=True,
            max_length=args.seq_len,
        ).input_ids
        samples.append(ids.to("cuda"))
    dataloader = DataLoader(samples, batch_size=1)

    quantizer = ModelQuantizer(quant_config)
    with torch.no_grad():
        model = quantizer.quantize_model(model, dataloader)

    export_safetensors(
        model,
        args.output_dir,
        custom_mode="quark",
        weight_format="real_quantized",
        pack_method="reorder",
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
