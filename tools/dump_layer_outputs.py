#!/usr/bin/env python3
"""Non-invasive layer output dumper for Llama models in vLLM.

Captures intermediate tensor outputs from each layer during inference
using PyTorch forward hooks. No vLLM source code modification required.

Usage:
    python tools/dump_layer_outputs.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --backend cuda \
        --prompt "Hello, world!" \
        --max-tokens 5 \
        --output-dir ./layer_dumps
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


class HookManager:
    """Manages forward hooks on model layers and captures tensor outputs."""

    def __init__(self, dump_mode: str = "coarse", layers: list[int] | None = None):
        self.dump_mode = dump_mode
        self.target_layers = layers
        self.captured: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        self.handles: list[torch.utils.hooks.RemovableHook] = []
        self._call_counts: dict[str, int] = defaultdict(int)

    def _get_step_name(self, module_name: str, input_tensor: torch.Tensor) -> str:
        """Determine step name (prefill vs decode) based on call count."""
        self._call_counts[module_name] += 1
        count = self._call_counts[module_name]
        if count == 1:
            return "prefill"
        return f"decode_step_{count - 1:03d}"

    def _make_hook(self, layer_name: str, tensor_names: list[str]):
        """Create a forward hook closure for the given layer."""

        def hook_fn(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                first_input = input[0] if not isinstance(input[0], torch.Tensor) else input[0]
                if isinstance(first_input, torch.Tensor):
                    ref_tensor = first_input
                else:
                    ref_tensor = input[1] if len(input) > 1 else None
            else:
                ref_tensor = None

            step = self._get_step_name(layer_name, ref_tensor)
            key_prefix = f"{step}/{layer_name}"

            if isinstance(output, tuple):
                for i, name in enumerate(tensor_names):
                    if i < len(output) and output[i] is not None:
                        self.captured[key_prefix][name] = (
                            output[i].detach().clone().cpu()
                        )
            elif isinstance(output, torch.Tensor):
                name = tensor_names[0] if tensor_names else "output"
                self.captured[key_prefix][name] = (
                    output.detach().clone().cpu()
                )

        return hook_fn

    def register_hooks(self, model: torch.nn.Module):
        """Register forward hooks on target layers of the model."""
        # Navigate to the inner LlamaModel
        if hasattr(model, "model"):
            llama_model = model.model
        else:
            llama_model = model

        # Hook on embed_tokens
        if hasattr(llama_model, "embed_tokens") and not isinstance(
            llama_model.embed_tokens, torch.nn.Identity
        ):
            h = llama_model.embed_tokens.register_forward_hook(
                self._make_hook("embed", ["output"])
            )
            self.handles.append(h)

        # Hook on each decoder layer
        if hasattr(llama_model, "layers"):
            for idx, layer in enumerate(llama_model.layers):
                if layer is None:
                    continue
                if self.target_layers is not None and idx not in self.target_layers:
                    continue

                layer_name = f"layer_{idx:02d}"
                h = layer.register_forward_hook(
                    self._make_hook(layer_name, ["hidden_states", "residual"])
                )
                self.handles.append(h)

                if self.dump_mode == "fine":
                    if hasattr(layer, "self_attn"):
                        h = layer.self_attn.register_forward_hook(
                            self._make_hook(
                                f"{layer_name}/self_attn", ["attn_output"]
                            )
                        )
                        self.handles.append(h)
                    if hasattr(layer, "mlp"):
                        h = layer.mlp.register_forward_hook(
                            self._make_hook(f"{layer_name}/mlp", ["mlp_output"])
                        )
                        self.handles.append(h)

        # Hook on final norm
        if hasattr(llama_model, "norm") and not isinstance(
            llama_model.norm, torch.nn.Identity
        ):
            h = llama_model.norm.register_forward_hook(
                self._make_hook("norm", ["output"])
            )
            self.handles.append(h)

        # Hook on lm_head
        if hasattr(model, "lm_head") and not isinstance(
            model.lm_head, torch.nn.Identity
        ):
            h = model.lm_head.register_forward_hook(
                self._make_hook("lm_head", ["logits"])
            )
            self.handles.append(h)

        print(f"Registered {len(self.handles)} hooks (mode={self.dump_mode})")

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def save(self, output_dir: Path, backend: str) -> dict:
        """Save captured tensors to disk and return manifest entries."""
        steps_manifest = {}

        for key_prefix, tensors in sorted(self.captured.items()):
            # key_prefix = "prefill/layer_00" or "decode_step_001/layer_00"
            parts = key_prefix.split("/", 1)
            step_name = parts[0]
            layer_path = parts[1] if len(parts) > 1 else ""

            dir_path = output_dir / backend / step_name / layer_path
            dir_path.mkdir(parents=True, exist_ok=True)

            if step_name not in steps_manifest:
                steps_manifest[step_name] = {}

            layer_manifest = {}
            for tensor_name, tensor in tensors.items():
                file_name = f"{tensor_name}.pt"
                file_path = dir_path / file_name
                torch.save(tensor, file_path)

                rel_path = str(
                    (Path(backend) / step_name / layer_path / file_name)
                )
                layer_manifest[tensor_name] = rel_path

            # Nest into steps_manifest
            if layer_path:
                if "layers" not in steps_manifest[step_name]:
                    steps_manifest[step_name]["layers"] = {}
                steps_manifest[step_name]["layers"][layer_path] = layer_manifest
            else:
                steps_manifest[step_name].update(layer_manifest)

        return steps_manifest


def get_model_from_engine(llm) -> torch.nn.Module:
    """Extract the nn.Module from the vLLM LLM engine internals."""
    engine = llm.llm_engine
    executor = engine.model_executor
    if hasattr(executor, "driver_worker"):
        worker = executor.driver_worker
    else:
        worker = executor.workers[0] if hasattr(executor, "workers") else None

    if worker is None:
        raise RuntimeError("Cannot locate worker from model_executor")

    runner = worker.model_runner
    model = runner.model
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Dump intermediate layer outputs from Llama model in vLLM"
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--backend", required=True, help="Backend identifier for directory naming"
    )
    parser.add_argument("--prompt", default="Hello, world!", help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=5, help="Max tokens to generate"
    )
    parser.add_argument(
        "--output-dir", default="./layer_dumps", help="Output root directory"
    )
    parser.add_argument(
        "--dump-mode",
        choices=["coarse", "fine"],
        default="coarse",
        help="coarse: decoder layer level; fine: include attn/mlp sublayers",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Comma-separated layer indices to dump (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-model-len", type=int, default=None, help="Max model context length"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--dtype", default="auto", help="Model dtype (auto, float16, bfloat16)"
    )
    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = [int(x.strip()) for x in args.layers.split(",")]

    # Lazy import vLLM to allow --help without GPU
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM with model={args.model}, enforce_eager=True")
    llm_kwargs = dict(
        model=args.model,
        enforce_eager=True,
        seed=args.seed,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    # Get internal model
    model = get_model_from_engine(llm)
    print(f"Model type: {type(model).__name__}")

    # Register hooks
    hook_manager = HookManager(dump_mode=args.dump_mode, layers=target_layers)
    hook_manager.register_hooks(model)

    # Run inference
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0
    )
    print(f"Running inference: prompt={args.prompt!r}, max_tokens={args.max_tokens}")
    outputs = llm.generate([args.prompt], sampling_params)

    generated_text = outputs[0].outputs[0].text if outputs else ""
    print(f"Generated: {generated_text!r}")

    # Save tensors
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_manifest = hook_manager.save(output_dir, args.backend)

    # Write manifest
    manifest = {
        "model": args.model,
        "backend": args.backend,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "tp_size": args.tp_size,
        "dump_mode": args.dump_mode,
        "layers_dumped": target_layers or "all",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": steps_manifest,
    }

    manifest_path = output_dir / f"{args.backend}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest to {manifest_path}")
    print(
        f"Dumped {len(hook_manager.captured)} layer/step combinations "
        f"to {output_dir / args.backend}/"
    )

    # Cleanup
    hook_manager.remove_hooks()


if __name__ == "__main__":
    main()
