# SPDX-License-Identifier: Apache-2.0
"""Worker-side evidence for the reduced-model day0 NCCL experiment."""

import hashlib

import torch

from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod


class MoEReloadEvidence:
    def inspect_moe_reload(self, arm=False):
        result = {}
        for name, layer in self.model_runner.model.named_modules():
            method = getattr(layer, "quant_method", None)
            if not isinstance(method, Fp8MoEMethod):
                continue
            assert method.supports_selective_reload()
            assert method.fp8_backend.name == "FLASHINFER_CUTLASS"
            if arm:
                def forbidden(*args, **kwargs):
                    raise AssertionError("MoE PWAL or runtime reconstruction after cold load")
                method.process_weights_after_loading = forbidden
                method._prepare_moe_runtime = forbidden
                method._install_moe_kernel = forbidden
            tensors = dict(layer.named_parameters(recurse=False))
            for key in ("a1_gscale", "a2_gscale", "g1_alphas", "g2_alphas"):
                value = getattr(method.moe_quant_config, key, None)
                if value is not None:
                    tensors[f"config.{key}"] = value
            evidence = {}
            for key, value in tensors.items():
                raw = value.detach().contiguous().reshape(-1).view(torch.uint8)
                evidence[key] = {"id": id(value), "ptr": value.data_ptr(),
                                 "shape": list(value.shape), "dtype": str(value.dtype),
                                 "hash": hashlib.sha256(raw.cpu().numpy().tobytes()).hexdigest()}
            result[name] = {"backend": method.fp8_backend.name, "block": method.block_quant,
                            "method": id(method), "kernel": id(method.moe_kernel),
                            "config": id(method.moe_quant_config), "tensors": evidence}
        assert result, "No Fp8MoEMethod found"
        return result
