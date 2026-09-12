#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal OOB repro against the shipped compute_global_topk_indices_and_lens.

Proven invalid condition:
  block_table.rows == 1
  token_to_req_indices includes 1

Patched kernel must: no illegal address, token1 all -1 / lens 0, token0 valid.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
from trackb_import_stubs import install_trackb_import_stubs  # noqa: E402

install_trackb_import_stubs()

BLOCK_SIZE = 64


def _import_compute(version: str):
    import importlib
    import importlib.util
    import types

    module_name = f"vllm.models.{version}.common.ops.cache_utils"
    try:
        return importlib.import_module(module_name).compute_global_topk_indices_and_lens
    except Exception:
        import vllm

        root = Path(vllm.__file__).resolve().parent / "models"
        version_root = root / version

        def ensure_ns(fullname, path):
            if fullname in sys.modules and hasattr(sys.modules[fullname], "__path__"):
                return
            m = types.ModuleType(fullname)
            m.__path__ = [str(path)]
            m.__file__ = str(path / "__init__.py")
            m.__package__ = fullname
            sys.modules[fullname] = m

        ensure_ns("vllm.models", root)
        ensure_ns(f"vllm.models.{version}", version_root)
        ensure_ns(f"vllm.models.{version}.common", version_root / "common")
        ensure_ns(f"vllm.models.{version}.common.ops", version_root / "common" / "ops")
        path = version_root / "common" / "ops" / "cache_utils.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.compute_global_topk_indices_and_lens


def make_tensors(device: torch.device):
    topk_indices = torch.tensor(
        [[0, -1, -1, -1], [0, -1, -1, -1]], dtype=torch.int32, device=device
    )
    token_to_req = torch.tensor([0, 1], dtype=torch.int32, device=device)
    block_table = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32, device=device)
    is_valid = torch.tensor([True, True], dtype=torch.bool, device=device)
    assert block_table.shape[0] == 1
    assert int(token_to_req.max().item()) == 1
    return topk_indices, token_to_req, block_table, is_valid


def main() -> int:
    if not torch.cuda.is_available():
        print(json.dumps({"GPU_REPRO_PATCHED_RESULT": "SKIP_NO_CUDA"}))
        return 2
    device = torch.device("cuda")
    report: dict = {
        "device": torch.cuda.get_device_name(0),
        "CUDA_ILLEGAL_ADDRESS_AFTER_PATCH": False,
    }
    for version in ("deepseek_v4", "deepseek_v4_1"):
        key = f"{version}"
        try:
            compute = _import_compute(version)
            topk, tok, bt, valid = make_tensors(device)
            out, lens = compute(topk, tok, bt, BLOCK_SIZE, valid)
            torch.cuda.synchronize()
            report[key] = {
                "out": out.tolist(),
                "lens": lens.tolist(),
                "token0_slot": int(out[0, 0].item()),
                "token1_all_neg1": all(v == -1 for v in out[1].tolist()),
                "token1_lens0": int(lens[1].item()) == 0,
                "no_alias_to_row0": int(out[1, 0].item()) != 10 * BLOCK_SIZE,
                "PASS": (
                    int(out[0, 0].item()) == 10 * BLOCK_SIZE
                    and all(v == -1 for v in out[1].tolist())
                    and int(lens[1].item()) == 0
                    and int(lens[0].item()) == 1
                ),
            }
        except Exception as e:
            err = str(e)
            illegal = "illegal" in err.lower() or "IllegalAddress" in err
            report["CUDA_ILLEGAL_ADDRESS_AFTER_PATCH"] = (
                report["CUDA_ILLEGAL_ADDRESS_AFTER_PATCH"] or illegal
            )
            report[key] = {
                "PASS": False,
                "error": err,
                "traceback": traceback.format_exc(),
            }
    report["GPU_REPRO_PATCHED_RESULT"] = (
        "PASS"
        if all(report.get(v, {}).get("PASS") for v in ("deepseek_v4", "deepseek_v4_1"))
        else "FAIL"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["GPU_REPRO_PATCHED_RESULT"] == "PASS" else 1


if __name__ == "__main__":
    # Prefer a disposable GPU when the caller has not pinned one.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    raise SystemExit(main())
