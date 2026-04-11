# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types


def test_rocm_gcn_arch_fallback_is_lazy(monkeypatch):
    import torch
    import vllm.platforms.rocm as rocm

    fallback_calls = 0

    def fake_get_device_properties(device):
        nonlocal fallback_calls
        fallback_calls += 1
        return types.SimpleNamespace(gcnArchName="gfx950")

    fake_amdsmi = types.ModuleType("amdsmi")
    fake_amdsmi.AmdSmiException = RuntimeError
    fake_amdsmi.amdsmi_get_gpu_asic_info = lambda handle: {}
    fake_amdsmi.amdsmi_get_gpu_device_uuid = lambda handle: "fake-uuid"
    fake_amdsmi.amdsmi_get_processor_handles = lambda: []
    fake_amdsmi.amdsmi_init = lambda: None
    fake_amdsmi.amdsmi_shut_down = lambda: None
    fake_amdsmi.amdsmi_topo_get_link_type = (
        lambda handle, peer: {"hops": 1, "type": 2}
    )

    with monkeypatch.context() as m:
        m.setitem(sys.modules, "amdsmi", fake_amdsmi)
        m.setattr(torch.cuda, "get_device_properties", fake_get_device_properties)

        rocm = importlib.reload(rocm)
        assert fallback_calls == 0

        assert rocm.on_gfx950()
        assert fallback_calls == 1

        rocm._get_gcn_arch.cache_clear()
        assert rocm.RocmPlatform.supports_fp8()
        assert fallback_calls == 2

    importlib.reload(rocm)
