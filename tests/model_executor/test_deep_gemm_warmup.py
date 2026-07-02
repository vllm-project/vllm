from unittest.mock import Mock

import torch

import vllm.model_executor.warmup.deep_gemm_warmup as warmup


class FakeFp8LinearMethod:
    def __init__(self, *, use_deep_gemm: bool):
        self.block_quant = True
        self.use_deep_gemm = use_deep_gemm
        self.use_marlin = False


class FakeMxfp8OnlineLinearMethod:
    pass


def test_fp8_linear_warmup_skips_when_deep_gemm_is_disabled(monkeypatch):
    module = torch.nn.Module()
    module.quant_method = FakeFp8LinearMethod(use_deep_gemm=False)
    extract_data = Mock()

    monkeypatch.setattr(warmup, "LinearBase", torch.nn.Module)
    monkeypatch.setattr(warmup, "Fp8LinearMethod", FakeFp8LinearMethod)
    monkeypatch.setattr(warmup, "Mxfp8OnlineLinearMethod", FakeMxfp8OnlineLinearMethod)
    monkeypatch.setattr(warmup, "_extract_data_from_linear_base_module", extract_data)

    assert not warmup._fp8_linear_may_use_deep_gemm(module)
    extract_data.assert_not_called()


def test_fp8_linear_warmup_allows_enabled_deep_gemm(monkeypatch):
    module = torch.nn.Module()
    module.quant_method = FakeFp8LinearMethod(use_deep_gemm=True)

    monkeypatch.setattr(warmup, "LinearBase", torch.nn.Module)
    monkeypatch.setattr(warmup, "Fp8LinearMethod", FakeFp8LinearMethod)
    monkeypatch.setattr(warmup, "Mxfp8OnlineLinearMethod", FakeMxfp8OnlineLinearMethod)
    monkeypatch.setattr(
        warmup, "get_mk_alignment_for_contiguous_layout", lambda: (128, 128)
    )
    monkeypatch.setattr(
        warmup,
        "_extract_data_from_linear_base_module",
        lambda _: (torch.empty((128, 128)), None, (128, 128)),
    )

    assert warmup._fp8_linear_may_use_deep_gemm(module)


def test_fused_moe_warmup_skips_when_quant_config_disables_deep_gemm(monkeypatch):
    class FakeQuantMethod:
        def __init__(self):
            self.quant_config = Mock(use_deep_gemm=False)
            self.get_fused_moe_quant_config = Mock()

    module = torch.nn.Module()
    module._quant_method = FakeQuantMethod()
    module.routed_experts = Mock()

    monkeypatch.setattr(warmup.envs, "VLLM_USE_DEEP_GEMM", True)
    monkeypatch.setattr(warmup.envs, "VLLM_MOE_USE_DEEP_GEMM", True)
    monkeypatch.setattr(warmup, "MoERunner", torch.nn.Module)

    assert not warmup._fused_moe_grouped_gemm_may_use_deep_gemm(module)
    module._quant_method.get_fused_moe_quant_config.assert_not_called()
