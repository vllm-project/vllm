# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct reload on a toy model whose post-processing changes layouts.

Post-processing transposes one weight in place (only its stride changes, the
matrix is square), shrinks another into a new, smaller tensor and pads a third
into a new, larger one. After each update the model must equal a cold start on
that checkpoint, with every tensor at its cold-start address.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import finish_reload, start_reload
from vllm.model_executor.model_loader.reload.direct import DirectReloadError
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.utils import replace_parameter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

MODEL_CONFIG = SimpleNamespace(
    dtype=torch.float32, quantization=None, word_embeddings_untied_by_checkpoint=True
)


class _Method(QuantizeMethodBase):
    def create_weights(self, *args, **kwargs):
        pass

    def apply(self, *args, **kwargs):
        pass


class Transpose(_Method):
    def process_weights_after_loading(self, layer):
        replace_parameter(layer, "weight", layer.weight.t())


class DropRowAndDouble(_Method):
    def process_weights_after_loading(self, layer):
        replace_parameter(layer, "weight", layer.weight[:-1] * 2)


class Pad(_Method):
    def process_weights_after_loading(self, layer):
        w = layer.weight
        replace_parameter(layer, "weight", torch.cat([w, w.new_zeros(1, w.shape[1])]))


class Layer(nn.Module):
    def __init__(self, quant_method, *shape):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(*shape, device="cuda"), False)
        self.quant_method = quant_method


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = Layer(Transpose(), 4, 4)
        self.b = Layer(DropRowAndDouble(), 3, 5)
        self.c = Layer(Pad(), 2, 3)
        self.norm = nn.Parameter(torch.empty(5, device="cuda"), False)

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, tensor in weights:
            param = params[name]
            getattr(param, "weight_loader", default_weight_loader)(param, tensor)


def checkpoint(seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "a.weight": torch.randn(4, 4, generator=g),
        "b.weight": torch.randn(3, 5, generator=g),
        "c.weight": torch.randn(2, 3, generator=g),
        "norm": torch.randn(5, generator=g),
    }


def assert_serves(model: Model, ckpt: dict[str, torch.Tensor]) -> None:
    torch.testing.assert_close(model.a.weight.cpu(), ckpt["a.weight"].t())
    torch.testing.assert_close(model.b.weight.cpu(), ckpt["b.weight"][:-1] * 2)
    torch.testing.assert_close(model.c.weight[:2].cpu(), ckpt["c.weight"])
    assert model.c.weight[2:].eq(0).all()
    torch.testing.assert_close(model.norm.cpu(), ckpt["norm"])


def addresses(model: nn.Module):
    return {n: (p.data_ptr(), p.shape, p.stride()) for n, p in model.named_parameters()}


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv("VLLM_WEIGHT_RELOAD_MODE", "direct")
    model = Model()
    model.load_weights(checkpoint(0).items())
    process_weights_after_loading(model, MODEL_CONFIG, torch.device("cuda"))
    return model


def update(model: Model, weights) -> None:
    start_reload(model)
    model.load_weights(weights)
    finish_reload(model, MODEL_CONFIG)


def test_updates_match_cold_start_at_cold_start_addresses(model):
    cold = addresses(model)
    for seed in (1, 1, 0):
        update(model, checkpoint(seed).items())
        assert_serves(model, checkpoint(seed))
        assert addresses(model) == cold


def test_missing_write_fails_and_refuses_later_updates(model):
    weights = [(n, t) for n, t in checkpoint(1).items() if n != "norm"]
    with pytest.raises(DirectReloadError, match="never written.*norm"):
        update(model, weights)
    with pytest.raises(DirectReloadError, match="Restart the engine"):
        start_reload(model)


def test_replaced_parameter_is_refused_before_any_write(model):
    model.norm = nn.Parameter(model.norm.data.clone(), False)
    with pytest.raises(DirectReloadError, match="norm was replaced"):
        start_reload(model)


def test_result_unlike_cold_start_fails(model):
    model.b.quant_method = Pad()
    with pytest.raises(DirectReloadError, match="moved"):
        update(model, checkpoint(1).items())
