# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drex v1.5: a Qwen3.5 backbone plus a trained pointer head.

The backbone is a plain causal text model; the checkpoint adds `head.pt`, a
two-layer projection that scores a question's options from the hidden states at
`</opt>` and `<decide>`. The model is a pooling model: it never generates.
"""

import io
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler import Pooler
from vllm.transformers_utils.repo_utils import get_hf_file_bytes

from .adapters import as_seq_cls_model
from .drex import DrexPointerPooler
from .drex_head import PointerHead
from .interfaces_base import default_pooling_type
from .qwen3_5 import Qwen3_5ForCausalLM
from .utils import AutoWeightsLoader

HEAD_FILE = "head.pt"

if TYPE_CHECKING:
    _Base = Qwen3_5ForCausalLM
else:
    _Base = as_seq_cls_model(Qwen3_5ForCausalLM)


@default_pooling_type(seq_pooling_type="LAST")
class Qwen3_5ForDrex(_Base):
    """Qwen3.5 backbone with the Drex pointer readout."""

    def _init_pooler(self, vllm_config: VllmConfig, prefix: str = "") -> Pooler:
        model_config = vllm_config.model_config
        head_dim = int(getattr(model_config.hf_config, "drex_head_dim", 256))
        # The trained head stays fp32; the pooler casts the bf16 backbone states.
        head = PointerHead(model_config.get_hidden_size(), head_dim).to(torch.float32)
        return DrexPointerPooler(head)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # `lm_head` is replaced by a missing-output stage in pooling models.
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["lm_head."])
        loaded = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        loaded |= self._load_pointer_head()
        return loaded

    def _load_pointer_head(self) -> set[str]:
        model_config = self.vllm_config.model_config
        data = get_hf_file_bytes(HEAD_FILE, model_config.model, model_config.revision)
        if data is None:
            raise ValueError(f"Drex checkpoints require {HEAD_FILE}")
        meta = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(meta, dict) or not isinstance(meta.get("head"), dict):
            raise ValueError(f"{HEAD_FILE} must hold a 'head' state dictionary")
        if meta.get("option_isolation") or meta.get("state_bidir"):
            raise ValueError(
                "Drex on a hybrid backbone uses plain causal rows; "
                "option_isolation and state_bidir are not supported"
            )
        self.pooler.head.load_state_dict(meta["head"])
        self.pooler.head.temperature = float(meta.get("temperature", 1.0))
        return {f"pooler.head.{name}" for name in meta["head"]}
