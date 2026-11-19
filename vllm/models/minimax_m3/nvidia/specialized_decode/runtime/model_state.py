# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model state of the specialized decode.

`MiniMaxM3SpecializedDecodeModelState` is the default model state (which
MiniMax-M3 runs on without the specialized decode) plus the decode executor,
built and warmed up once the weights are loaded: the model runner creates the
model state after loading the target and draft models, and before memory
profiling and CUDA graph capture.

The attention metadata builds still run eagerly before every step, as in the
stock decode. Recording them into CUDA graphs (as DeepSeek-V4.1's
`metadata_graphs` does) needs the MiniMax-M3 sparse and indexer builders
audited for host reads first, e.g. the CUTLASS sparse decode's plan cache.
"""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from .. import MAX_REQUESTS, MAX_SPECULATIVE_TOKENS
from ..peer_logits import peer_logits, peer_logits_unsupported
from .warmup import warm_up

logger = init_logger(__name__)


class MiniMaxM3SpecializedDecodeModelState(DefaultModelState):
    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ):
        super().__init__(vllm_config, model, encoder_cache, device)
        language_model = (
            model.get_language_model()
            if hasattr(model, "get_language_model")
            else model
        )
        self.decode = language_model.build_specialized_decode()
        warm_up(self.decode, vllm_config)
        self._peer_logits = None
        reason = peer_logits_unsupported(
            language_model.lm_head, language_model.logits_processor
        )
        if reason:
            logger.info("Target logits stay gathered: %s.", reason)
        else:
            self._peer_logits = peer_logits(
                language_model.lm_head,
                language_model.logits_processor,
                MAX_REQUESTS * (1 + MAX_SPECULATIVE_TOKENS),
                "target_logits",
            )
            logger.info("Target logits: through peer memory.")
        # Peer buffers, their handshake and kernel compiles now, on every rank in the
        # same order and before memory profiling: the target's, then the DSpark
        # drafter's.
        if self._peer_logits is not None:
            self._peer_logits.warm_up()
        spec = vllm_config.speculative_config
        if spec is not None and spec.num_speculative_tokens:
            from ..drafter import warm_up_drafters

            warm_up_drafters(int(spec.num_speculative_tokens))

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The same on every TP rank: every rank samples every request.
        if self._peer_logits is not None:
            logits = self._peer_logits(hidden_states)
            if logits is not None:
                return logits
        return super().compute_sample_logits(hidden_states)
