# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""XPress speculator: DFlash parallel drafting + K-pass Jacobi causal refinement.

Greedy (T=0) only for now. Structure mirrors DSparkSpeculator, but the
sequential per-position Markov stage is replaced by K parallel Jacobi passes of
the XPress refiner over the whole block (K ~ 4-8, fixed -> CUDA-graph friendly).

Block convention: fill-in (sample_from_anchor=False layout), 1 + N query tokens
per request [anchor, mask_1..mask_N]; hidden row 0 is the anchor-slot hidden
(consumed by the refiner as block slot 0), rows 1..N read out the draft tokens.
This matches the z-lab DFlash checkpoints the XPress head was co-trained on.
"""

from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.cudagraph_dispatcher import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dspark.utils import load_dspark_model


logger = init_logger(__name__)


class XPressSpeculator(DFlashSpeculator):
    _speculator_name = "XPress"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        hf = self.draft_model_config.hf_config
        # fill-in block: anchor + N masked slots
        self.num_query_per_req = 1 + self.num_speculative_steps
        # K = Jacobi passes. From config.json; XPRESS_NUM_PASSES overrides it for
        # sweeps (read once at init, so the captured graph matches).
        import os
        self.num_jacobi_passes = int(
            os.environ.get("XPRESS_NUM_PASSES") or getattr(hf, "xpress_num_passes", 6)
        )
        logger.info("XPress: K=%d Jacobi passes", self.num_jacobi_passes)
        # 0 (DEFAULT): full-vocab scoring — exact reference semantics.
        # >0: OPT-IN speedup — restrict per-pass argmax to the base-logits top-C
        # (one-time w2 column gather; ~10x less readout traffic). Measured
        # AL-neutral at C=256 but NOT mathematically identical to full-vocab;
        # keep 0 for released/paper numbers.
        self.candidate_topc = int(getattr(hf, "xpress_candidate_topc", 0))
        # Single-launch Triton refine (all K passes on-chip). Latency lever for
        # small batches; needs candidate_topc > 0. Off until GPU-validated.
        self.use_fused_kernel = bool(getattr(hf, "xpress_fused_kernel", False))
        # Block slot 0 is the anchor: it is a verified token, so its refined output is
        # discarded and only its latent matters, reaching the draft slots through the
        # single mixer column L[:, k, 0]. Its own prev token therefore has a narrow
        # path to the draft, and the anchor id stands in for it -- the predecessor
        # lives in the KV cache, not in any id buffer the speculator can reach.
        self._anchor_idx = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
        )
        # XPress consumes the DRAFTER's own hidden width (the base class widens
        # self.hidden_size by hc_mult for HC-multiplexed models, which we are not).
        draft_hidden = self.draft_model_config.get_hidden_size()
        self.hidden_states = torch.zeros(
            self.max_num_tokens, draft_hidden, dtype=self.dtype, device=device
        )

    def load_draft_model(
        self,
        target_model: torch.nn.Module,
        target_attn_layer_names: set[str],
    ) -> torch.nn.Module:
        # arch comes from the draft model config ("Qwen3XPressForCausalLM");
        # the DSpark loader body is architecture-agnostic (embed/lm_head sharing,
        # non-causal attention config, KV setup) so it is reused as-is.
        return load_dspark_model(target_model, self.vllm_config)

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        # Parallel backbone forward over all query blocks, then K Jacobi passes
        # of the XPress head over each request's full block. Both stages are
        # fixed-shape and loop-free per pass -> capturable under CUDA graph.
        head_hidden = self._run_model(
            num_tokens_padded,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )
        self._jacobi_refine(num_reqs, head_hidden)

    def _jacobi_refine(self, num_reqs: int, head_hidden: torch.Tensor) -> None:
        B = self.num_query_per_req                       # 1 + N (anchor + drafts)
        n_rows = num_reqs * B
        h_full = head_hidden[:n_rows].view(num_reqs, B, -1)
        base_full = self.model.compute_draft_logits(h_full)     # [R, B, V]
        anchor_ids = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]
        tok_am1 = anchor_ids                              # see the note in __init__
        head = self.model.model.xpress_head
        # fused single-launch kernel is per-request-persistent: wins while
        # launch-latency-bound, loses to cuBLAS weight reuse at large batch
        if self.use_fused_kernel and self.candidate_topc > 0 and num_reqs <= 4:
            draft = head.jacobi_refine_greedy_fused(
                base_full, h_full, anchor_ids, tok_am1,
                self.num_jacobi_passes, self.candidate_topc,
            )
        else:
            draft = head.jacobi_refine_greedy(
                base_full, h_full, anchor_ids, tok_am1, self.num_jacobi_passes,
                candidate_topc=self.candidate_topc,
            )                                             # [R, N]
        self.draft_tokens[:num_reqs, : self.num_speculative_steps] = draft
