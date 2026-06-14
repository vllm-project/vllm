# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import VllmConfig
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.spec_decode.utils import eagle_step_update_slot_mapping_and_metadata


class EagleProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        super().__init__(
            vllm_config,
            device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

    def dry_run_helper_kernels(self) -> None:
        # Base warms shared per-padded-batch kernels and the
        # copy_and_expand_eagle_inputs_kernel (when allocated).
        super().dry_run_helper_kernels()

        # ``eagle_step_update_slot_mapping_and_metadata`` fires inside the
        # per-step inner loop of ``propose()`` only for sequential Eagle
        # with ``num_speculative_tokens > 1``. DFlash short-circuits the
        # loop via ``parallel_drafting`` and single-token Eagle never
        # enters it.
        if self.num_speculative_tokens <= 1 or self.parallel_drafting:
            return
        if self.block_size <= 0:
            return

        device = self.device
        num_reqs = 1
        block_size = self.block_size
        max_model_len = self.max_model_len
        n_blocks_per_req = (max_model_len + block_size - 1) // block_size

        positions_1d = torch.zeros(num_reqs, dtype=torch.int64, device=device)
        block_table = torch.zeros(
            (num_reqs, n_blocks_per_req), dtype=torch.int32, device=device
        )
        seq_lens = torch.ones(num_reqs, dtype=torch.int32, device=device)
        out_clamped_positions = torch.empty(num_reqs, dtype=torch.int64, device=device)
        out_slot_mapping = torch.empty(num_reqs, dtype=torch.int64, device=device)
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions_1d,
            block_table_tensor=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            max_model_len=max_model_len,
            out_clamped_positions=out_clamped_positions,
            out_slot_mapping=out_slot_mapping,
        )
