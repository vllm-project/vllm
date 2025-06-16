# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

logger = init_logger(__name__)


class XPUModelRunner(GPUModelRunner):
    """A model runner for XPU devices."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)
        # FIXME: To be verified.
        self.cascade_attn_enabled = False
        # this is XPU specific
        self.seq_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                             dtype=torch.int32,
                                             device="cpu",
                                             pin_memory=self.pin_memory)
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def _init_device_properties(self) -> None:
        pass

    def _sync_device(self) -> None:
        torch.xpu.synchronize()

    def _prepare_inputs(
        self, scheduler_output: "SchedulerOutput"
    ) -> tuple[dict[str, Any], torch.Tensor, Optional[SpecDecodeMetadata]]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        # ======== XPU start =========
        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=self.seq_start_loc_np[1:num_reqs + 1])
        # ======== XPU end =========
        return super()._prepare_inputs(scheduler_output)
