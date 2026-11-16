# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_v2 import (
    DeepEPV2PrepareAndFinalize,
)


class MooncakePrepareAndFinalize(DeepEPV2PrepareAndFinalize):
    """Prepare/Finalize using Mooncake EP's ElasticBuffer.

    Mooncake's ElasticBuffer follows the DeepEP v2 API, so both modes are
    inherited from DeepEPV2PrepareAndFinalize. It differs in two places:

    - Expanded dispatch (prefill mode) still returns a recv_topk_idx tensor,
      which the kernels do not fill in that mode; it is dropped so the ids
      are rebuilt from the per-expert counts, as for DeepEP.
    - combine() has no allocate_on_comm_stream argument; its output is
      allocated on the compute stream.
    """

    def _receiver(self, event, has_scales, recv_x, recv_topk_idx, *args, **kwargs):
        if not self.use_cudagraph:
            recv_topk_idx = None
        return super()._receiver(
            event, has_scales, recv_x, recv_topk_idx, *args, **kwargs
        )

    def _combine(self, x, handle, combine_async):
        return self.buffer.combine(
            x=x,
            handle=handle,
            topk_weights=None,
            async_with_compute_stream=combine_async,
        )
