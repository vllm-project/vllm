# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.v1.outputs import ModelRunnerOutput, PoolerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput as GpuAsyncOutput
from vllm.v1.worker.gpu.async_utils import AsyncPoolingOutput as GpuAsyncPoolingOutput
from vllm.v1.worker.gpu.sample.output import SamplerOutput

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.worker import PendingAuxOutput


class _NoopEvent:
    """Stands in for the copy event there is nothing to wait on."""

    def synchronize(self) -> None:
        pass


_NOOP_EVENT = _NoopEvent()


class AsyncOutput(GpuAsyncOutput):
    """CPU stand-in that fills the same fields without the copy stream.

    The stream, the event and the queue-then-read split exist to overlap a
    device-to-host copy. There is no such copy here, so the fields are read
    straight off and `get_output` is inherited unchanged.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool,
        pending_aux_output: "PendingAuxOutput | None",
    ):
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.pending_aux_output = pending_aux_output
        self.copy_event = _NOOP_EVENT
        self._has_fault: torch.Tensor | None = None

        self.sampled_token_ids = sampler_output.sampled_token_ids.numpy()
        self.logprobs_tensors = sampler_output.logprobs_tensors
        self.num_nans: np.ndarray | None = None
        if sampler_output.num_nans is not None:
            self.num_nans = sampler_output.num_nans.numpy()
        self.num_sampled_tokens_np = num_sampled_tokens.numpy()
        self.sampling_mask_tensors = sampler_output.sampling_mask_tensors
        self.prompt_logprobs_dict = dict(model_runner_output.prompt_logprobs_dict)
        self.prompt_token_id_logprobs_dict = dict(
            model_runner_output.prompt_token_id_logprobs_dict
        )
        if self.pending_aux_output is not None:
            self.pending_aux_output.enqueue_cpu_copy(
                num_sampled=self.num_sampled_tokens_np,
                num_rejected=sampler_output.num_rejected.numpy(),
            )
        if check_ep_fault:
            self._has_fault = get_ep_all2all_manager().query_fault()


class AsyncPoolingOutput(GpuAsyncPoolingOutput):
    """CPU stand-in for the pooling path, same reasoning as `AsyncOutput`."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        self.copy_event = _NOOP_EVENT

        if isinstance(pooler_output, torch.Tensor) and all(finished_mask):
            self.pooler_output_cpu: PoolerOutput = pooler_output
        else:
            outputs = (
                pooler_output.unbind()
                if isinstance(pooler_output, torch.Tensor)
                else pooler_output
            )
            self.pooler_output_cpu = [
                None if output is None or not is_finished else output
                for output, is_finished in zip(outputs, finished_mask, strict=True)
            ]
