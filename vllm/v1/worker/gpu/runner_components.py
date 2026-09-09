# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
V2RunnerComponents — platform-supplied component bundle for GPUModelRunner V2.

Out-of-tree platforms override Platform.get_v2_runner_components() to return a
customised bundle.  The runner resolves the bundle exactly once during __init__
and stores it, so no factory calls appear on hot paths.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.pcp_manager import PCPManager
    from vllm.v1.worker.gpu.sample.sampler import Sampler
    from vllm.v1.worker.gpu.states import RequestState


@dataclass(frozen=True)
class V2RunnerComponents:
    """Bundle of class objects used by GPUModelRunner V2.

    Each field is a *class* (not an instance).  The runner instantiates them
    using the arguments appropriate for each component.

    Platforms that extend GPUModelRunner V2 override
    ``Platform.get_v2_runner_components()`` to supply custom subclasses.  All
    fields must remain compatible with the corresponding base-class constructor
    signatures unless the platform also subclasses the runner itself.
    """

    request_state_cls: "type[RequestState]"
    input_buffers_cls: "type[InputBuffers]"
    input_batch_cls: "type[InputBatch]"
    cudagraph_manager_cls: "type[ModelCudaGraphManager]"
    sampler_cls: "type[Sampler]"
    pcp_manager_cls: "type[PCPManager]"
