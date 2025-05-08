# SPDX-License-Identifier: Apache-2.0
import gc
import weakref
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from vllm.attention import get_attn_backend
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, LayerBlockType, cdiv,
                        check_use_alibi, is_pin_memory_available)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.sliding_window = model_config.get_sliding_window()
        self.interleaved_sliding_window = getattr(
            model_config.hf_text_config, "interleaved_sliding_window", None)
        self.window_size = (self.sliding_window
                            or self.interleaved_sliding_window)

        self.is_multimodal_model = model_config.is_multimodal_model
        self.block_size = cache_config.block_size
        self.max_model_len = model_config.max_model_len
        self.max_num_blocks_per_req = cdiv(self.max_model_len, self.block_size)
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_attn_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.head_size = model_config.get_head_size()
        self.hidden_size = model_config.get_hidden_size()
        self.attention_chunk_size = model_config.attention_chunk_size

        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        if self.attn_backend is None:
            error_msg = (
                f"Error with get_att_backend: {self.head_size=}, "
                f"{self.dtype=}, {self.kv_cache_dtype=}, {self.block_size=}, "
                f"{self.model_config.is_attention_free=}, "
                f"{self.model_config.use_mla=}")
            logger.error(error_msg)
            raise NotImplementedError(
                "Non-Attention backend is not supported by V1 GPUModelRunner.")
        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            weakref.proxy(self))
        self.cascade_attn_enabled = False

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        # FIXME: support mrope
        self.uses_mrope = False

        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=model_config,
            scheduler_config=scheduler_config,
            mm_registry=self.mm_registry,
        )
        self.max_num_encoder_input_tokens = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size

        # Sampler
        self.sampler = Sampler()

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: list[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: dict[str, dict[int, torch.Tensor]] = {}

        # Set up speculative decoding.
        self.use_spec_decode = False
        self.use_aux_hidden_state_outputs = False
        if self.speculative_config:
            self.use_spec_decode = True
            if get_pp_group().is_last_rank:
                if self.speculative_config.method == "ngram":
                    self.drafter = NgramProposer(self.vllm_config)
                elif self.speculative_config.use_eagle():
                    self.drafter = EagleProposer(self.vllm_config,
                                                 self.device)  # type: ignore
                    if self.speculative_config.method == "eagle3":
                        self.use_aux_hidden_state_outputs = True
                else:
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")
                self.rejection_sampler = RejectionSampler()

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # Persistent batch.
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_blocks_per_req=self.max_num_blocks_per_req,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
        )

        self.use_cuda_graph = (self.vllm_config.compilation_config.level
                               == CompilationLevel.PIECEWISE
                               and not self.model_config.enforce_eager)
        # TODO(woosuk): Provide an option to tune the max cudagraph batch size.
        # The convention is different.
        # self.cudagraph_batch_sizes sorts in ascending order.
        # The batch sizes in the config are in descending order.
        self.cudagraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # Persistent buffers for CUDA graphs.
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=self.pin_memory)

        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = check_use_alibi(model_config)

        self.inputs_embeds = torch.zeros(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.dtype,
            device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.input_ids_np = self.input_ids_cpu.numpy()
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.positions_np = self.positions_cpu.numpy()
        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=self.pin_memory)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=self.pin_memory)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()
        # this is XPU specific
        self.seq_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                             dtype=torch.int32,
                                             device="cpu",
                                             pin_memory=self.pin_memory)
        self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

    # we can enable this if GPUModelRunner or parent class don't have
    # torch.cuda in the future
    # def __init__(self,
    #     vllm_config: VllmConfig,
    #     device: torch.device,):
    #     super().__init__(vllm_config, device)
    #     self.cascade_attn_enabled = False
    #     # FIXME: support mrope
    #     self.uses_mrope = False
    #     # this is XPU specific
    #     self.seq_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
    #                                          dtype=torch.int32,
    #                                          device="cpu",
    #                                          pin_memory=self.pin_memory)
    #     self.seq_start_loc_np = self.seq_start_loc_cpu.numpy()

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        # ======== XPU  start =========
        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        self.seq_start_loc_np[0] = 0
        np.cumsum(seq_lens, out=self.seq_start_loc_np[1:num_reqs + 1])
        # ======== XPU end =========
        return super()._prepare_inputs(scheduler_output)

    def profile_run(self) -> None:
        # Trigger compilation for general shape.
        hidden_states = self._dummy_run(self.max_num_tokens)
        logits = self.model.compute_logits(hidden_states, None)
        logits = logits[:self.max_num_tokens]
        torch.xpu.synchronize()
        gc.collect()
