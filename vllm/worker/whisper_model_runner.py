import gc
import time
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from vllm.attention import AttentionMetadata
from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig,
                         ModelConfig, ParallelConfig, SchedulerConfig,
                         VisionLanguageConfig, WhisperConfig)
from vllm.distributed import broadcast_tensor_dict
from vllm.distributed.parallel_state import graph_capture
from vllm.model_executor import SamplingMetadata
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.model_runner import ModelRunner

logger = init_logger(__name__)

_PAD_SLOT_ID = -1
LORA_WARMUP_RANK = 8
_BATCH_SIZE_ALIGNMENT = 8
_BATCH_SIZES_TO_CAPTURE = [1, 2, 4] + [
    _BATCH_SIZE_ALIGNMENT * i for i in range(1, 33)
]
_NUM_WARMUP_ITERS = 2

class WhisperModelRunner(ModelRunner):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        cache_config: CacheConfig,
        load_config: LoadConfig,
        lora_config: Optional[LoRAConfig],
        kv_cache_dtype: Optional[str] = "auto",
        is_driver_worker: bool = False,
        vision_language_config: Optional[VisionLanguageConfig] = None,
        whisper_config: Optional[WhisperConfig] = None,
    ):
        super().__init__(model_config,
                         parallel_config,
                         scheduler_config,
                         device_config,
                         cache_config,
                         load_config,
                         lora_config=lora_config,
                         kv_cache_dtype=kv_cache_dtype,
                         is_driver_worker=is_driver_worker,
                         vision_language_config=vision_language_config,
                         whisper_config=whisper_config)

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
        kv_caches: List[torch.Tensor],
    ) -> Optional[SamplerOutput]:
        (whisper_data, input_tokens, input_positions, attn_metadata, sampling_metadata,
         lora_requests, lora_mapping, multi_modal_kwargs
         ) = self.prepare_input_tensors(seq_group_metadata_list)

        if self.lora_config:
            self.set_active_loras(lora_requests, lora_mapping)

        # Currently cuda graph is only supported by the decode phase.
        prefill_meta = attn_metadata.prefill_metadata
        decode_meta = attn_metadata.decode_metadata
        if prefill_meta is None and decode_meta.use_cuda_graph:
            graph_batch_size = input_tokens.shape[0]
            model_executable = self.graph_runners[graph_batch_size]
        else:
            model_executable = self.model

        model_executable = self.model

        hidden_states = model_executable(
            whisper_data=whisper_data,
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            **multi_modal_kwargs,
        )

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, sampling_metadata)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return None

        output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        return output

    def _prepare_encoder_model_input(
            self, seq_group_metadata_list: List[SequenceGroupMetadata],
            attn_metadata: AttentionMetadata):
        
        whisper_data_list = []

        for seq_group_metadata in seq_group_metadata_list:
            seq_ids = list(seq_group_metadata.seq_data.keys())
            is_prompt = seq_group_metadata.is_prompt

            for seq_id in seq_ids:

                whisper_data = seq_group_metadata.whisper_data
                if whisper_data is not None:
                    # Process multi-modal data
                    if self.whisper_processor is None:
                        raise ValueError("Whisper Processor not initialized")
                    
                    if len(whisper_data.shape) == 1:
                        whisper_data = self.whisper_processor(
                            whisper_data, 
                            sampling_rate = self.whisper_config.sample_rate,
                            return_tensors = 'pt',
                        )
                        whisper_data = whisper_data.to(self.model_config.dtype).input_features[0]
            
                    whisper_data_list.append(whisper_data)
        
        whisper_data = torch.cat(whisper_data_list, dim = 1).cuda()

        return whisper_data

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: Optional[List[SequenceGroupMetadata]],
    ) -> Tuple[torch.Tensor, torch.Tensor, AttentionMetadata,
               Set[LoRARequest], LoRAMapping, Dict[str, torch.Tensor]]:
        if self.is_driver_worker:
            assert seq_group_metadata_list is not None
            # Prepare input tensors.
            (
                input_tokens,
                input_positions,
                attn_metadata,
                seq_lens,
                query_lens,
                lora_mapping,
                lora_requests,
                multi_modal_kwargs,
                slot_mapping,
                num_prefill_tokens,
                num_decode_tokens,
                num_prefills,
            ) = self._prepare_model_input(seq_group_metadata_list)
            whisper_data = self._prepare_encoder_model_input(seq_group_metadata_list, attn_metadata)
            sampling_metadata = SamplingMetadata.prepare(
                seq_group_metadata_list, seq_lens, query_lens, self.device,
                self.pin_memory)

            metadata_dict = {
                'whisper_data': whisper_data,
                "input_tokens": input_tokens,
                "input_positions": input_positions,
                "lora_requests": lora_requests,
                "lora_mapping": lora_mapping,
                "multi_modal_kwargs": multi_modal_kwargs,
                "num_prefill_tokens": num_prefill_tokens,
                "num_decode_tokens": num_decode_tokens,
                "slot_mapping": slot_mapping,
                "num_prefills": num_prefills,
            }
            if attn_metadata:
                metadata_dict.update(attn_metadata.asdict_zerocopy())
            broadcast_tensor_dict(metadata_dict, src=0)
        else:
            metadata_dict = broadcast_tensor_dict(src=0)
            whisper_data = metadata_dict.pop("whisper_data")
            input_tokens = metadata_dict.pop("input_tokens")
            input_positions = metadata_dict.pop("input_positions")
            lora_mapping = metadata_dict.pop("lora_mapping")
            lora_requests = metadata_dict.pop("lora_requests")
            multi_modal_kwargs = metadata_dict.pop("multi_modal_kwargs")
            if metadata_dict:
                attn_metadata = self.attn_backend.make_metadata(
                    **metadata_dict)
            else:
                attn_metadata = None
            sampling_metadata = SamplingMetadata(
                seq_groups=None,
                selected_token_indices=selected_token_indices,
                categorized_sample_indices=None,
                num_prompts=0,
            )

        return (whisper_data, input_tokens, input_positions, attn_metadata,
                sampling_metadata, lora_requests, lora_mapping,
                multi_modal_kwargs)


    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_local_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for vision encoding, which needs
        # to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.
        model_config = self.model_config
        vlm_config = self.vision_language_config
        whisper_config = self.whisper_config

        if vlm_config:
            max_num_seqs = min(
                max_num_seqs,
                int(max_num_batched_tokens / vlm_config.image_feature_size))
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            
            whisper_data = None
            if vlm_config is not None:
                seq_data, dummy_multi_modal_data = MULTIMODAL_REGISTRY \
                    .dummy_data_for_profiling(seq_len, model_config, vlm_config)
            else:
                seq_data = SequenceData([0] * seq_len)
                dummy_multi_modal_data = None

                if whisper_config is not None:
                    whisper_data = torch.zeros(
                        (30 * whisper_config.sample_rate), 
                        dtype=torch.float16)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_multi_modal_data,
                whisper_data=whisper_data
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        self.execute_model(seqs, kv_caches)
        torch.cuda.synchronize()
        return
    
    @torch.inference_mode()
    def capture_model(self, kv_caches: List[torch.Tensor]) -> None:
        """Cuda graph capture a model.

        Note that CUDA graph's performance gain is negligible if number
        of batched tokens are larger than 200. And since CUDA graph
        requires fixed sized tensors, supporting large/variable batch
        size requires high GPU memory overhead. Thus, vLLM only captures
        decoding requests. Mixed batch (chunked prefill + decoding) or
        prefill requests are not captured.

        Since it is used for decoding-only, it assumes there's only 1 token
        per sequence in the batch.
        """
        assert not self.model_config.enforce_eager
        logger.info("Capturing the model for CUDA graphs. This may lead to "
                    "unexpected consequences if the model is not static. To "
                    "run the model in eager mode, set 'enforce_eager=True' or "
                    "use '--enforce-eager' in the CLI.")
        logger.info("CUDA graphs can take additional 1~3 GiB memory per GPU. "
                    "If you are running out of memory, consider decreasing "
                    "`gpu_memory_utilization` or enforcing eager mode. "
                    "You can also reduce the `max_num_seqs` as needed "
                    "to decrease memory usage.")
        start_time = time.perf_counter()

        # Prepare dummy inputs. These will be reused for all batch sizes.
        max_batch_size = max(_BATCH_SIZES_TO_CAPTURE)
        input_tokens = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        input_positions = torch.zeros(max_batch_size, dtype=torch.long).cuda()
        slot_mapping = torch.empty(max_batch_size, dtype=torch.long).cuda()
        slot_mapping.fill_(_PAD_SLOT_ID)
        seq_lens = torch.ones(max_batch_size, dtype=torch.int32).cuda()
        block_tables = torch.from_numpy(self.graph_block_tables).cuda()
        whisper_data = torch.zeros(
                        (30 * self.whisper_config.sample_rate), 
                        dtype=torch.float16)
        whisper_data = self.whisper_processor(
            whisper_data, 
            sampling_rate = self.whisper_config.sample_rate,
            return_tensors = 'pt',
        )
        whisper_data = whisper_data.to(self.model_config.dtype).input_features[0].cuda()

        # Prepare buffer for outputs. These will be reused for all batch sizes.
        # It will be filled after the first graph capture.
        hidden_states: Optional[torch.Tensor] = None

        graph_batch_size = _get_graph_batch_size(
            self.scheduler_config.max_num_seqs)
        batch_size_capture_list = [
            bs for bs in _BATCH_SIZES_TO_CAPTURE if bs <= graph_batch_size
        ]

        with graph_capture() as graph_capture_context:
            # NOTE: Capturing the largest batch size first may help reduce the
            # memory usage of CUDA graph.
            for batch_size in reversed(batch_size_capture_list):
                # Create dummy attn_metadata.
                attn_metadata = self.attn_backend.make_metadata(
                    num_prefills=0,
                    num_prefill_tokens=0,
                    num_decode_tokens=batch_size,
                    slot_mapping=slot_mapping[:batch_size],
                    seq_lens=None,
                    seq_lens_tensor=seq_lens[:batch_size],
                    max_query_len=None,
                    max_prefill_seq_len=0,
                    max_decode_seq_len=self.max_seq_len_to_capture,
                    query_start_loc=None,
                    seq_start_loc=None,
                    context_lens_tensor=None,
                    block_tables=block_tables[:batch_size],
                    use_cuda_graph=True,
                )

                if self.lora_config:
                    lora_mapping = LoRAMapping(
                        [0] * batch_size,
                        [0] * batch_size,
                    )
                    self.set_active_loras(set(), lora_mapping)

                graph_runner = CUDAGraphRunner(self.model)
                hidden_states = graph_runner.capture(
                    whisper_data,
                    input_tokens[:batch_size],
                    input_positions[:batch_size],
                    hidden_states[:batch_size]
                    if hidden_states is not None else None,
                    kv_caches,
                    attn_metadata,
                    memory_pool=self.graph_memory_pool,
                    stream=graph_capture_context.stream,
                )
                self.graph_memory_pool = graph_runner.graph.pool()
                self.graph_runners[batch_size] = graph_runner

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        # This usually takes < 10 seconds.
        logger.info("Graph capturing finished in %.0f secs.", elapsed_time)

class CUDAGraphRunner:

    def __init__(self, model: nn.Module):
        self.model = model
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

        self._graph: Optional[torch.cuda.CUDAGraph] = None

    @property
    def graph(self):
        assert self._graph is not None
        return self._graph

    def capture(
        self,
        whisper_data: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: Optional[torch.Tensor],
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        memory_pool: Optional[Tuple[int, int]],
        stream: torch.cuda.Stream,
        **kwargs,
    ) -> torch.Tensor:
        assert self._graph is None
        # Run the model a few times without capturing the graph.
        # This is to make sure that the captured graph does not include the
        # kernel launches for initial benchmarking (e.g., Triton autotune).
        # Note one iteration is not enough for torch.jit.script
        for _ in range(_NUM_WARMUP_ITERS):
            self.model(
                whisper_data,
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
        torch.cuda.synchronize()

        # Capture the graph.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph, pool=memory_pool, stream=stream):
            output_hidden_states = self.model(
                whisper_data,
                input_ids,
                positions,
                kv_caches,
                attn_metadata,
                **kwargs,
            )
            if hidden_states is not None:
                hidden_states.copy_(output_hidden_states)
            else:
                hidden_states = output_hidden_states
            del output_hidden_states
            # make sure `output_hidden_states` is deleted
            # in the graph's memory pool
            gc.collect()
        torch.cuda.synchronize()

        # Save the input and output buffers.
        self.input_buffers = {
            "whisper_data": whisper_data,
            "input_ids": input_ids,
            "positions": positions,
            "kv_caches": kv_caches,
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        self.output_buffers = {"hidden_states": hidden_states}
        return hidden_states

    def forward(
        self,
        whisper_data: torch.Tensor,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> torch.Tensor:
        # KV caches are fixed tensors, so we don't need to copy them.
        del kv_caches

        # Copy the input tensors to the input buffers.
        self.input_buffers["whisper_data"].copy_(whisper_data, non_blocking=True)
        self.input_buffers["input_ids"].copy_(input_ids, non_blocking=True)
        self.input_buffers["positions"].copy_(positions, non_blocking=True)
        self.input_buffers["slot_mapping"].copy_(attn_metadata.slot_mapping,
                                                 non_blocking=True)
        self.input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        self.input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        # Run the graph.
        self.graph.replay()

        # Return the output tensor.
        return self.output_buffers["hidden_states"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def _get_graph_batch_size(batch_size: int) -> int:
    """Returns the padded batch size given actual batch size.

    Batch sizes are 1, 2, 4, _BATCH_SIZE_ALIGNMENT,
    2*_BATCH_SIZE_ALIGNMENT, 3*_BATCH_SIZE_ALIGNMENT...
    """
    if batch_size <= 2:
        return batch_size
    elif batch_size <= 4:
        return 4
    else:
        return ((batch_size + _BATCH_SIZE_ALIGNMENT - 1) //
                _BATCH_SIZE_ALIGNMENT * _BATCH_SIZE_ALIGNMENT)


def _is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    if isinstance(block_tables, dict) and all(
            value is None for value in block_tables.values()):
        return True
    return False