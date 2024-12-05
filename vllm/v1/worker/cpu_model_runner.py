import torch
import numpy as np
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.model_executor.model_loader import get_model
from vllm.v1.outputs import ModelRunnerOutput
from vllm.forward_context import set_forward_context
from vllm.worker.cpu_model_runner import ModelInputForCPUBuilder
from vllm.attention import get_attn_backend

if TYPE_CHECKING:
    from vllm.v1.core.scheduler import SchedulerOutput

class CPUModelRunner(GPUModelRunner):
    #
    def __init__(self, vllm_config):
        super().__init__(vllm_config)
        self.use_cuda_graph = False
        num_attn_heads = self.model_config.get_num_attention_heads(
            self.parallel_config)
        needs_attn_backend = (num_attn_heads != 0
                              or self.model_config.is_attention_free)
        self.attn_backend = get_attn_backend(
            self.model_config.get_head_size(),
            self.model_config.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
        ) if needs_attn_backend else None
        self.input_data = ModelInputForCPUBuilder.ModelInputData(False)
        self.chunked_prefill = True
        self.att_metadata_builder = self.attn_backend.get_builder_cls()(
            self)

    @torch.inference_mode()
    def execute_model(
            self,
            scheduler_output: "SchedulerOutput",
    ) -> ModelRunnerOutput:
        self._update_states(scheduler_output)

        # Run the encoder.
        self._execute_encoder(scheduler_output)
        encoder_outputs = self._gather_encoder_outputs(scheduler_output)

        # Prepare the decoder inputs.
        input_ids, attn_metadata, logits_indices = self._prepare_inputs(
            scheduler_output)
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # only eager mode
        num_input_tokens = num_scheduled_tokens

        # Get the inputs embeds.
        if encoder_outputs:
            inputs_embeds = self.model.get_input_embeddings(
                input_ids, encoder_outputs)
        else:
            inputs_embeds = self.model.get_input_embeddings(input_ids)
        # NOTE(woosuk): To unify token ids and soft tokens (vision embeddings),
        # always use embeddings (rather than token ids) as input to the model.
        # TODO(woosuk): Avoid the copy. Optimize.
        self.inputs_embeds[:num_scheduled_tokens].copy_(inputs_embeds)

        # Run the decoder.
        # Use persistent buffers for CUDA graphs.
        with set_forward_context(attn_metadata, self.vllm_config):
            hidden_states = self.model(
                input_ids=None,
                positions=self.positions[:num_input_tokens],
                kv_caches=self.kv_caches,
                attn_metadata=None,
                inputs_embeds=self.inputs_embeds[:num_input_tokens],
            )
        hidden_states = hidden_states[:num_scheduled_tokens]
        hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self._prepare_sampling(scheduler_output)
        sampler_output = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

        # NOTE: CPU-GPU synchronization happens here.
        sampled_token_ids = sampler_output.sampled_token_ids.cpu()
        sampled_token_ids_list = sampled_token_ids.tolist()
        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        num_reqs = self.input_batch.num_reqs
        for i, req_id in enumerate(self.input_batch.req_ids[:num_reqs]):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            assert seq_len <= req_state.num_tokens
            if seq_len == req_state.num_tokens:
                # Append the sampled token to the output token ids.
                token_id = sampled_token_ids_list[i]
                self.input_batch.token_ids_cpu[i, seq_len] = token_id
                req_state.output_token_ids.append(token_id)
            else:
                # Ignore the sampled token from the partial request.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    # This relies on cuda-specific torch-internal impl details
                    generator.set_offset(generator.get_offset() - 4)

        if sampler_output.logprob_token_ids is None:
            logprob_token_ids = None
        else:
            logprob_token_ids = sampler_output.logprob_token_ids
        if sampler_output.logprobs is None:
            logprobs = None
        else:
            logprobs = sampler_output.logprobs
        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids[:num_reqs],
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids_cpu=sampled_token_ids,
            logprob_token_ids_cpu=logprob_token_ids,
            logprobs_cpu=logprobs,
        )
        return model_runner_output

    def load_model(self) -> None:
        self.model = get_model(vllm_config=self.vllm_config)

    def _prepare_inputs(self, scheduler_output: "SchedulerOutput"):
        # build input_data
        '''
            self.use_mrope = use_mrope
            self.input_tokens: List[int] = []
            self.input_positions: Optional[
                List[int]] = [] if not self.use_mrope else None
            self.token_type_ids: Optional[List[int]] = []
            self.seq_lens: List[int] = []
            self.query_lens: List[int] = []
            self.prefill_block_tables: List[List[int]] = []
            self.decode_block_tables: List[List[int]] = []
            self.max_decode_seq_len: int = 0
            self.num_prefills: int = 0
            self.num_prefill_tokens: int = 0
            self.num_decode_tokens: int = 0
            self.slot_mapping: List[int] = []
            self.multi_modal_inputs_list: List[MultiModalKwargs] = []
            self.multi_modal_placeholder_maps: Dict[
                str, MultiModalPlaceholderMap] = defaultdict(
                    MultiModalPlaceholderMap)
            self.input_mrope_positions: Optional[List[List[int]]] = [
                [] for _ in range(3)
            ] if self.use_mrope else None
        '''
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0
        #
        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        # self.input_batch.block_table[:num_reqs].copy_(
        #     self.input_batch.block_table_cpu_tensor[:num_reqs],
        #     non_blocking=True)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = []
        max_num_scheduled_tokens = 0
        for req_id in self.input_batch.req_ids[:num_reqs]:
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            if num_tokens == 1:
                num_decode_tokens += 1
            else:
                num_prefills += 1
                num_prefill_tokens += num_tokens
            num_scheduled_tokens.append(num_tokens)
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)
        num_scheduled_tokens = np.array(num_scheduled_tokens, dtype=np.int32)
        assert max_num_scheduled_tokens > 0

        # calculate block tables info for cpu
        block_tables = self.input_batch.block_table_cpu_tensor[:num_reqs]
        decode_block_tables = block_tables[num_decode_tokens:]
        prefill_block_tables = block_tables[:num_prefills]
        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        indices = np.arange(num_reqs)
        req_indices = np.repeat(indices, num_scheduled_tokens)

        # Get batched arange.
        # E.g., [2, 5, 3] -> [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange_matrix = np.tile(np.arange(max_num_scheduled_tokens),
                                (num_reqs, 1))
        mask = arange_matrix < num_scheduled_tokens[:, np.newaxis]
        arange = arange_matrix[mask]

        # Get positions.
        positions = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        positions_np = positions.numpy()
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = positions_np + req_indices * self.max_model_len
        token_indices = torch.from_numpy(token_indices)
        input_ids = torch.empty((total_num_scheduled_tokens, ),
                                dtype=torch.int32,
                                device="cpu",
                                pin_memory=self.pin_memory)
        torch.index_select(torch.from_numpy(
            self.input_batch.token_ids_cpu).flatten(),
                           0,
                           token_indices,
                           out=input_ids)

        # Calculate the slot mapping.
        block_numbers = self.input_batch.block_table_cpu_tensor.flatten()[
            token_indices // self.block_size]
        block_offsets = token_indices % self.block_size
        slot_mapping = torch.empty((total_num_scheduled_tokens, ),
                                   dtype=torch.int32,
                                   device="cpu",
                                   pin_memory=self.pin_memory)
        torch.add(block_numbers * self.block_size,
                  block_offsets,
                  out=slot_mapping)

        # Prepare the attention metadata.
        query_start_loc = torch.empty((num_reqs + 1, ),
                                      dtype=torch.int32,
                                      device="cpu",
                                      pin_memory=self.pin_memory)
        query_start_loc_np = query_start_loc.numpy()
        query_start_loc_np[0] = 0
        np.cumsum(num_scheduled_tokens, out=query_start_loc_np[1:])

        seq_lens = (self.input_batch.num_computed_tokens_cpu[:num_reqs] +
                    num_scheduled_tokens)
        max_seq_len = seq_lens.max()
        # seq_start_loc = torch.empty((num_reqs + 1, ),
        #                             dtype=torch.int32,
        #                             device="cpu",
        #                             pin_memory=self.pin_memory)
        # seq_start_loc_np = seq_start_loc.numpy()
        # seq_start_loc_np[0] = 0
        # np.cumsum(seq_lens, out=seq_start_loc_np[1:])

        # input_ids = input_ids.to(self.device, non_blocking=True)
        self.positions[:total_num_scheduled_tokens].copy_(positions,
                                                          non_blocking=True)
        # build input_data for cpu
        data = self.input_data
        data.use_mrope = False
        data.seq_lens = seq_lens
        data.query_lens = num_scheduled_tokens
        data.num_decode_tokens = num_decode_tokens
        data.num_prefills = num_prefills
        data.num_prefill_tokens = num_prefill_tokens
        data.input_tokens = input_ids
        data.max_decode_seq_len = max_seq_len #?
        data.decode_block_tables = decode_block_tables
        data.prefill_block_tables = prefill_block_tables
        data.slot_mapping = slot_mapping
        attn_metadata = self.att_metadata_builder.build(
            data.seq_lens, data.query_lens, -1, -1)
        # NOTE(woosuk): Due to chunked prefills, there can be at most 1 partial
        # request in the batch. While we should not sample any token from this
        # partial request, we do so for simplicity. We will ignore the sampled
        # token from the partial request.
        # TODO: Support prompt logprobs.
        logits_indices = query_start_loc[1:] - 1
        return input_ids, attn_metadata, logits_indices