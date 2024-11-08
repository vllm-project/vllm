"""
    This file implements a simple torch distributed connector by 2 classes:
    - `TorchDistributedPipe`: a tensor transmission pipe between P/D instance,
      using `torch.distributed`
    - `TorchDistributedConnector`: a torch distributed connector between P/D 
      instance, implemented on top of `TorchDistributedPipe`
"""
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Deque, List, Optional, Union

import torch
from torch.distributed import Backend

from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.logger import init_logger



logger = init_logger(__name__)

# if the tensor is only one-element and only contains NONE_INT
# this means that the sended object is None.
NONE_INT = -150886311

# Mapping tensor dtype to INT64, used for tensor metadata transmission
FLOAT16_INT = -543205003776624
INT64_INT = -375623078607432
BOOL_INT = -28035262008646
BFLOAT16_INT = -452084912267662
FLOAT32_INT = -1049557997456592
FLOAT64_INT = -452201007054137
FLOAT8_E4M3FN_INT = -1066697177659525
FLOAT8_E5M2_INT = -618182574682355

DTYPE2INT = {
    torch.float16: FLOAT16_INT,
    torch.int64: INT64_INT,
    torch.bool: BOOL_INT,
    torch.bfloat16: BFLOAT16_INT,
    torch.float32: FLOAT32_INT,
    torch.float64: FLOAT64_INT,
    torch.float8_e4m3fn: FLOAT8_E4M3FN_INT,
    torch.float8_e5m2: FLOAT8_E5M2_INT,
}

INT2DTYPE = {
    FLOAT16_INT: torch.float16,
    INT64_INT: torch.int64,
    BOOL_INT: torch.bool,
    BFLOAT16_INT: torch.bfloat16,
    FLOAT32_INT: torch.float32,
    FLOAT64_INT: torch.float64,
    FLOAT8_E4M3FN_INT: torch.float8_e4m3fn,
    FLOAT8_E5M2_INT: torch.float8_e5m2,
}


class BrokenPipeException(Exception):

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class TorchDistributedPipe:
    
    METADATA_LENGTH = 16
    MAX_TENSOR_DIMENSIONS = 14
    METADATA_DTYPE = torch.int64

    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
    ):
        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank
        self.device_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend)
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self.device_group = device_group

        assert self.device_group is not None
        assert self.rank_in_group <= 1

        self.device = self._select_device(torch_distributed_backend)

        self.target_rank_for_send = self.ranks[(self.rank_in_group + 1) %
                                               self.world_size]
        self.target_rank_for_recv = self.ranks[(self.rank_in_group - 1) %
                                               self.world_size]

        self.transport_thread: Optional[ThreadPoolExecutor] = None
        self.buffer_size = 0
        self.buffer_size_lock = threading.Lock()

        self.none_tensor = torch.tensor([NONE_INT], device=self.device)

        # On-device tensors to be reused for recv
        self.rcv_metadata_buffer = torch.zeros(self.METADATA_LENGTH,
                                               dtype=self.METADATA_DTYPE,
                                               device=self.device)

    def _select_device(self, backend: Union[str, Backend]):
        if torch.cuda.is_available() and backend == Backend.NCCL:
            return torch.device(f"cuda:{self.local_rank}")
        else:
            return "cpu"

    def _make_metadata(self, tensor: torch.Tensor) -> torch.Tensor:
        """Create the metadata on based on the input tensor, and move it to GPU.
        The metadata's length is `TorchDistributedPipe.METADATA_LENGTH`.

        Currently, the metadata is a int64 tensor and it includes dtype, number
        of dimensions, and the shape information of the input tensor.


        The information follows the layout below:
        - metadata[0] -- dtype
        - metadata[1] -- number of dimensions
        - metadata[2 : 2+ndims] -- the shape of the input tensor

        Parameters:
            - tensor: the input tensor

        Returns:
            - metadata: the metadata tensor, on self.device
        """
        buffer = torch.empty(self.METADATA_LENGTH,
                             dtype=self.METADATA_DTYPE,
                             device="cpu")
        buffer[0] = DTYPE2INT[tensor.dtype]
        ndims = len(tensor.shape)
        buffer[1] = len(tensor.shape)
        buffer[2:2 + ndims] = torch.tensor(tensor.shape,
                                           dtype=self.METADATA_DTYPE)
        return buffer.to(self.device)

    def _prepare_recv_buffer(self,
                             d_metadata_buffer: torch.Tensor) -> torch.Tensor:
        """Create a buffer to receive the tensor based on the metadata.

        Parameters:
            - d_metadata_buffer: the metadata tensor on self.device

        Returns:
            - buffer: the buffer tensor to receive the tensor, on self.device
        """
        h_buffer = d_metadata_buffer.cpu().numpy()
        dtype = INT2DTYPE[h_buffer[0]]
        ndims = h_buffer[1]
        shape = tuple(h_buffer[2:2 + ndims])
        return torch.empty(shape, dtype=dtype, device=self.device)

    def _send_metadata(self, d_metadata_buffer: torch.Tensor):
        """Send the metadata buffer to the target rank.
        """
        torch.distributed.send(
            d_metadata_buffer,
            dst=self.target_rank_for_send,
            group=self.device_group,
        )

    def _recv_metadata(self) -> torch.Tensor:
        """Receive the metadata buffer from the target rank.

        Returns:
            - metadata_buffer: the metadata buffer tensor, on self.device

        Note:
            The current implementation uses the assumption that there is no
            race conditions during sending/receiving. Therefore, the metadata
            buffer can be reused
        """
        torch.distributed.recv(
            self.rcv_metadata_buffer,
            src=self.target_rank_for_recv,
            group=self.device_group,
        )

        return self.rcv_metadata_buffer

    def _send_impl(self, tensor):
        """
        The actual implementation of sending the tensor to the target rank.
        This function will first send the metadata, and then send the tensor.

        Parameters:
            - tensor: the input tensor to be sent
        """

        metadata = self._make_metadata(tensor)
        self._send_metadata(metadata)
        torch.distributed.send(tensor.to(self.device),
                               dst=self.target_rank_for_send,
                               group=self.device_group)

    def _recv_impl(self) -> torch.Tensor:
        """
        The actual implementation of receiving the tensor from the target rank.
        This function will first receive the metadata, then receive the tensor.

        This function will block if there is no tensor to receive.

        Returns:
            - buffer: the received tensor, on self.device
        """
        d_metadata = self._recv_metadata()
        buffer = self._prepare_recv_buffer(d_metadata)

        torch.distributed.recv(buffer,
                               src=self.target_rank_for_recv,
                               group=self.device_group)

        return buffer

    def send_tensor_wrapper(self, tensor):
        try:
            """Wrapper for send_tensor_dict"""
            tensor_size = tensor.element_size() * tensor.numel()
            self._send_impl(tensor)

            with self.buffer_size_lock:
                self.buffer_size = self.buffer_size - tensor_size
        except Exception as e:
            logger.error("[rank%d]: Exception when trying to send %s, msg: %s",
                         torch.distributed.get_rank(), str(tensor), str(e))
            import traceback
            traceback.print_exc()

    def block_if_full(self):
        """Block the current thread if the buffer size is larger than 1e9."""
        # TODO: replace this 1e9 with a configurable parameter or a constant
        while self.buffer_size > 1e9:
            logger.debug("KV cache transfer pipe is full. Waiting...")
            time.sleep(0.05)

    def send_tensor(self, tensor: Optional[torch.Tensor]) -> None:
        """Sends a tensor to the destination rank in a non-blocking way.
        Flow: send tensor dim -- send tensor shape -- send tensor data
        """

        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        if tensor is None:
            tensor = self.none_tensor

        tensor_size = tensor.element_size() * tensor.numel()

        assert (
            0 < len(tensor.shape) < self.MAX_TENSOR_DIMENSIONS
        ), f"Only support dimensions within 1-{self.MAX_TENSOR_DIMENSIONS}"

        self.block_if_full()

        with self.buffer_size_lock:
            self.buffer_size = self.buffer_size + tensor_size

        self.transport_thread.submit(
            self.send_tensor_wrapper,
            tensor,
        )

    def recv_tensor(self) -> Optional[torch.Tensor]:
        """Receives a tensor from the src rank. Blocking."""
        if self.transport_thread is None:
            self.transport_thread = ThreadPoolExecutor(max_workers=1)

        future = self.transport_thread.submit(self._recv_impl)

        try:
            tensor = future.result()
        except Exception as e:
            # the underlying pipe is likely broken
            logger.error("Encountering exception in KV receiving thread")
            logger.error("%s", e)
            # fault tolerance: if the pipe is broken, return None
            return None

        if tensor.numel() == 1 and tensor.item() == NONE_INT:
            return None
        else:
            return tensor

    def close(self):
        """Close the pipe and release the resources."""
        if (hasattr(self, "transport_thread")
                and self.transport_thread is not None):
            self.transport_thread.shutdown()


class TorchDistributedBuffer:

    def __init__(self, 
                 signal_pipe: TorchDistributedPipe,
                 data_pipe: TorchDistributedPipe,
                 buffer_size_thresh: int):
        """
        signal_pipe: on CPU 
        
        NOTE: on-device recv will block all threads in the process, making the 
        KV cache producer unable to listen to new request while transmitting 
        KV cache. Luckily CPU recv only blocks the current thread so we use 
        CPU recv to listen to new request.
        
        data_pipe: on device (e.g. GPU)
        """

        self.buffer: Deque[List[torch.Tensor]] = deque()

        self.buffer_size = 0
        self.buffer_size_threshold = buffer_size_thresh
        self.buffer_lock = threading.Lock()
        self.signal_pipe = signal_pipe
        self.data_pipe = data_pipe
        self.request_handling_thread: Optional[threading.Thread] = None

        self.normal_signal = torch.tensor([0])
        self.end_signal = None

    def _matches(self, tokens_roi_sender: List[torch.Tensor],
                 tokens_roi_recver: List[torch.Tensor]):

        # tokens_roi_sender: tokens and roi of the producer (in the buffer)
        # tokens_roi_recver: tokens and roi of the consumer (query)

        tokens_sender = tokens_roi_sender[0]
        tokens_recver = tokens_roi_recver[0]
        roi_sender = tokens_roi_sender[1]
        roi_recver = tokens_roi_recver[1]

        if tokens_recver is None:
            # consumer sends an empty request
            # semantics: DROP SELECT * LIMIT 1
            # so any of the data in the buffer can be drop-selected
            return True

        # Assuming that roi is a binary mask on tokens
        tokens_sender = tokens_sender[roi_sender]
        tokens_recver = tokens_recver[roi_recver]

        # simple common prefix matching
        min_length = min(len(tokens_sender), len(tokens_recver))
        if torch.allclose(tokens_sender[:min_length],
                          tokens_recver[:min_length]):
            return min_length

        return 0

    def _send_tensor_and_dec_size(self,
                                  tensor: Optional[torch.Tensor]) -> None:

        assert tensor is not None, "Use self.data_pipe.send(None) instead"
        self.buffer_size -= tensor.element_size() * tensor.numel()
        self.data_pipe.send_tensor(tensor)

    def _get_element_size(self, data: Optional[Union[List, torch.Tensor]]):

        if isinstance(data, torch.Tensor):
            return data.element_size() * data.numel()
        if not data:
            # cannot perform `not data` on a tensor
            # so this check needs to go after the check above
            return 0

        raise AssertionError("Unknown data type %s" % type(data))

    def _add_to_buffer(self, input_tokens: torch.Tensor, roi: torch.Tensor,
                       key: torch.Tensor, value: torch.Tensor,
                       hidden: torch.Tensor):

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()
        if isinstance(key, torch.Tensor):
            key = key.clone()
        if isinstance(value, torch.Tensor):
            value = value.clone()
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.clone()

        buffer_item = [input_tokens, roi, key, value, hidden]

        with self.buffer_lock:
            for data in buffer_item:
                self.buffer_size += self._get_element_size(data)
            self.buffer.append(buffer_item)

    def _is_end_signal(self, signal):
        return signal is None

    def drop_select_handler(self):

        try:

            while True:
                signal = self.signal_pipe.recv_tensor()
                if self._is_end_signal(signal):
                    logger.info("Received end signal!")
                    break

                input_tokens = self.data_pipe.recv_tensor()

                roi = self.data_pipe.recv_tensor()
                tokens_roi_recver = [input_tokens, roi]

                matched_length = 0

                # perform input tokens and roi matching
                # FIXME: this matching is O(n), ideally it should be O(1)
                # but this buffer size won't (and shouldn't) be too large so
                # the fix is not urgent.
                with self.buffer_lock:

                    for _ in range(len(self.buffer)):

                        temp_length = self._matches(self.buffer[0],
                                                    tokens_roi_recver)
                        if temp_length > 0:
                            matched_length = temp_length
                            break
                        # rotate the element we just accessed to the end
                        self.buffer.rotate(-1)

                    if matched_length > 0:
                        # need to clone the tensor
                        # in case the tensor is freed before sending finishes
                        matched_item = self.buffer.popleft()
                        for tensor in matched_item:
                            self._send_tensor_and_dec_size(tensor)

                    else:
                        # no match, just send None
                        for _ in range(5):
                            self.data_pipe.send_tensor(None)

        except RuntimeError as e:
            if 'Connection closed by peer' not in str(e):
                raise e

        logger.debug("Closing drop_select_handler")

    def drop_select(
            self, input_tokens: Optional[torch.Tensor],
            roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:

        assert self.request_handling_thread is None, \
            "drop_select should be called by the KV cache consumer "\
            "(e.g. the decode vLLM instance)"

        if isinstance(input_tokens, torch.Tensor):
            input_tokens = input_tokens.clone()
        if isinstance(roi, torch.Tensor):
            roi = roi.clone()

        self.signal_pipe.send_tensor(self.normal_signal)
        self.data_pipe.send_tensor(input_tokens)
        self.data_pipe.send_tensor(roi)

        input_tokens = self.data_pipe.recv_tensor()
        roi = self.data_pipe.recv_tensor()
        key = self.data_pipe.recv_tensor()
        value = self.data_pipe.recv_tensor()
        hidden = self.data_pipe.recv_tensor()

        return [input_tokens, roi, key, value, hidden]

    def full_handler(self):
        time.sleep(0.001)

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:

        if self.buffer_size > self.buffer_size_threshold:
            # log outside the while loop to avoid this message being logged
            # repeatedly.
            logger.debug("KV transfer buffer is full. Handling...")
        while self.buffer_size > self.buffer_size_threshold:
            self.full_handler()

        self._add_to_buffer(input_tokens, roi, key, value, hidden)

        # when calling the insert, the current process is a sender
        # need to launch the request handler and start listening to request.
        if self.request_handling_thread is None:
            self.request_handling_thread = threading.Thread(
                target=self.drop_select_handler)
            self.request_handling_thread.start()

    def close(self):

        if hasattr(self, "request_handling_thread"
                   ) and self.request_handling_thread is not None:
            self.request_handling_thread.join()

        else:
            # TODO: have a explicit close signal and have a explicit way to
            # check if it's requester
            self.signal_pipe.send_tensor(self.end_signal)
            
            
class TorchDistributedConnector(KVConnectorBase):
    
    def __init__(
        self,
        group_ranks: List[List[int]],
        local_rank: int,
        torch_distributed_backend: Union[str, Backend],
        # FIXME(Kuntai): remove this hardcoding
        lookup_buffer_size: int):

        self.lookup_buffer_size = lookup_buffer_size

        self.send_buffer: Optional[TorchDistributedBuffer] = None
        self.recv_buffer: Optional[TorchDistributedBuffer] = None

        SimpleKVLookupBuffer = sklb.SimpleKVLookupBuffer

        # In disaggregated prefill, the prefill vLLM only uses send pipe
        # and the decode vLLM only uses recv pipe
        # In remote KV cache store, vLLM will use both send pipe and recv pipe
        # So we build both send pipe and recv pipe for simplicity.
        if IS_KV_PRODUCER:

            self.send_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.send_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.recv_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.recv_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.send_buffer = SimpleKVLookupBuffer(self.send_signal_pipe,
                                                    self.send_pipe,
                                                    self.lookup_buffer_size)
            self.recv_buffer = SimpleKVLookupBuffer(self.recv_signal_pipe,
                                                    self.recv_pipe,
                                                    self.lookup_buffer_size)
            self.tensor_device = DISTRIBUTED_DEVICE
        else:

            # the current vLLM instance is KV consumer, so it needs to connect
            # its recv pipe to the send pipe of KV producder

            self.recv_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.recv_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.send_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                DISTRIBUTED_BACKEND,
            )
            self.send_signal_pipe = TorchDistributedPipe(
                group_ranks,
                local_rank,
                "gloo",
            )
            self.send_buffer = SimpleKVLookupBuffer(self.send_signal_pipe,
                                                    self.send_pipe,
                                                    self.lookup_buffer_size)
            self.recv_buffer = SimpleKVLookupBuffer(self.recv_signal_pipe,
                                                    self.recv_pipe,
                                                    self.lookup_buffer_size)
            self.tensor_device = DISTRIBUTED_DEVICE
            
            
    def select(
        self, input_tokens: Optional[torch.Tensor],
        roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        return self.send_buffer.drop_select(input, roi)
    
    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        return self.recv_buffer.insert(
            input_tokens,
            roi,
            key,
            value,
            hidden
        )

            
            
    def build_partial_prefill_input(
        self,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        input_tokens_list: List[torch.Tensor],
        num_computed_tokens_list: List[int],
        start_pos_list: List[int],
        slot_mapping_flat: torch.Tensor,
        device: torch.device,
    ) -> "ModelInputForGPUWithSamplingMetadata":
        """
        Helper function to rebuild the model input for the current request.
        Goal: avoid running redundant prefill on those tokens that already has
        KV caches received.
        """
        rebuilt_input_tokens = []
        rebuilt_input_positions = []
        rebuilt_query_lens = []

        rebuilt_num_prefills = 0
        rebuilt_num_prefill_tokens = 0
        rebuilt_slot_mapping = []
        rebuilt_max_query_len = 0

        rebuilt_block_tables = []

        rebuilt_query_start_loc = [0]
        rebuilt_context_lens_tensor = []
        rebuilt_selected_token_indices = []

        # recounting query and context lengths
        for idx in range(len(input_tokens_list)):
            token_tensor = input_tokens_list[idx]
            num_token = len(token_tensor)
            num_computed_token = num_computed_tokens_list[idx]
            # currently attention kernel cannot handle the case where there is 0
            # query token.
            if num_computed_token == num_token:
                num_computed_token -= 1
            start_pos = start_pos_list[idx]

            rebuilt_input_tokens.append(token_tensor[num_computed_token:])
            # TODO(Jiayi): please check the correctness of next line
            rebuilt_input_positions.append(
                model_input.input_positions[start_pos +
                                            num_computed_token:start_pos +
                                            num_token])
            q_len = num_token - num_computed_token
            rebuilt_query_lens.append(q_len)

            # Attn metadata-related
            rebuilt_num_prefills += 1
            rebuilt_num_prefill_tokens += q_len
            new_slot_mapping = slot_mapping_flat[start_pos +
                                                num_computed_token:start_pos +
                                                num_token]
            rebuilt_slot_mapping.append(new_slot_mapping)
            rebuilt_max_query_len = max(q_len, rebuilt_max_query_len)
            # TODO(Jiayi): remove hard-code (block_size=16)
            blk_size = 16
            temp_block_table = [
                slot_mapping_flat[i] // blk_size
                for i in range(start_pos, start_pos + num_token, blk_size)
            ]
            rebuilt_block_tables.append(temp_block_table)
            rebuilt_query_start_loc.append(
                rebuilt_num_prefill_tokens)  #start with 0
            rebuilt_context_lens_tensor.append(num_computed_token)

            # Sampling metadata related
            #seq_groups (use rebuilt query lens)
            rebuilt_selected_token_indices.append(rebuilt_num_prefill_tokens - 1)

        # rebuilt attn_metadata
        rebuilt_attn_metadata = deepcopy(model_input.attn_metadata)
        rebuilt_attn_metadata.num_prefills = rebuilt_num_prefills
        rebuilt_attn_metadata.num_prefill_tokens = rebuilt_num_prefill_tokens
        rebuilt_attn_metadata.slot_mapping = torch.cat(rebuilt_slot_mapping).to(
            device)
        rebuilt_attn_metadata.max_query_len = rebuilt_max_query_len

        rebuilt_attn_metadata.block_tables = torch.tensor(
            rebuilt_block_tables,
            dtype=model_input.attn_metadata.block_tables.dtype).to(device)

        rebuilt_attn_metadata.query_start_loc = torch.tensor(
            rebuilt_query_start_loc,
            dtype=model_input.attn_metadata.query_start_loc.dtype).to(device)
        rebuilt_attn_metadata.context_lens_tensor = torch.tensor(
            rebuilt_context_lens_tensor,
            dtype=model_input.attn_metadata.context_lens_tensor.dtype,
        ).to(device)

        rebuilt_attn_metadata._cached_prefill_metadata = None

        # rebuilt sampling_metadata
        rebuilt_sampling_metadata = deepcopy(model_input.sampling_metadata)
        for idx, q_len in enumerate(rebuilt_query_lens):
            if rebuilt_sampling_metadata.seq_groups is not None:
                rebuilt_sampling_metadata.seq_groups[idx].query_len = q_len

        rebuilt_sampling_metadata.selected_token_indices = torch.tensor(
            rebuilt_selected_token_indices,
            dtype=model_input.sampling_metadata.selected_token_indices.dtype,
        ).to(device)

        # import here to avoid circular import.
        from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
        rebuilt_model_input = ModelInputForGPUWithSamplingMetadata(
            input_tokens=torch.cat(rebuilt_input_tokens).to(device),
            input_positions=torch.cat(rebuilt_input_positions).to(device),
            seq_lens=model_input.seq_lens,
            query_lens=rebuilt_query_lens,
            lora_mapping=model_input.lora_mapping,
            lora_requests=model_input.lora_requests,
            attn_metadata=rebuilt_attn_metadata,
            prompt_adapter_mapping=model_input.prompt_adapter_mapping,
            prompt_adapter_requests=model_input.prompt_adapter_requests,
            multi_modal_kwargs=model_input.multi_modal_kwargs,
            request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
            finished_requests_ids=model_input.finished_requests_ids,
            virtual_engine=model_input.virtual_engine,
            sampling_metadata=rebuilt_sampling_metadata,
            is_prompt=model_input.is_prompt,
        )

        return rebuilt_model_input

