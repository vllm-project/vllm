from dataclasses import dataclass
import torch
from typing import Dict, List, Optional, Union, Tuple

from vllm.sequence import IntermediateTensors
from vllm.config import MultiStreamConfig

from vllm.attention.backends.abstract import AttentionMetadata
from .base import MSAttentionMetadataSplitConfig, MSEventKey


def split_micro_batches_tensors(input_tensors, split_index: int, keys: List[str] = None):
    if isinstance(input_tensors, list):
        micro_batches = []
        for tensor in input_tensors:
            if tensor is None:
                micro_batches.append([None, None])
            else:
                micro_batches.append([tensor[:split_index], tensor[split_index:]])
        return micro_batches
    elif isinstance(input_tensors, torch.Tensor):
        return [input_tensors[:split_index], input_tensors[split_index:]]
    elif input_tensors is None:
        return [None, None]
    elif isinstance(input_tensors, Dict):
        assert keys is not None
        micro_batches_pre = {}
        for key in keys:
            micro_batches_pre[key] = input_tensors[key][:split_index]
        micro_batches_post = {}
        for key in keys:
            micro_batches_post[key] = input_tensors[key][split_index:]
        return [micro_batches_pre, micro_batches_post]
    else:
        raise NotImplementedError

def make_multistream_metadata(
        start_layer: int,
        end_layer: int,
        causal_lm: bool = True,
        multistream_config: Optional[MultiStreamConfig] = None,
):
    if multistream_config is None:
        return None

    return MultiStreamMetadata(
        calculate_stream=torch.cuda.current_stream(),
        communicate_stream=torch.cuda.Stream(),
        start_layer=start_layer,
        end_layer=end_layer,
        multistream_config=multistream_config,
        event_keys=[MSEventKey.ATTN_COM_FINISH, MSEventKey.ATTN_AR_FINISH,
                    MSEventKey.FFN_COM_FINISH, MSEventKey.FFN_AR_FINISH],
        causal_lm=causal_lm,
    )


@dataclass
class MultiStreamStepMetadata:
    comm_stream: torch.cuda.Stream = None
    before_comm_event: torch.cuda.Event = None
    after_comm_event: torch.cuda.Event = None

class MultiStreamMetadata:
    # direct stream
    calculate_stream = None
    # delay stream
    communicate_stream = None
    # events
    ms_events: Dict[int, Dict[int, Dict[MSEventKey, torch.cuda.Event]]] = {}
    # multi-stream-flag
    enable_multi_stream: bool = False
    # step wise metadata
    ms_step_metadata: Dict[int, Dict[int, Dict[str, MultiStreamStepMetadata]]] = {}

    def __init__(self,
                 calculate_stream: torch.cuda.Stream,
                 communicate_stream: torch.cuda.Stream,
                 start_layer: int,
                 end_layer: int,
                 event_keys: List[MSEventKey],
                 multistream_config: Optional[MultiStreamConfig],
                 causal_lm: bool = True,
                 ):
        self.calculate_stream = calculate_stream
        self.communicate_stream = communicate_stream
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.ms_config = multistream_config
        self.causal_lm = causal_lm

        self._build_events(event_keys)
        self._build_step_metadata()
        self._build_ms_split_config()

    def _build_events(self, event_keys):
        for i in range(self.start_layer - 1, self.end_layer):
            self.ms_events[i] = {}
            for j in range(self.ms_config.num_micro_batches):
                self.ms_events[i][j] = {}
                for key in event_keys:
                    self.ms_events[i][j][key] = torch.cuda.Event()

    def _build_step_metadata(self, ):
        for i in range(self.start_layer - 1, self.end_layer):
            self.ms_step_metadata[i] = {}
            for j in range(self.ms_config.num_micro_batches):
                self.ms_step_metadata[i][j] = {}
                self.ms_step_metadata[i][j]["forward_attn"] = MultiStreamStepMetadata(
                    comm_stream=self.communicate_stream,
                    before_comm_event=self.ms_events[i][j][MSEventKey.ATTN_COM_FINISH],
                    after_comm_event=self.ms_events[i][j][MSEventKey.ATTN_AR_FINISH],
                )
                self.ms_step_metadata[i][j]["forward_ffn"] = MultiStreamStepMetadata(
                    comm_stream=self.communicate_stream,
                    before_comm_event=self.ms_events[i][j][MSEventKey.FFN_COM_FINISH],
                    after_comm_event=self.ms_events[i][j][MSEventKey.FFN_AR_FINISH],
                )

    def _build_ms_split_config(self):
        self.ms_split_config = MSAttentionMetadataSplitConfig(
            num_micro_batches=self.ms_config.num_micro_batches,
            min_total_tokens_to_split=self.ms_config.min_total_tokens_to_split,
            min_prefill_tokens_to_split=self.ms_config.min_prefill_tokens_to_split,
            imbalance_ratio=self.ms_config.imbalance_ratio,
            enable_request_split=self.causal_lm,
        )

    def try_wait_event(self, layer_index: int, micro_batch_index: int, event_key: MSEventKey):
        self.ms_events[layer_index][micro_batch_index][event_key].wait()

    def try_record_event(self, layer_index: int, micro_batch_index: int, event_key: MSEventKey):
        self.ms_events[layer_index][micro_batch_index][event_key].record()

    def get_ms_step_metadata(self, layer_index: int, micro_batch_index: int, forward_stage: str):
        return self.ms_step_metadata[layer_index][micro_batch_index][forward_stage]

    def split_micro_batch(self,
                          attn_metadata: "AttentionMetadata",
                          intput_tensors: List[torch.Tensor],
                          intermediate_tensors: Optional[IntermediateTensors] = None,
                          intermediate_tensors_keys: Optional[List[str]] = None,
        ) -> Tuple[bool,
             Union[AttentionMetadata, List[AttentionMetadata]],
             Union[List[torch.Tensor], List[List[torch.Tensor]]],
             Union[IntermediateTensors, List[IntermediateTensors]]]:
        attn_metadata = attn_metadata.split_metadata_for_multistream(self.ms_split_config)
        if len(attn_metadata) == 1:
            return False, attn_metadata[0], intput_tensors, intermediate_tensors

        split_index = attn_metadata[0].slot_mapping.shape[0]
        input_tensors = split_micro_batches_tensors(intput_tensors, split_index)

        if intermediate_tensors is not None:
            inter_tensors_list = split_micro_batches_tensors(intermediate_tensors.tensors, split_index, intermediate_tensors_keys)
            intermediate_tensors = [
                IntermediateTensors(inter_tensors) for inter_tensors in inter_tensors_list
            ]

        return True, attn_metadata, input_tensors, intermediate_tensors

    def merge_micro_batches(self,
                            input_tensors: Union[List[torch.Tensor], List[List[torch.Tensor]]]
                            ) -> List[torch.Tensor]:
        if input_tensors is None or isinstance(input_tensors[0], torch.Tensor):
            return input_tensors

        batch = []
        for tensors in input_tensors:
            if tensors is None or tensors[0] is None:
                batch.append(None)
            else:
                batch.append(torch.cat(tensors, dim=0))

        return batch