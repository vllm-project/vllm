import msgspec
from vllm.executor.msgspec_utils import encode_hook, decode_hook
from vllm.sequence import ExecuteModelRequest
from ..spec_decode.utils import create_batch

def test_msgspec_serialization():
    num_lookahead_slots = 4
    seq_group_metadata_list, _, _ = create_batch(16, num_lookahead_slots)
    execute_model_req = ExecuteModelRequest(
        seq_group_metadata_list=seq_group_metadata_list,
        num_lookahead_slots=num_lookahead_slots,
        running_queue_size=4)
    
    encoder = msgspec.msgpack.Encoder(enc_hook=encode_hook)
    decoder = msgspec.msgpack.Decoder(
        ExecuteModelRequest, dec_hook=decode_hook)
    req = decoder.decode(encoder.encode(execute_model_req))
    assert (len(req.seq_group_metadata_list)
                == len(execute_model_req.seq_group_metadata_list))
