# SPDX-License-Identifier: Apache-2.0

import msgspec

from vllm.executor.msgspec_utils import decode_hook, encode_hook
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
    decoder = msgspec.msgpack.Decoder(ExecuteModelRequest,
                                      dec_hook=decode_hook)
    req = decoder.decode(encoder.encode(execute_model_req))
    expected = execute_model_req.seq_group_metadata_list
    actual = req.seq_group_metadata_list
    assert (len(expected) == len(actual))
    expected = expected[0]
    actual = actual[0]

    assert expected.block_tables == actual.block_tables
    assert expected.is_prompt == actual.is_prompt
    assert expected.request_id == actual.request_id
    assert (expected.seq_data[0].prompt_token_ids ==
            actual.seq_data[0].prompt_token_ids)
    assert (expected.seq_data[0].output_token_ids ==
            actual.seq_data[0].output_token_ids)
