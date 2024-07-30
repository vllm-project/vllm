import time
import sys
from array import array
from vllm.sequence import ExecuteModelRequest, SequenceData, SequenceDataDelta, SequenceStage
import msgspec

with open('example.bin', 'rb') as file:
    data = file.read()

def dec_hook(type, obj):
    # `type` here is the value of the custom type annotation being decoded.
    if type is array:
        deserialized = array('l')
        deserialized.frombytes(obj)
        return deserialized

def enc_hook(obj):
    if isinstance(obj, array):
        # convert the complex to a tuple of real, imag
        return obj.tobytes()
    
class Timer:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self  # This allows access to the instance in the 'as' part of the context manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.elapsed_us = (self.end - self.start) * 1000 * 1000
        print(f"{self.msg=}. Elapsed time: {self.elapsed_us:.2f} us")

# encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
# decoder = msgspec.msgpack.Decoder(ExecuteModelRequest, dec_hook=dec_hook)

# with Timer("Serialization"):
    # serialized = encoder.encode(data)
# print(f"{sys.getsizeof(data)=}")
# with Timer("Deserialization original"):
#     decoder.decode(data)
# with Timer("Deserialization original"):
#     data = decoder.decode(data)

# with Timer("Serialization, big block tables"):
#     data = encoder.encode(data)
# with Timer("Deserialization, big block tables"):
#     data = decoder.decode(data)

# for i, metadata in enumerate(data.seq_group_metadata_list):
#     for key, value in metadata.block_tables.items():
#         metadata.block_tables[key] = [i]

# with Timer("Serialization, small block tables"):
#     data = encoder.encode(data)
# with Timer("Deserialization, small block tables"):
#     data = decoder.decode(data)

# print(decoder.decode(encoder.encode(data)))

encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
decoder = msgspec.msgpack.Decoder(SequenceDataDelta, dec_hook=dec_hook)

data = SequenceDataDelta([i for i in range(2048)], 0, 0, SequenceStage.DECODE)
with Timer("Serialization, big block tables"):
    data = encoder.encode(data)
with Timer("Deserialization, big block tables"):
    data = decoder.decode(data)
