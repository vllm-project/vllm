from array import array
from vllm.sequence import ExecuteModelRequest, SequenceData
import msgspec

with open('example.bin', 'rb') as file:
    data = file.read()

def dec_hook(type, obj):
    # `type` here is the value of the custom type annotation being decoded.
    if type is array:
        deserialized = array('l')
        deserialized.frombytes(obj)
        return deserialized
    
# decoder = msgspec.msgpack.Decoder(ExecuteModelRequest, dec_hook=dec_hook)


# print(decoder.decode(data))

def enc_hook(obj):
    if isinstance(obj, array):
        # convert the complex to a tuple of real, imag
        return obj.tobytes()

encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
decoder = msgspec.msgpack.Decoder(SequenceData, dec_hook=dec_hook)

data = SequenceData([1, 2, 3])
print(decoder.decode(encoder.encode(data)))
