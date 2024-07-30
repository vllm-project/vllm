import time
from array import array

def t():
    l = [i for i in range(256)]
    s = time.time()
    a = array('l')
    a.fromlist(l)
    print((time.time() - s) * 1000 * 1000, "us")

t()


import msgspec

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

encoder = msgspec.msgpack.Encoder(enc_hook=enc_hook)
decoder = msgspec.msgpack.Decoder(dec_hook=dec_hook)

l = [i for i in range(256)]
d = {"1": l}


with Timer("Serialization array"):
    # a = array('l')
    # a.fromlist(l)
    data = encoder.encode(a)
with Timer("Deserialization"):
    data = decoder.decode(data)

l = [i for i in range(256)]
a = array('l')
a.fromlist(l)


with Timer("Serialization bigger array"):
    # a = array('l')
    # a.fromlist(l)
    data = encoder.encode(a)
with Timer("Deserialization"):
    data = decoder.decode(data)


# for _ in range(5):
#     with Timer("Serialization list"):
#         data = encoder.encode(l)
#     with Timer("Deserialization"):
#         data = decoder.decode(data)
