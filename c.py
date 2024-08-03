import time
import numpy as np


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


# l = [i for i in range(4096)]
# from array import array
# with Timer("converesion"):
#     arr = array("I", l)

from ray import cloudpickle as pickle
# import pickle

bytes = b"1" * 65665
with Timer("bytes pickling"):
    data = pickle.dumps(bytes)
with Timer("bytes deser"):
    pickle.loads(data)

