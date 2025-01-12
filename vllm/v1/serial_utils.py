import pickle

import cloudpickle


class PickleEncoder:

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, data):
        return pickle.loads(data)


class CloudPickleEncoder:

    def encode(self, obj):
        return cloudpickle.dumps(obj)

    def decode(self, data):
        return cloudpickle.loads(data)
