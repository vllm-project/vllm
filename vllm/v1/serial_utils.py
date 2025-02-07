# SPDX-License-Identifier: Apache-2.0

import pickle


class PickleEncoder:

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, data):
        return pickle.loads(data)
