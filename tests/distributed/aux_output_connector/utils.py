# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections import defaultdict

from vllm.distributed.aux_output_connector.object_store import MooncakeAuxObjectStore


class FakeMooncake:
    def __init__(self, code=0):
        self.code = code
        self.objects = {}
        self.payloads = defaultdict(list)
        self.writes = []
        self.removes = []
        self.configs = []
        self.closed = False

    def put_batch(self, keys, values, config):
        self.writes.append(list(keys))
        self.configs.append(config)
        if self.code == 0:
            for key, value in zip(keys, values, strict=True):
                self.objects.setdefault(key, value)
                self.payloads[key].append(value)
        return self.code

    def remove_by_regex(self, pattern, force):
        removed = [key for key in self.objects if re.search(pattern, key)]
        for key in removed:
            del self.objects[key]
        return len(removed)

    def batch_remove(self, keys, force):
        self.removes.append(list(keys))
        return [0 if self.objects.pop(key, None) is not None else -704 for key in keys]

    def get(self, key):
        return self.objects.get(key, b"")

    def close(self):
        self.closed = True


def fake_store(code=0):
    client = FakeMooncake(code)
    return client, MooncakeAuxObjectStore(client, None)
