# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time

import regex as re

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectWriter,
    MooncakeAuxObjectStore,
)


class FakeMooncake:
    def __init__(self, code=0):
        self.code = code
        self.objects = {}
        self.writes = []
        self.configs = []
        self.closed = False
        self.patterns = []

    def put_batch(self, keys, values, config):
        self.writes.append(list(keys))
        self.configs.append(config)
        if self.code == 0:
            for key, value in zip(keys, values, strict=True):
                self.objects.setdefault(key, value)
        return self.code

    def remove_by_regex(self, pattern, force):
        self.patterns.append(pattern)
        removed = [key for key in self.objects if re.search(pattern, key)]
        for key in removed:
            del self.objects[key]
        return len(removed)

    def remove(self, key, force):
        return 0 if self.objects.pop(key, None) is not None else -704

    def get(self, key):
        return self.objects.get(key, b"")

    def close(self):
        self.closed = True


class DictObjectStore:
    """`AuxObjectStore` on a dict, as a store other than Mooncake."""

    created: list["DictObjectStore"] = []

    def __init__(self, **extra_config):
        self.extra_config = extra_config
        self.objects = {}

    @classmethod
    def create(cls, **extra_config):
        cls.created.append(cls(**extra_config))
        return cls.created[-1]

    def put_batch(self, keys, values):
        for key, value in zip(keys, values, strict=True):
            self.objects.setdefault(key, value)

    def get(self, key):
        return self.objects.get(key)

    def remove_prefix(self, prefix):
        for key in [key for key in self.objects if key.startswith(prefix)]:
            del self.objects[key]

    def remove(self, keys):
        for key in keys:
            self.objects.pop(key, None)

    def close(self):
        pass


def fake_store(code=0):
    client = FakeMooncake(code)
    return client, MooncakeAuxObjectStore(client, None)


def wait_for_writer(writer: AuxObjectWriter, interval_s: float = 0.01) -> None:
    deadline = time.monotonic() + 5
    while writer._pending and time.monotonic() < deadline:
        time.sleep(interval_s)
    assert not writer._pending
