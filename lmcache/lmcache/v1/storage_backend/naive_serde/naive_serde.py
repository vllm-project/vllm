# SPDX-License-Identifier: Apache-2.0
# First Party
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


class NaiveSerializer(Serializer):
    def __init__(self):
        pass

    def serialize(self, memory_obj: MemoryObj) -> MemoryObj:
        memory_obj.ref_count_up()
        return memory_obj


class NaiveDeserializer(Deserializer):
    def deserialize(self, memory_obj: MemoryObj) -> MemoryObj:
        return memory_obj
