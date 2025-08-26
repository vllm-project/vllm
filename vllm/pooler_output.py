import msgspec

from vllm.sequence import PoolingSequenceGroupOutput


class PoolerOutput(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """The output from a pooling operation in the pooling model."""
    outputs: list[PoolingSequenceGroupOutput]

    def get_data_nbytes(self) -> int:
        return sum(o.get_data_nbytes() for o in self.outputs)

    def __getitem__(self, idx: int) -> PoolingSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value: PoolingSequenceGroupOutput):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self.outputs == other.outputs
