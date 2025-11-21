import msgspec
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, EngineCoreRequest


class StreamingEngineCoreRequest(EngineCoreRequest):
    streaming_sequence_id: int = -1
    close_session: bool = False


class StreamingEngineCoreOutput(EngineCoreOutput):
    close_session: bool = False


class StreamingEngineCoreOutputs(EngineCoreOutputs):
    outputs: list[StreamingEngineCoreOutput] = msgspec.field(default_factory=list)
