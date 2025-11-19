from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCoreProc
from vllm.v1.request import Request
from vllm.v1.streaming.engine import StreamingEngineCoreRequest
from vllm.v1.streaming.streaming_request import StreamingRequest


class StreamingEngineCoreProc(EngineCoreProc):
    request_cls: type[Request] = StreamingRequest
    engine_request_cls: type[EngineCoreRequest] = StreamingEngineCoreRequest
