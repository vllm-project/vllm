import pickle
import signal
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Dict, Iterable, List, Optional, Tuple, Union

import msgspec
import psutil
import zmq

from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.transformers_utils.detokenizer_utils import (
    AnyTokenizer, convert_prompt_ids_to_tokens, detokenize_incrementally)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.utils import get_exception_traceback, make_zmq_socket
from vllm.v1.engine import (EngineCoreAbort, EngineCoreOutput,
                            EngineCoreOutputs, EngineCoreRequest,
                            EngineCoreRequestType)

logger = init_logger(__name__)


@dataclass
class IncrementalDetokenizer:

    # Generation data
    output_text: str
    tokens: List[str]
    token_ids: List[int]

    # Stop strings
    stop: List[str]
    include_stop_str_in_output: bool

    # Metadata for incremental detokenization
    prefix_offset: int
    read_offset: int

    # Parameters for detokenization
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    output_kind: RequestOutputKind

    # TODO: Probably decouple these
    request_id: str
    prompt: Optional[str]
    prompt_token_ids: List[int]

    # Tokenizer for this request
    tokenizer: AnyTokenizer

    # Accounting for stop string buffering
    stop_buffer_length: int
    _last_output_text_offset: int = 0

    @property
    def output_token_ids(self) -> List[int]:
        assert len(self.token_ids) >= len(self.prompt_token_ids)
        return self.token_ids[len(self.prompt_token_ids):]

    @classmethod
    def from_new_request(
        cls,
        tokenizer: AnyTokenizer,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":

        tokens, prefix_offset, read_offset = convert_prompt_ids_to_tokens(
            tokenizer=tokenizer,
            prompt_ids=request.prompt_token_ids,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
        )

        stops = request.sampling_params.stop
        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if stops and not request.sampling_params.include_stop_str_in_output:
            stop_buffer_length = max(len(s) for s in stops) - 1
        else:
            stop_buffer_length = 0

        return cls(
            output_text="",
            tokens=tokens,
            # Detokenizer mutates this list, so need a unique copy.
            # NOTE(Nick): could we take ownership of it though?
            token_ids=request.prompt_token_ids.copy(),
            stop=stops,
            include_stop_str_in_output=request.sampling_params.
            include_stop_str_in_output,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=request.sampling_params.skip_special_tokens,
            spaces_between_special_tokens=request.sampling_params.
            spaces_between_special_tokens,
            output_kind=request.sampling_params.output_kind,
            request_id=request.request_id,
            prompt=request.prompt,
            prompt_token_ids=request.prompt_token_ids,
            tokenizer=tokenizer,
            stop_buffer_length=stop_buffer_length,
        )

    def add_tokens(
        self,
        new_token_ids: List[int],
        finish_reason: Optional[str],
        stop_reason: Optional[Union[int, str, None]],
    ) -> Optional[RequestOutput]:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Update the RequestOutput with the new text.
        """

        # 1) Detokenize the new token ids incrementally.
        # TODO(woosuk): This method becomes very inefficient when the number of
        # new_token_ids is more than 1. We need to optimize this.
        decoded_text = ""
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            (new_tokens, new_decoded_token_text, prefix_offset,
             read_offset) = detokenize_incrementally(
                 tokenizer=self.tokenizer,
                 all_input_ids=self.token_ids,
                 prev_tokens=self.tokens,
                 prefix_offset=self.prefix_offset,
                 read_offset=self.read_offset,
                 skip_special_tokens=self.skip_special_tokens,
                 spaces_between_special_tokens=self.
                 spaces_between_special_tokens,
             )

            self.tokens.extend(new_tokens)
            self.prefix_offset = prefix_offset
            self.read_offset = read_offset
            self.output_text += new_decoded_token_text

            decoded_text += new_decoded_token_text

        # 2) Evaluate stop criteria.
        if self.stop:
            stop = StopChecker.check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(decoded_text),
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_str, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]
                finish_reason = "stop"  # TODO: use constant
                stop_reason = stop_str

        # TODO: handle stop_token_ids here too?

        # 3) Update the RequestOutput object with the new text.
        finished = bool(finish_reason)
        if self.output_kind == RequestOutputKind.FINAL_ONLY \
            and not finished:
            return None

        delta = self.output_kind == RequestOutputKind.DELTA
        output_text = self._get_next_output_text(finished, delta)
        token_ids = new_token_ids if delta else self.output_token_ids

        request_output = RequestOutput.new(
            self.request_id,
            self.prompt,
            self.prompt_token_ids,
            output_text,
            token_ids,
            finished,
        )

        if finished:
            completion_output = request_output.outputs[0]
            completion_output.finish_reason = finish_reason
            completion_output.stop_reason = stop_reason

        return request_output

    def _get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            return self.output_text[:-buffer_length] if buffer_length else (
                self.output_text)
        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


class Detokenizer:

    def __init__(self,
                 tokenizer_name: str,
                 tokenizer_mode: str = "auto",
                 trust_remote_code: bool = False,
                 revision: Optional[str] = None):
        # TODO: once we support LoRA, we should should pass the tokenizer
        # here. We currently have two copies (this + in the LLMEngine).
        self.tokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                       tokenizer_mode=tokenizer_mode,
                                       trust_remote_code=trust_remote_code,
                                       revision=revision)

        # Request id -> IncrementalDetokenizer
        self.request_states: Dict[str, IncrementalDetokenizer] = {}

    def is_request_active(self, request_id: str):
        return request_id in self.request_states

    def get_num_unfinished_requests(self):
        return len(self.request_states)

    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    def abort_requests(
        self,
        request_ids: Iterable[str],
    ) -> None:
        """Remove the request_ids from the Detokenizer."""

        for request_id in request_ids:
            self.request_states.pop(request_id, None)

    def add_request(
        self,
        request: EngineCoreRequest,
    ):
        """Add new request to the Detokenizer."""

        assert (request.request_id not in self.request_states)

        request_state = IncrementalDetokenizer.from_new_request(
            self.tokenizer, request)
        self.request_states[request.request_id] = request_state

    def step(
        self, encore_core_outputs: List[EngineCoreOutput]
    ) -> Tuple[List[RequestOutput], List[str]]:
        """Update state and request the RequestOutputs to the LLMEngine."""

        request_outputs: List[RequestOutput] = []
        requests_to_abort: List[str] = []
        for engine_core_output in encore_core_outputs:
            request_id = engine_core_output.request_id
            detokenizer = self.request_states.get(request_id)
            if detokenizer is None:
                # Ignore output for already-aborted request.
                continue

            # Detokenize and update state.
            request_output = detokenizer.add_tokens(
                new_token_ids=engine_core_output.new_token_ids,
                finish_reason=engine_core_output.finish_reason,
                stop_reason=engine_core_output.stop_reason,
            )

            if request_output is not None:
                # Add to RequestOutputs list.
                request_outputs.append(request_output)

                # Free completed requests.
                if request_output.finished:
                    self.request_states.pop(request_id)
                    if not engine_core_output.finished:
                        requests_to_abort.append(request_id)

        # Return to EngineClient.
        return request_outputs, requests_to_abort


class DetokenizerProc(Detokenizer):
    """ZMQ-wrapper for running Detokenizer in background process."""

    def __init__(
        self,
        *args,
        input_path: str,
        output_path: str,
        to_engine_core_path: str,
        ready_pipe: Connection,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.input_path = input_path
        self.output_path = output_path
        self.to_engine_core_path = to_engine_core_path

        # Send Readiness signal to DetokenizerClient.
        ready_pipe.send({"status": "READY"})

    @staticmethod
    def run_detokenizer(*args, **kwargs):
        """Launch Detokenizer busy loop in background process."""

        # Signal handler used for graceful termination.
        # SystemExit exception is only raised once to allow this and worker
        # processes to terminate without error
        shutdown_requested = False

        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            if not shutdown_requested:
                shutdown_requested = True
                raise SystemExit()

        # Either SIGTERM or SIGINT will terminate the engine_core
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        parent_process = psutil.Process().parent()

        detokenizer = None
        try:
            detokenizer = DetokenizerProc(*args, **kwargs)
            detokenizer.run_busy_loop()

        except SystemExit:
            logger.debug("Detokenizer interrupted.")

        except Exception:
            traceback = get_exception_traceback()
            logger.error("Detokenizer hit an exception: %s", traceback)
            parent_process.send_signal(signal.SIGQUIT)

        finally:
            if detokenizer is not None:
                detokenizer = None

    def _handle_from_llm_engine(
            self,
            request_bytes: bytes,
            to_engine_core: zmq.Socket,  # type: ignore[name-defined]
    ) -> None:
        """Handle inputs from the LLM Engine."""

        req = pickle.loads(request_bytes)

        if isinstance(req, EngineCoreRequest):
            self.add_request(req)
        elif isinstance(req, EngineCoreAbort):
            self.abort_requests(req.request_ids)
        else:
            raise ValueError(f"Unknown type: {req}")

        # Forward to EngineCore.
        to_engine_core.send(request_bytes)

    def _handle_from_engine_core(
        self,
        output_bytes: bytes,
        to_engine_core: zmq.Socket,  # type: ignore[name-defined]
        to_llm_engine: zmq.Socket,  # type: ignore[name-defined]
        decoder: msgspec.msgpack.Decoder,
    ) -> None:
        """Handle Outputs from the EngineCore."""

        # Deserialize the EngineOutput (use msgpack for performance).
        outputs: List[EngineCoreOutput] = decoder.decode(output_bytes).outputs

        # Detokenize.
        request_outputs, requests_to_abort = self.step(outputs)

        # Send request outputs back to LLMEngine.
        if request_outputs:
            # TODO: check whether faster to send this copy free?
            to_llm_engine.send_pyobj(request_outputs)

        # Abort requests that finished due to stop strings in EngineCore.
        if requests_to_abort:
            to_engine_core.send_pyobj(EngineCoreAbort(requests_to_abort))

    def run_busy_loop(self):
        """Core busy loop of the Detokenizer."""

        decoder = msgspec.msgpack.Decoder(EngineCoreOutputs)

        ctx = zmq.Context(io_threads=2)  # type: ignore[attr-defined]
        try:
            input_socket = make_zmq_socket(ctx, self.input_path,
                                           zmq.constants.PULL)
            to_llm_engine = make_zmq_socket(ctx, self.output_path,
                                            zmq.constants.PUSH)
            to_engine_core = make_zmq_socket(ctx, self.to_engine_core_path,
                                             zmq.constants.PUSH)

            while True:
                (msg_type, msg_bytes) = input_socket.recv_multipart()

                # Handle message from LLMEngine (Abort or New Request).
                if msg_type == EngineCoreRequestType.FROM_ENGINE.value:
                    self._handle_from_llm_engine(msg_bytes, to_engine_core)

                # Handle message from EngineCore (EngineCoreOutputs).
                elif msg_type == EngineCoreRequestType.FROM_ENGINE_CORE.value:
                    self._handle_from_engine_core(
                        output_bytes=msg_bytes,
                        to_engine_core=to_engine_core,
                        to_llm_engine=to_llm_engine,
                        decoder=decoder,
                    )
                else:
                    raise ValueError(f"Unknown Message Type: {msg_type}")

        finally:
            ctx.destroy(linger=0)
