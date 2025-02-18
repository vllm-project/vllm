# SPDX-License-Identifier: Apache-2.0
import asyncio
import multiprocessing
import signal
from argparse import Namespace
from contextlib import asynccontextmanager
from functools import partial
from typing import Annotated, Any, AsyncGenerator, AsyncIterator

import numpy as np
import uvloop
from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.client import MQLLMEngineClient
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.entrypoints.launcher import serve_http
from vllm.entrypoints.utils import with_cancellation
from vllm.entrypoints.whisper_server.helper import (load_audio_from_bytes,
                                                    validate_length)
from vllm.logger import init_logger
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import (FlexibleArgumentParser, get_open_zmq_ipc_path,
                        random_uuid, set_ulimit)
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger("vllm.entrypoints.api_server_whisper")

TRANSCRIBE_PROMPT = "<|startoftranscript|><|en|><|transcribe|><|notimestamps|>"
SAMPLING_RATE = 16000
TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()


def format_prompt(waveform: np.ndarray, sampling_rate: int):
    assert sampling_rate == SAMPLING_RATE
    return {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {
                "audio": (waveform, sampling_rate),
            }
        },
        "decoder_prompt": TRANSCRIBE_PROMPT,
    }


class TranscriptionResponse(BaseModel):
    """The response object from the transcription."""
    text: str


class TranscribeFromFile(BaseModel):
    """The audio file (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm)."""
    file: UploadFile

    async def to_prompt(self):
        audio_bytes = await self.file.read()
        audio_data = load_audio_from_bytes(audio_bytes, SAMPLING_RATE)
        validate_length(audio_data)
        return format_prompt(audio_data, SAMPLING_RATE)


class TranscribeFromWaveform(BaseModel):
    """Numpy array of audio waveform to be transcribed."""

    waveform_bytes: UploadFile
    sampling_rate: Annotated[str, Form()]

    async def to_prompt(self):
        waveform = np.frombuffer(await self.waveform_bytes.read(),
                                 dtype=np.float32)
        sampling_rate = int(self.sampling_rate)
        if sampling_rate != SAMPLING_RATE:
            raise ValueError(
                f"Model uses sampling rate of {SAMPLING_RATE}, but got "
                f"sampling_rate = {sampling_rate}.")
        return format_prompt(waveform, SAMPLING_RATE)


@app.post("/generate_from_waveform")
async def generate_from_waveform(data: Annotated[TranscribeFromWaveform,
                                                 Form()],
                                 raw_request: Request):
    """Transcribe from a waveform."""

    prompt = await data.to_prompt()
    return await _generate(prompt, raw_request=raw_request)


@app.post("/generate_from_file")
async def generate_from_file(data: Annotated[TranscribeFromFile,
                                             Form()], raw_request: Request):
    """Transcribe from a file."""

    prompt = await data.to_prompt()
    return await _generate(prompt, raw_request=raw_request)


@with_cancellation
async def _generate(prompt, raw_request: Request) -> Response:

    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=440,
                                     output_kind=RequestOutputKind.DELTA)
    request_id = random_uuid()

    engine = raw_request.app.state.engine
    results_generator = engine.generate(prompt, sampling_params, request_id)

    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            assert len(request_output.outputs) == 1
            chunk = TranscriptionResponse(text=request_output.outputs[0].text)
            response_json = chunk.model_dump_json(exclude_unset=False)
            yield f"data: {response_json}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_results())


@asynccontextmanager
async def build_engine(
    engine_args: AsyncEngineArgs, ) -> AsyncIterator[MQLLMEngineClient]:

    # Select random path for IPC.
    ipc_path = get_open_zmq_ipc_path()
    context = multiprocessing.get_context("spawn")
    engine_alive = multiprocessing.Value('b', True, lock=False)
    engine_process = context.Process(target=run_mp_engine,
                                     args=(engine_args,
                                           UsageContext.OPENAI_API_SERVER,
                                           ipc_path, engine_alive))
    engine_process.start()
    engine_pid = engine_process.pid
    assert engine_pid is not None, "Engine process failed to start."
    logger.info("Started engine process with PID %d", engine_pid)

    # Build RPCClient, which conforms to EngineClient Protocol.
    engine_config = engine_args.create_engine_config()
    build_client = partial(MQLLMEngineClient, ipc_path, engine_config,
                           engine_pid)
    mq_engine_client = await asyncio.get_running_loop().run_in_executor(
        None, build_client)
    try:
        while True:
            try:
                await mq_engine_client.setup()
                break
            except TimeoutError:
                if (not engine_process.is_alive() or not engine_alive.value):
                    raise RuntimeError(
                        "Engine process failed to start. See stack "
                        "trace for the root cause.") from None

        yield mq_engine_client  # type: ignore[misc]
    finally:
        # Ensure rpc server process was terminated
        engine_process.terminate()

        # Close all open connections to the backend
        mq_engine_client.close()

        # Wait for engine process to join
        engine_process.join(4)
        if engine_process.exitcode is None:
            engine_process.kill()


async def run_server(args: Namespace, **uvicorn_kwargs: Any) -> None:
    logger.info("vLLM API server version %s", VLLM_VERSION)
    logger.info("args: %s", args)

    set_ulimit()

    def signal_handler(*_) -> None:
        # Interrupt server on sigterm while initializing
        raise KeyboardInterrupt("terminated")

    signal.signal(signal.SIGTERM, signal_handler)

    engine_args = AsyncEngineArgs.from_cli_args(args)
    async with build_engine(engine_args) as engine:
        # Build App.
        app.state.engine = engine

        shutdown_task = await serve_http(
            app,
            host=args.host,
            port=args.port,
            timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            **uvicorn_kwargs,
        )

    # NB: Await server shutdown only after the backend context is exited
    await shutdown_task


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    uvloop.run(run_server(args))
