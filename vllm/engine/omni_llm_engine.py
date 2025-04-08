#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import asyncio
import collections
import concurrent.futures
import contextlib
import copy
import glob
import multiprocessing
import os
import pickle
import queue
import signal
import time
from contextlib import contextmanager
from threading import Thread
from typing import (Any, AsyncGenerator, Dict, Generator, Iterator, List,
                    Mapping, Optional, Set, Tuple, Union)

import numpy as np
import psutil
import torch
import zmq
import zmq.asyncio
from zmq import Frame  # type: ignore[attr-defined]
from zmq.asyncio import Socket

import vllm.envs as envs
from vllm.config import LoadConfig, ModelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.engine.async_llm_engine import AsyncEngineArgs, AsyncLLMEngine
from vllm.engine.multiprocessing import (ENGINE_DEAD_ERROR, IPC_DATA_EXT,
                                         IPC_HEALTH_EXT, IPC_INPUT_EXT,
                                         IPC_OUTPUT_EXT, REQUEST_OUTPUTS_T,
                                         RPC_REQUEST_T, VLLM_RPC_SUCCESS_STR,
                                         RPCError, RPCStartupRequest,
                                         RPCStartupResponse)
from vllm.engine.multiprocessing.client import (MQClientClosedError,
                                                MQLLMEngineClient)
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.envs import VLLM_RPC_TIMEOUT
from vllm.inputs import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.loader import (
    safetensors_weights_iterator)
from vllm.model_executor.models.qwen2_code2wav_dit import Qwen2Code2wav
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import SamplingParams
from vllm.transformers_utils.config import (get_config,
                                            try_get_generation_config)
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.usage.usage_lib import UsageContext
from vllm.utils import LRUCache, get_open_zmq_ipc_path

POLLING_TIMEOUT_MS = 10000
HEALTHY_RESPONSE = (pickle.dumps(VLLM_RPC_SUCCESS_STR), )

logger = init_logger(__name__)

# omni: use v0 engine.
envs.VLLM_USE_V1 = 0


@contextlib.contextmanager
def envvars(
    key: Union[str, Dict[str, str]],
    value: Union[str, None] = None,
) -> Generator[os._Environ, Any, Any]:
    if isinstance(key, str):
        if value is None:
            items = dict()
        else:
            items: Dict[str, str] = {key: value}
    else:
        items: Dict[str, str] = key
    original_items = dict()
    for k, v in items.items():
        original_items[k] = os.environ.get(k, None)
        os.environ[k] = v

    yield os.environ

    for k, v in original_items.items():
        if v is not None:
            os.environ[k] = v
        else:
            del os.environ[k]


class SynchronizedGenerator(Generator[RequestOutput, None, None]):

    def __init__(
        self,
        generator: AsyncGenerator[RequestOutput, None],
        loop: asyncio.AbstractEventLoop,
    ):
        self._generator = generator
        self._loop = loop

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return asyncio.run_coroutine_threadsafe(
                self._generator.__anext__(),
                self._loop,
            ).result()
        except StopAsyncIteration as e:
            raise StopIteration from e
        except concurrent.futures._base.CancelledError as e:
            raise StopIteration from e

    def send(self, value):
        return self.__next__()

    def throw(self, type, value=None, traceback=None):
        raise StopIteration


# create a new class for code2wav engine request
class Code2WavChunk:

    def __init__(
        self,
        request_id: str,
        voice_type: str,
        code: List[int],
        finished: bool,
        progress: int,
    ) -> None:
        self.request_id = request_id
        self.voice_type = voice_type
        self.code = code
        self.finished = finished
        self.progress = progress
        self.prev_generated = None

    def is_little_chunk(self):
        return self.progress == 0 and self.finished

    def is_first_chunk(self):
        return self.progress == 0 and not self.finished

    def is_intermediate_chunk(self):
        return self.progress > 0 and not self.finished

    def is_last_chunk(self):
        return self.progress != 0 and self.finished

    def __eq__(self, value):
        return self.request_id == value.request_id and \
            self.voice_type == value.voice_type and \
            self.code == value.code and \
            self.finished == value.finished and \
            self.progress == value.progress

    def __str__(self):
        return f"request_id: {self.request_id}, voice_type: {self.voice_type}, code: {self.code}, finished: {self.finished}, progress: {self.progress}"


class ChunkScheduler:

    def __init__(self):
        self.little_chunks = []
        self.first_chunks = []
        self.intermediate_last_chunks = [
        ]  # store is_intermediate_chunk and is_last_chunk
        self.request_progress = collections.defaultdict(
            lambda: -1
        )  # record the last execution progress of each request_id.

    def add_chunk(self, chunk: Code2WavChunk):
        if chunk.is_little_chunk():
            self.little_chunks.append(chunk)
        elif chunk.is_first_chunk():
            self.first_chunks.append(chunk)
        elif chunk.is_intermediate_chunk() or chunk.is_last_chunk():
            self.intermediate_last_chunks.append(chunk)
        else:
            raise ValueError("Unknown chunk type")

    def can_execute(self, chunk: Code2WavChunk) -> bool:
        """Determine whether the chunk meets the execution conditions (i.e., the dependent preceding chunks have been executed)."""
        last_progress = self.request_progress[chunk.request_id]
        return chunk.progress == last_progress + 1

    def schedule(self) -> List[Code2WavChunk]:
        """Select the list of chunks to be executed based on priority and dependency relationships."""
        selected_chunks = []

        # 1. process is_little_chunk
        if self.little_chunks:
            for i, chunk in enumerate(self.little_chunks):
                if self.can_execute(chunk):
                    selected_chunks.append(chunk)
                    del self.little_chunks[i]
                    self.request_progress[chunk.request_id] = chunk.progress
                    return selected_chunks  # Execute only one is_little_chunk at a time.
            # If there are no executable is_little_chunk, continue.

        # 2. process is_first_chunk
        if self.first_chunks:
            sorted_chunks = copy.deepcopy(self.first_chunks)
            for chunk in sorted_chunks:
                if self.can_execute(chunk):
                    selected_chunks.append(chunk)
                    self.first_chunks.remove(chunk)
                    self.request_progress[chunk.request_id] = chunk.progress
                    return selected_chunks

        # 3. process is_intermediate_chunk and is_last_chunk
        if self.intermediate_last_chunks:
            sorted_chunks = copy.deepcopy(self.intermediate_last_chunks)
            first_chunk = sorted_chunks[0]

            if self.can_execute(first_chunk):
                if first_chunk.is_last_chunk():
                    # execute single is_last_chunk
                    selected_chunks.append(first_chunk)
                    self.intermediate_last_chunks.remove(first_chunk)
                    self.request_progress[
                        first_chunk.request_id] = first_chunk.progress
                    return selected_chunks
                elif first_chunk.is_intermediate_chunk():
                    # Execute multiple is_intermediate_chunk, with a maximum of one per request_id.
                    selected_request_ids = set()
                    for chunk in sorted_chunks:
                        if chunk.is_intermediate_chunk() and self.can_execute(
                                chunk):
                            if chunk.request_id not in selected_request_ids:
                                selected_chunks.append(chunk)
                                selected_request_ids.add(chunk.request_id)
                                self.request_progress[
                                    chunk.request_id] = chunk.progress
                                self.intermediate_last_chunks.remove(chunk)
                    return selected_chunks

        # If there are no executable chunks, return empty list
        return selected_chunks


class _MQOmniCode2WavEngine:

    def __init__(
        self,
        ipc_path: str,
        model_path: str,
        enable_torch_compile: bool,
        enable_torch_compile_first_chunk: bool,
        odeint_method: str = "euler",
        odeint_method_relaxed: bool = False,
        batched_chunk: int = 3,
        frequency: str = "50hz",
        device: Union[int, str] = 'cuda',
        log_requests: bool = True,
        code2wav_dynamic_batch: bool = False,
    ) -> None:
        if isinstance(device, int):
            device = f'cuda:{device}'
        self.device = torch.device(device)

        logger.info(
            "Code2WavEngine starting up on device %s, with model %s, method: %s, relaxed: %r",
            self.device, model_path, odeint_method, odeint_method_relaxed)

        if os.path.exists(os.path.join(model_path, 'spk_dict.pt')):
            self.code2wav_conds, self.code2wav_ref_mels = self._load_spk_dict(
                model_path)
        else:
            self.code2wav_conds, self.code2wav_ref_mels = self._load_spk_dict_legacy(
                model_path)

        if 'm02' not in self.code2wav_conds and 'Ethan' in self.code2wav_conds:
            self.code2wav_conds['m02'] = self.code2wav_conds['Ethan']
            self.code2wav_ref_mels['m02'] = self.code2wav_ref_mels['Ethan']
        if 'f030' not in self.code2wav_conds and 'Chelsie' in self.code2wav_conds:
            self.code2wav_conds['f030'] = self.code2wav_conds['Chelsie']
            self.code2wav_ref_mels['f030'] = self.code2wav_ref_mels['Chelsie']

        self.frequency = frequency
        self.code2wav_steps: int = 10
        self.code2wav_bs_mel: int = 24 if frequency == "50hz" else 32
        self.factor: int = 2 if frequency == "50hz" else 4

        dit_model, bigvgan_model = self.load_code2wav_legacy(model_path)
        if not dit_model and not bigvgan_model:
            dit_model, bigvgan_model = self.load_code2wav(model_path)
            with_weight_norm = False
        else:
            with_weight_norm = True

        self.code2wav = Qwen2Code2wav(
            dit_ckpt=dit_model,
            bigvgan_ckpt=bigvgan_model,
            steps=self.code2wav_steps,
            bs_mel=self.code2wav_bs_mel,
            odeint_method=odeint_method,
            odeint_method_relaxed=odeint_method_relaxed,
            batched_chunk=batched_chunk,
            frequency=frequency,
            device=self.device,
            with_weight_norm=with_weight_norm,
        )
        self._torch_compile_model(enable_torch_compile,
                                  enable_torch_compile_first_chunk)

        self.code2wav_y_all = torch.randn(
            1,
            32768,
            80,
            device=self.device,
            dtype=list(self.code2wav_ref_mels.values())[0].dtype)

        self.code2wav_chunk_size: int = self.code2wav.chunk_size
        self.code2wav_future_cache_size: int = self.code2wav.future_cache_size

        # request id -> (progress, prev_generated)
        self.code2wav_progress: Dict[
            str,
            Optional[torch.Tensor]] = collections.defaultdict(lambda: (0, []))
        self.code2wav_progress_error: LRUCache = LRUCache(capacity=1024)

        self.log_requests = log_requests

        self.ctx = zmq.Context()  # type: ignore[attr-defined]

        # Receive input from the client.
        self.input_socket = self.ctx.socket(zmq.constants.PULL)
        self.input_socket.bind(f"{ipc_path}{IPC_INPUT_EXT}")

        # Send output stream back to client.
        self.output_socket = self.ctx.socket(zmq.constants.PUSH)
        self.output_socket.bind(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # Send heartbeats back to client.
        self.heartbeat_socket = self.ctx.socket(zmq.constants.PUSH)
        self.heartbeat_socket.bind(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Error state.
        self._errored_with: Optional[BaseException] = None

        # open batch mode or not
        self.code2wav_dynamic_batch = code2wav_dynamic_batch
        if code2wav_dynamic_batch:
            self.chunk_scheduler = ChunkScheduler()

    def _load_spk_dict(self, model_path):
        code2wav_conds, code2wav_ref_mels = {}, {}

        if not os.path.exists(os.path.join(model_path, 'spk_dict.pt')):
            return code2wav_conds, code2wav_ref_mels

        for key, value in torch.load(os.path.join(model_path,
                                                  'spk_dict.pt')).items():
            code2wav_conds[key] = value["cond"].to(self.device)
            code2wav_ref_mels[key] = value["ref_mel"].to(self.device)
        return code2wav_conds, code2wav_ref_mels

    def _load_spk_dict_legacy(self, model_path):
        code2wav_conds = {
            _MQOmniCode2WavEngine.parse_key(os.path.basename(f), 'spk_emb.npy'):
            torch.tensor(np.load(f)).to(self.device)
            for f in sorted(
                glob.glob(os.path.join(model_path, 'inputs', '*spk_emb.npy')) +
                glob.glob(
                    os.path.join(model_path, 'inputs_sft4spks',
                                 '*spk_emb.npy')))
        }
        code2wav_ref_mels = {
            _MQOmniCode2WavEngine.parse_key(os.path.basename(f), 'ref_mel.npy'):
            torch.tensor(np.load(f)).to(self.device)
            for f in sorted(
                glob.glob(os.path.join(model_path, 'inputs', '*ref_mel.npy')) +
                glob.glob(
                    os.path.join(model_path, 'inputs_sft4spks',
                                 '*ref_mel.npy')))
        }
        return code2wav_conds, code2wav_ref_mels

    def load_code2wav_legacy(self, model_path):
        # code2wav model
        self.dit_model_path = glob.glob(
            os.path.join(model_path, 'dit', 'model_*.pt'))
        self.dit_model_path = self.dit_model_path[
            0] if self.dit_model_path else None
        self.bigvgan_model_path = glob.glob(
            os.path.join(model_path, 'bigvgan', 'g_*'))
        self.bigvgan_model_path = self.bigvgan_model_path[
            0] if self.bigvgan_model_path else None
        return self.dit_model_path, self.bigvgan_model_path

    def load_code2wav(self, model_path):
        import safetensors.torch

        dit_model, bigvgan_model = {}, {}
        safetensors = sorted(
            glob.glob(os.path.join(model_path, '*.safetensors')))
        legacy_weights = False
        for key, value in safetensors_weights_iterator(safetensors,
                                                       use_tqdm_on_load=True):
            legacy_weights = legacy_weights or 'input_embed.spk_encoder.fc.conv.weight' in key
            if legacy_weights:
                break
        for key, value in safetensors_weights_iterator(safetensors,
                                                       use_tqdm_on_load=True):
            if key.startswith('token2wav.code2wav_bigvgan_model.'):
                if 'generator' not in bigvgan_model:
                    bigvgan_model['generator'] = {}
                bigvgan_model['generator'][key.replace(
                    'token2wav.code2wav_bigvgan_model.', '')] = value
            if key.startswith('token2wav.code2wav_dit_model.'):
                key = key.replace('token2wav.code2wav_dit_model.',
                                  'transformer.')
                if key.startswith('transformer.input_embed.spk_encoder'):
                    if legacy_weights:
                        dit_model[key] = value
                    else:
                        dit_model[key.replace('.bias', '.conv.bias').replace(
                            '.weight', '.conv.weight')] = value
                elif '.ff.ff.0.weight' in key or '.ff.ff.0.bias' in key:
                    dit_model[key.replace('.ff.ff.0.weight',
                                          '.ff.ff.0.0.weight').replace(
                                              '.ff.ff.0.bias',
                                              '.ff.ff.0.0.bias')] = value
                elif '.ff.ff.3.weight' in key or '.ff.ff.3.bias' in key:
                    dit_model[key.replace('.ff.ff.3.weight',
                                          '.ff.ff.2.weight').replace(
                                              '.ff.ff.3.bias',
                                              '.ff.ff.2.bias')] = value
                else:
                    dit_model[key] = value
        return dit_model, bigvgan_model

    @staticmethod
    def parse_key(fname: str, key: str) -> str:
        if fname == key:
            return 'default'
        return fname.split('_')[0].lower()

    @property
    def dead_error(self) -> BaseException:
        if self._errored_with is not None:
            return ENGINE_DEAD_ERROR(self._errored_with)
        else:
            return ENGINE_DEAD_ERROR()

    def start(self):
        try:
            try:
                logger.debug("Starting Startup Loop.")
                self.run_startup_loop()
                logger.debug("Starting Engine Loop.")
                self.run_engine_loop()
            except Exception as e:
                logger.exception(repr(e))
        except KeyboardInterrupt:
            logger.debug("Shutting down MQOmniCode2WavEngine.")
        finally:
            logger.debug("MQOmniCode2WavEngine is shut down.")
            self.cleanup()

    def cleanup(self):
        """Cleanup zeromq state on shutdown."""
        # Closes all sockets and destroys context.
        self.ctx.destroy(linger=0)
        del self.code2wav
        if self.code2wav_dynamic_batch:
            del self.chunk_scheduler

    @contextmanager
    def make_data_socket(
            self) -> Iterator[zmq.Socket]:  # type: ignore[name-defined]
        socket = self.ctx.socket(zmq.constants.ROUTER)
        try:
            socket.bind(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    def run_startup_loop(self) -> None:
        """Startup loop for sending data from Engine -> Client."""

        with self.make_data_socket() as socket:
            response: Union[RPCStartupResponse, BaseException]
            try:
                identity, message = socket.recv_multipart(copy=False)
                request: RPCStartupRequest = pickle.loads(message.buffer)

                # Handle the query from the Client.
                if request == RPCStartupRequest.IS_SERVER_READY:
                    response = RPCStartupResponse(False)
            except Exception as e:
                response = e

            socket.send_multipart((identity, pickle.dumps(response)),
                                  copy=False)

    def run_engine_loop(self):
        """Core busy loop of the LLMEngine."""

        while True:
            if self.code2wav_dynamic_batch:
                # Get new inputs, may contains multiple requests with more than one chunk
                # Schedule arrived requests
                self.handle_new_input_batch()
                chunks_list = self.chunk_scheduler.schedule()
                if len(chunks_list) == 0:
                    # Poll until there is work to do.
                    while self.input_socket.poll(
                            timeout=POLLING_TIMEOUT_MS) == 0:
                        # When there's no work, check on engine health and send
                        # health status back to client
                        self._health_check()
                else:
                    # Execute selected requests
                    request_outputs = self.step_batch(chunks_list)
                    # Send outputs back to client
                    self.send_outputs_batch(request_outputs)
                continue
            # Poll until there is work to do.
            while self.input_socket.poll(timeout=POLLING_TIMEOUT_MS) == 0:
                # When there's no work, check on engine health and send
                # health status back to client
                self._health_check()

            # Handle any input from the client.
            request_output = self.handle_new_input()

            # Send request outputs
            if request_output is None:
                continue
            elif isinstance(request_output, Tuple):
                request_id, audio, finished, output_tokens = request_output
                self._send_outputs((request_id, audio, output_tokens))
                if finished:
                    self._send_outputs((request_id, None, output_tokens))
            else:
                logger.warning("Invalid request output: %r", request_output)

    def step_batch(
        self, chunks_list: List[Code2WavChunk]
    ) -> List[Tuple[str, np.ndarray, bool, int]]:
        request_outputs = []
        # filter illegal chunks
        execute_chunk_list = []
        for chunk in chunks_list:
            if chunk.code is None:
                if chunk.request_id in self.code2wav_progress:
                    del self.code2wav_progress[chunk.request_id]
                continue

            if chunk.finished and len(chunk.code) <= 1:
                if chunk.request_id in self.code2wav_progress:
                    del self.code2wav_progress[chunk.request_id]
                request_outputs.append(
                    (chunk.request_id, np.empty(
                        (0),
                        dtype=np.float32), chunk.finished, len(chunk.code)))
                continue

            if self.code2wav_progress_error.get(chunk.request_id) is not None:
                request_outputs.append(
                    (chunk.request_id, np.empty(
                        (0),
                        dtype=np.float32), chunk.finished, len(chunk.code)))
                continue
            execute_chunk_list.append(chunk)

        if len(execute_chunk_list) == 0:
            # No chunk to process, return directly
            return request_outputs

        if len(execute_chunk_list) == 1:
            # Process single chunk
            chunk = chunks_list[0]
            request_id = chunk.request_id
            voice_type = chunk.voice_type
            code = chunk.code
            finished = chunk.finished
            request_outputs.append(
                self.step(request_id, voice_type, code, finished))
            return request_outputs

        # Get execute input
        cond_list, ref_mel_list, codec_list, y_list, prev_generated_list = [], [], [], [], []
        for chunk in execute_chunk_list:
            cond = self.code2wav_conds[chunk.voice_type]
            ref_mel = self.code2wav_ref_mels[chunk.voice_type]
            _, prev_generated = self.code2wav_progress[chunk.request_id]
            # prev_generated = chunk.prev_generated
            # Get codec and y for current progress
            progress, code = chunk.progress, chunk.code
            code = torch.tensor(code, dtype=torch.long,
                                device=self.device).unsqueeze(0)
            start_index = max(
                progress * self.code2wav.chunk_size -
                self.code2wav.past_cache_size, 0)
            end_index = min((progress + 1) * self.code2wav.chunk_size +
                            self.code2wav.future_cache_size,
                            code.shape[1] * self.factor)
            y0 = self.code2wav_y_all[:, start_index:end_index].reshape(
                1, -1, 80).contiguous()
            codec = code[:, start_index // self.factor:end_index //
                         self.factor].reshape(1, -1).contiguous()
            cond_list.append(cond)
            ref_mel_list.append(ref_mel)
            codec_list.append(codec)
            y_list.append(y0)
            prev_generated_list.append(prev_generated)

        cond = torch.cat(cond_list, dim=0)
        ref_mel = torch.cat(ref_mel_list, dim=0)
        codec = torch.cat(codec_list, dim=0)
        y0 = torch.cat(y_list, dim=0)
        prev_generated = torch.cat(prev_generated_list, dim=0)

        generated_batch = self.code2wav.process_chunk_dit_batch(
            cond=cond,
            ref_mel=ref_mel,
            codec=codec,
            y0=y0,
            steps=self.code2wav_steps,
        )

        generated_batch = generated_batch.to(
            torch.float32)[:, self.code2wav.
                           past_cache_size:-self.code2wav.future_cache_size, :]
        if self.frequency == "50hz":
            mel = torch.cat([
                prev_generated[:, -self.code2wav.future_size * 2:, :],
                generated_batch
            ],
                            dim=1)
        else:
            mel = torch.cat([prev_generated, generated_batch], dim=1)

        audio_batch = self.code2wav.process_chunk_bigvgan_batch(mel)
        audio_output = audio_batch[:, self.code2wav.future_size *
                                   240:-self.code2wav.future_size * 240]

        # update code2wav progress
        for idx, chunk in enumerate(execute_chunk_list):
            request_id = chunk.request_id
            progress = chunk.progress
            if chunk.finished:
                del self.code2wav_progress[chunk.request_id]
            elif generated_batch[idx] is not None:
                self.code2wav_progress[request_id] = (
                    progress + 1, generated_batch[idx].unsqueeze(0))

        # fill request_outputs
        for idx, chunk in enumerate(execute_chunk_list):
            request_outputs.append(
                (chunk.request_id, audio_output[idx].detach().cpu().numpy(),
                 chunk.finished, len(chunk.code)))

        return request_outputs

    def handle_new_input_batch(self):
        try:
            recv_chunks = 0
            while self.input_socket.poll(timeout=0) != 0:
                frame: Frame = self.input_socket.recv(copy=False)
                request_id, voice_type, code, finished = pickle.loads(
                    frame.buffer)
                code_len = len(
                    code
                ) * self.factor - self.code2wav.future_cache_size - self.code2wav.chunk_size
                progress = max((code_len + self.code2wav.chunk_size - 1) //
                               self.code2wav.chunk_size, 0)
                new_chunk = Code2WavChunk(request_id, voice_type, code,
                                          finished, progress)
                self.chunk_scheduler.add_chunk(new_chunk)
                recv_chunks += 1

        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)
            raise e

    def send_outputs_batch(self, request_outputs: List[Tuple[str, np.ndarray,
                                                             bool]]):
        for request_output in request_outputs:
            if request_output is None:
                continue
            elif isinstance(request_output, Tuple):
                request_id, audio, finished, output_tokens = request_output
                self._send_outputs((request_id, audio, output_tokens))
                if finished:
                    self._send_outputs((request_id, None, output_tokens))
            else:
                logger.warning("Invalid request output: %r", request_output)
        return

    def handle_new_input(self) -> Optional[Tuple[str, np.ndarray, bool, int]]:
        """Handle new input from the socket"""
        try:
            if self.input_socket.poll(timeout=0) == 0:
                return None

            frame: Frame = self.input_socket.recv(copy=False)
            request_id, voice_type, code, finished = pickle.loads(frame.buffer)
            try:
                return self.step(request_id, voice_type, code, finished)
            except SystemExit:
                raise
            except BaseException as e:
                self._set_errored(e)
                rpc_err = RPCError(request_id=request_id,
                                   is_engine_errored=True,
                                   exception=e)
                self._send_outputs(rpc_err)
                raise e
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)
            raise e

    def step(
        self,
        request_id: str,
        voice_type: str,
        code: List[int],
        finished: bool,
    ) -> Optional[Tuple[str, np.ndarray, bool, int]]:
        if code is None:
            if request_id in self.code2wav_progress:
                del self.code2wav_progress[request_id]
            return None
        output_tokens = len(code)

        if finished and output_tokens <= 1:
            if request_id in self.code2wav_progress:
                del self.code2wav_progress[request_id]
            return request_id, np.empty(
                (0), dtype=np.float32), finished, output_tokens

        if self.code2wav_progress_error.get(request_id) is not None:
            return request_id, np.empty(
                (0), dtype=np.float32), finished, output_tokens

        # list to tensor
        code = torch.tensor(code, dtype=torch.long,
                            device=self.device).unsqueeze(0)

        start_code2wav_chunk_time = time.perf_counter()
        progress, prev_generated = self.code2wav_progress[request_id]

        if progress == 0 and finished:
            process_chunk = self.code2wav.process_little_chunk
        else:
            process_chunk = self.code2wav.process_chunk

        try:
            generated, audio = process_chunk(
                self.code2wav_conds[voice_type],
                self.code2wav_ref_mels[voice_type],
                codec_all=code,
                y_all=self.code2wav_y_all,
                i=progress,
                steps=self.code2wav_steps,
                prev_generated=prev_generated,
                finished=finished,
            )
            audio = audio.detach().cpu().numpy()
        except RuntimeError as e:
            generated, audio = None, np.empty((0), dtype=np.float32)
            self.code2wav_progress_error.put(request_id, True)
            logger.exception("Code2wav request %s chunk %d exception %s",
                             request_id, progress, e)

        end_code2wav_chunk_time = time.perf_counter()
        logger.info("Code2wav request %s chunk %d took %.6f seconds",
                    request_id, progress + 1,
                    end_code2wav_chunk_time - start_code2wav_chunk_time)

        # cleanup
        if finished:
            del self.code2wav_progress[request_id]
        elif generated is not None:
            self.code2wav_progress[request_id] = (progress + 1, generated)
        return request_id, audio, finished, output_tokens

    def _health_check(self):
        # Send unhealthy if engine has already errored
        if self._errored_with is not None:
            self._send_unhealthy(self._errored_with)
        try:
            self._send_healthy()
        except Exception as e:
            self._set_errored(e)
            self._send_unhealthy(e)

    def _send_outputs(self, outputs: REQUEST_OUTPUTS_T):
        """Send List of RequestOutput to RPCClient."""
        if outputs:
            output_bytes = pickle.dumps(outputs)
            self.output_socket.send_multipart((output_bytes, ), copy=False)

    def _send_healthy(self):
        """Send HEALTHY message to RPCClient."""
        if not self.heartbeat_socket.closed:
            self.heartbeat_socket.send_multipart(HEALTHY_RESPONSE, copy=False)

    def _send_unhealthy(self, error: BaseException):
        """Send UNHEALTHY message to RPCClient."""
        if not self.heartbeat_socket.closed:
            error_bytes = pickle.dumps(error)
            self.heartbeat_socket.send_multipart((error_bytes, ), copy=False)

    def _set_errored(self, e: BaseException):
        """Log and set errored status if this is the first issue."""
        if self._errored_with is None:
            self._errored_with = e

    def _torch_compile_model(
        self,
        enable_torch_compile,
        enable_torch_compile_first_chunk,
    ):
        if not enable_torch_compile:
            return

        # compile the bigvgan
        self.code2wav.code2wav_bigvgan_model.vocoder.forward = torch.compile(
            self.code2wav.code2wav_bigvgan_model.vocoder.forward, )
        # compile the dit
        if hasattr(self.code2wav, 'enable_torch_compile'):
            self.code2wav.enable_torch_compile(
                enable_torch_compile_first_chunk)

        logger.info("Code2Wav model torch compiled")

    @staticmethod
    def signal_handler(*_) -> None:
        raise KeyboardInterrupt("Code2WavEngine terminated")

    @staticmethod
    def run_code2wav_engine(
        ipc_path: str,
        model_path: str,
        enable_torch_compile: bool,
        enable_torch_compile_first_chunk: bool,
        odeint_method: str,
        odeint_method_relaxed: bool,
        batched_chunk: int = 3,
        frequency: str = "50hz",
        engine_alive: Any = None,
        device: Union[int, str] = 'cuda',
        code2wav_dynamic_batch: Optional[bool] = False,
    ):
        try:
            engine = _MQOmniCode2WavEngine(
                ipc_path,
                model_path,
                enable_torch_compile=enable_torch_compile,
                enable_torch_compile_first_chunk=
                enable_torch_compile_first_chunk,
                odeint_method=odeint_method,
                odeint_method_relaxed=odeint_method_relaxed,
                batched_chunk=batched_chunk,
                frequency=frequency,
                device=device,
                code2wav_dynamic_batch=code2wav_dynamic_batch,
            )

            signal.signal(signal.SIGTERM, engine.signal_handler)

            engine.start()
        except BaseException as e:
            logger.exception("Code2WavEngine failed with exception: %s", e)
            engine_alive.value = False
            raise e


class _MQCode2WavEngineClient:

    def __init__(self, ipc_path: str, engine_pid: int):
        self.context = zmq.asyncio.Context()
        self._errored_with: Optional[BaseException] = None

        # Send RPCGenerateRequest to the MQLLMEngine.
        self.input_socket: Socket = self.context.socket(zmq.constants.PUSH)
        self.input_socket.connect(f"{ipc_path}{IPC_INPUT_EXT}")

        # Receive streams of RequestOutput from the MQLLMEngine.
        self.output_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.output_socket.connect(f"{ipc_path}{IPC_OUTPUT_EXT}")

        # IPC path for acking heartbeats.
        self.heartbeat_socket: Socket = self.context.socket(zmq.constants.PULL)
        self.heartbeat_socket.connect(f"{ipc_path}{IPC_HEALTH_EXT}")

        # IPC path for the data socket.
        self.data_ipc_path = f"{ipc_path}{IPC_DATA_EXT}"

        # Stream for requests.
        self.output_queue: asyncio.Queue[Tuple[str, Union[
            np.ndarray, BaseException]]] = asyncio.Queue()

        # Loop to handle output of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready so that we can
        # build the Client in an executor to enable clean shutdown.
        self.output_loop: Optional[asyncio.Task] = None

        # Loop to check health of the LLMEngine periodically.
        # Started after the MQLLMEngine is ready.
        self.health_loop: Optional[asyncio.Task] = None
        self._engine_process = psutil.Process(engine_pid)

    @contextmanager
    def get_data_socket(self) -> Iterator[Socket]:
        socket = self.context.socket(zmq.constants.DEALER)
        try:
            socket.connect(self.data_ipc_path)
            yield socket
        finally:
            socket.close(linger=0)

    async def run_heartbeat_loop(self, timeout: int):
        """Background loop that continually checks to ensure the engine process
        is still alive.
        """
        try:
            while True:
                # Check if the engine process is running:
                if not self._engine_process.is_running() or (
                        self._engine_process.status() == psutil.STATUS_ZOMBIE):
                    # NB: is_running() returns True for zombies
                    self._set_errored(
                        RuntimeError(
                            f"Engine process (pid {self._engine_process.pid}) "
                            "died."))
                    break

                if await self.heartbeat_socket.poll(timeout=timeout):
                    # Heartbeat received- check the message
                    await self._check_success(
                        error_message="Heartbeat failed.",
                        socket=self.heartbeat_socket)

                logger.debug("Heartbeat successful.")

        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient check health loop.")

        except psutil.NoSuchProcess:
            self._set_errored(
                RuntimeError(
                    f"Engine process (pid {self._engine_process.pid}) died."))

        except Exception as e:
            self._set_errored(e)

    async def run_output_handler_loop(self):
        """Get RequestOutputs from Engine and stream to Request Queues"""

        try:
            while True:
                await self.run_output_handler()
        except asyncio.CancelledError:
            logger.debug("Shutting down MQLLMEngineClient output handler.")

    async def run_output_handler(self):
        # Poll, checking for ENGINE_DEAD
        while await self.output_socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
            logger.debug("Waiting for output from MQLLMEngine.")

            # If errored, alert all running requests.
            if self.errored:
                logger.error("Engine is errored.")
                return

        message: Frame = await self.output_socket.recv(copy=False)
        request_outputs: Tuple[str, np.ndarray,
                               int] = pickle.loads(message.buffer)

        is_error = isinstance(request_outputs, (BaseException, RPCError))
        if is_error:
            if isinstance(request_outputs, RPCError):
                rpc_error: RPCError = request_outputs
                request_id = rpc_error.request_id
                exception = rpc_error.exception
                is_engine_errored = rpc_error.is_engine_errored
            else:
                # MPLLMEngine should always return an RPCError to
                # the output_socket when an issue arises.
                # If we are here, we are in a bad state and
                # should shut down the server.
                error: BaseException = request_outputs
                logger.error(
                    "Received Exception %s rather than RPCError from "
                    "MPLLMEngine. This should never happen.", error)
                request_id = None
                exception = error
                is_engine_errored = True

            # Set to error state only on engine critical error
            # (and record only the first one)
            if is_engine_errored and not self._errored_with:
                self._errored_with = exception
                # If engine is errored, no matter the type of exception
                # it will no longer be able to receive new requests,
                # therefore we have to inform that the current
                # processed requests failed as well. Send back a dead
                # engine error give this feedback and also give a
                # 'hint' to the server to shutdown next.
                exception = self.dead_error

            if request_id is not None:
                self.output_queue.put_nowait((request_id, exception))
        else:
            # Put each output into the appropriate steam.
            request_id, request_output, output_tokens = request_outputs
            self.output_queue.put_nowait(
                (request_id, request_output, output_tokens))

    async def setup(self):
        """Setup the client before it starts sending server requests."""

        # Start output_loop
        self.output_loop = asyncio.create_task(self.run_output_handler_loop())

        with self.get_data_socket() as socket:
            # Wait until server is ready.
            response = await self._wait_for_server_rpc(socket)

            self.tracing_flag = response.tracing_enabled

            # Start health_loop.
            self.health_loop = asyncio.create_task(
                self.run_heartbeat_loop(timeout=VLLM_RPC_TIMEOUT))

    def close(self):
        """Destroy the ZeroMQ Context."""
        # Close all sockets and terminate the context.
        self.context.destroy(linger=0)

        # Cancel background tasks.
        if self.health_loop is not None:
            self.health_loop.cancel()
        if self.output_loop is not None:
            self.output_loop.cancel()

    def _set_errored(self, e: BaseException):
        logger.exception(repr(e))
        if self._errored_with is None:
            self._errored_with = e

    @staticmethod
    async def _send_get_data_rpc_request(request: RPCStartupRequest,
                                         expected_type: Any,
                                         error_message: str,
                                         socket: Socket) -> Any:
        """Send an RPC request that is expecting data back."""

        # Ping RPCServer with a request.
        await socket.send_multipart((pickle.dumps(request), ), copy=False)

        # Make sure the server responds in time.
        if await socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
            raise TimeoutError("RPCServer didn't reply within "
                               f"{VLLM_RPC_TIMEOUT} ms")

        # Await the data from the Server.
        frame = await socket.recv(copy=False)
        data = pickle.loads(frame.buffer)

        if isinstance(data, BaseException):
            raise data
        elif not isinstance(data, expected_type):
            raise ValueError(error_message)

        return data

    @staticmethod
    async def _send_one_way_rpc_request(request: RPC_REQUEST_T,
                                        socket: Socket):
        """Send one-way RPC request to trigger an action."""

        if socket.closed:
            raise MQClientClosedError()

        await socket.send_multipart((pickle.dumps(request), ))

    async def _await_ack(self, error_message: str, socket: Socket):
        """Await acknowledgement that a request succeeded."""

        if socket.closed:
            raise MQClientClosedError()

        if await socket.poll(timeout=VLLM_RPC_TIMEOUT) == 0:
            raise TimeoutError("MQLLMEngine didn't reply within "
                               f"{VLLM_RPC_TIMEOUT}ms")

        await self._check_success(error_message, socket)

    @staticmethod
    async def _check_success(error_message: str, socket: Socket):
        """Confirm that socket has a VLLM_RPC_SUCCESS_STR message"""

        if socket.closed:
            raise MQClientClosedError()

        frame = await socket.recv(copy=False)
        response = pickle.loads(frame.buffer)

        # Raise error if unsuccessful
        if isinstance(response, BaseException):
            raise response
        elif (not isinstance(response, str)
              or response != VLLM_RPC_SUCCESS_STR):
            raise ValueError(error_message)

    async def check_health(self):
        """
        The check health loop probes the health status of the
        Engine's health every N seconds and sets _errored_with
        if the engine is unhealthy.
        """
        if self._errored_with is not None:
            raise self._errored_with

    @property
    def is_running(self) -> bool:
        return not self.errored

    @property
    def is_stopped(self) -> bool:
        return self.errored

    @property
    def errored(self) -> bool:
        return self._errored_with is not None

    @property
    def dead_error(self) -> BaseException:
        return ENGINE_DEAD_ERROR(self._errored_with)

    async def _wait_for_server_rpc(self, socket: Socket) -> RPCStartupResponse:
        """Wait for the RPCServer to start up."""

        return await self._send_get_data_rpc_request(
            request=RPCStartupRequest.IS_SERVER_READY,
            expected_type=RPCStartupResponse,
            error_message="Unable to start RPC Server",
            socket=socket)

    async def process_chunk(
        self,
        request_id: str,
        voice_type: str,
        code: List[int],
        finished: bool = False,
    ):
        """Send an RPCGenerateRequest to the RPCServer and stream responses."""

        # If already dead, error out.
        if self._errored_with is not None:
            raise ENGINE_DEAD_ERROR(self._errored_with)

        request_bytes = pickle.dumps((request_id, voice_type, code, finished))
        await self.input_socket.send(request_bytes, copy=False)

    async def get_chunk(self) -> Tuple[str, Union[np.ndarray, BaseException]]:
        """Get the next chunk of audio from the engine."""
        return await self.output_queue.get()


class suppress_output_queue_exception(contextlib.AbstractContextManager):
    """Context manager to suppress specified exceptions

    After the exception is suppressed, execution proceeds with the next
    statement following the with statement.

         with suppress(FileNotFoundError):
             os.remove(somefile)
         # Execution still resumes here if the file was already removed
    """

    def __init__(self):
        self._exceptions = (Exception, )

    def __enter__(self):
        pass

    def __exit__(self, exctype, excinst, exctb):
        # Unlike isinstance and issubclass, CPython exception handling
        # currently only looks at the concrete type hierarchy (ignoring
        # the instance and subclass checking hooks). While Guido considers
        # that a bug rather than a feature, it's a fairly hard one to fix
        # due to various internal implementation details. suppress provides
        # the simpler issubclass based semantics, rather than trying to
        # exactly reproduce the limitations of the CPython interpreter.
        #
        # See http://bugs.python.org/issue12029 for more details
        if exctype is not None:
            logger.exception("Exception in output queue: %s, %r, %r", exctype,
                             excinst, exctb)
        return exctype is not None and issubclass(exctype, self._exceptions)


class OmniLLMEngine:

    def __init__(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: Union[Mapping[str, AsyncEngineArgs],
                                  AsyncEngineArgs] = None,
        code2wav_model_path: Optional[str] = None,
        code2wav_data_parallelism: int = 1,
        code2wav_enable_torch_compile: bool = False,
        code2wav_enable_torch_compile_first_chunk: bool = False,
        code2wav_odeint_method: str = "rk4",
        code2wav_odeint_method_relaxed: bool = False,
        code2wav_batched_chunk: int = None,
        code2wav_frequency: str = '50hz',
        thinker_visible_devices: Optional[List[int]] = None,
        talker_visible_devices: Optional[List[int]] = None,
        code2wav_visible_devices: Optional[List[int]] = None,
        code2wav_dynamic_batch: Optional[bool] = False,
    ):
        self.thinker_engine_args = thinker_engine_args
        if talker_engine_args is None:
            talker_engine_args = {}
        if not isinstance(talker_engine_args, dict):
            talker_engine_args = {
                'default': talker_engine_args,
            }
        self.talker_engine_args = talker_engine_args
        self.code2wav_model_path = code2wav_model_path
        self.code2wav_enable_torch_compile = code2wav_enable_torch_compile
        self.code2wav_enable_torch_compile_first_chunk = code2wav_enable_torch_compile_first_chunk
        self.code2wav_data_parallelism = code2wav_data_parallelism
        self.code2wav_odeint_method = code2wav_odeint_method
        self.code2wav_odeint_method_relaxed = code2wav_odeint_method_relaxed
        if code2wav_batched_chunk is None:
            if code2wav_frequency == "50hz":
                code2wav_batched_chunk = 2
            else:
                code2wav_batched_chunk = 1
        self.code2wav_batched_chunk = code2wav_batched_chunk
        self.code2wav_frequency = code2wav_frequency

        self.thinker_visible_devices = thinker_visible_devices
        self.talker_visible_devices = talker_visible_devices
        self.code2wav_visible_devices = code2wav_visible_devices

        if (self.thinker_visible_devices
                and not isinstance(self.thinker_visible_devices, list)):
            self.thinker_visible_devices = [self.thinker_visible_devices]
        if (self.talker_visible_devices
                and not isinstance(self.talker_visible_devices, list)):
            self.talker_visible_devices = [self.talker_visible_devices]
        if (self.code2wav_visible_devices
                and not isinstance(self.code2wav_visible_devices, list)):
            self.code2wav_visible_devices = [self.code2wav_visible_devices]

        # Enable hidden state caching for the thinker engine.
        self.thinker_engine_args.enable_hidden_state_caching = True

        if not code2wav_model_path:
            self.code2wav_data_parallelism = 0

        # Start RPCServer in separate process (holds the LLMEngine).
        # the current process might have CUDA context,
        # so we need to spawn a new process
        self.context = multiprocessing.get_context("spawn")

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._async_loop_thread = Thread(target=self._start_async_loop,
                                         daemon=True)
        self._async_loop_thread.start()

        # thinker: async engine
        self.thinker_engine = self._start_thinker_engine(
            thinker_engine_args, self.thinker_visible_devices)
        self.device = self.thinker_engine.engine.device_config.device

        # thinker state
        # request_id -> prompt_embeds
        self.thinker_prompt_embeds: Dict[str, torch.Tensor] = dict()
        # request_id -> listener thread
        self.thinker_listener: Dict[str, Thread] = dict()
        # talker type -> (request_id, voice_type, thinker request_output)
        self.thinker_outputs: Dict[
            str,
            queue.Queue[Tuple[str, str, RequestOutput]],
        ] = collections.defaultdict(queue.Queue)
        self.thinker_finished: Set[str] = set()

        # thinker model config
        self.thinker_config = get_config(
            thinker_engine_args.model,
            trust_remote_code=thinker_engine_args.trust_remote_code)
        self.thinker_generation_config = try_get_generation_config(
            thinker_engine_args.model,
            trust_remote_code=thinker_engine_args.trust_remote_code,
        )

        # thinker tokenizer
        self.thinker_tokenizer = get_tokenizer(
            thinker_engine_args.tokenizer,
            tokenizer_mode=thinker_engine_args.tokenizer_mode,
            trust_remote_code=thinker_engine_args.trust_remote_code,
            revision=thinker_engine_args.revision,
            download_dir=thinker_engine_args.download_dir,
        )

        # talker: async mp engine(s)
        self._talker_pids = []
        self._talker_processes = []
        self._talker_engine_clients: Dict[str, MQLLMEngineClient] = {}
        self._talker_engine_threads: Dict[str, Thread] = {}
        for i, (talker_type,
                engine_args) in enumerate(talker_engine_args.items()):
            # use thinker as talkers tokenizer
            if thinker_engine_args.tokenizer:
                engine_args.tokenizer = thinker_engine_args.tokenizer
            else:
                engine_args.tokenizer = thinker_engine_args.model
            if thinker_engine_args.processor:
                engine_args.processor = thinker_engine_args.processor
            else:
                engine_args.processor = thinker_engine_args.model
            if not engine_args.hf_overrides:
                engine_args.hf_overrides = {}

            if hasattr(self.thinker_config, 'audio_config'):
                engine_args.hf_overrides[
                    'audio_config'] = self.thinker_config.audio_config
            if hasattr(self.thinker_config, 'vision_config'):
                engine_args.hf_overrides[
                    'vision_config'] = self.thinker_config.vision_config

            # Adapt the qwen2_5_omni_model
            if 'Qwen2_5OmniModel' in self.thinker_config.architectures:
                engine_args.hf_overrides['architectures'] = [
                    'Qwen2_5OmniTalkerModel'
                ]

            if self.talker_visible_devices:
                visible_devices = [self.talker_visible_devices[i]]
            else:
                visible_devices = None

            (
                pid,
                process,
                engine_client,
                thread,
            ) = self._start_talker_engine(talker_type, engine_args,
                                          visible_devices)
            self._talker_pids.append(pid)
            self._talker_processes.append(process)
            self._talker_engine_clients[talker_type] = engine_client
            self._talker_engine_threads[talker_type] = thread

        if talker_engine_args:
            # talker model config
            default_talker_args = next(iter(talker_engine_args.values()))
            self.talker_config = get_config(
                default_talker_args.model,
                trust_remote_code=default_talker_args.trust_remote_code)
            self.talker_generation_config = try_get_generation_config(
                default_talker_args.model,
                trust_remote_code=default_talker_args.trust_remote_code,
            )
            # init embeddings for special tokens
            self._init_special_tokens_embeddings(thinker_engine_args,
                                                 default_talker_args)

        # talker request state
        self.talker_requests: Dict[str, Tuple[PromptType,
                                              SamplingParams]] = dict()
        self.talker_listener: Dict[str, Thread] = dict()
        self.talker_finished: Set[str] = set()
        self.talker_inputs: Dict[
            str,
            queue.Queue[Optional[Tuple[torch.Tensor, bool]]],
        ] = dict()

        # code2wav: async mp engine(s)
        self._code2wav_pids = []
        self._code2wav_processes = []
        self._code2wav_engine_clients: List[_MQCode2WavEngineClient] = []
        for i in range(self.code2wav_data_parallelism):
            assert code2wav_model_path, "code2wav_model_path is required"

            (
                pid,
                process,
                engine_client,
            ) = self._start_code2wav_engine(
                get_open_zmq_ipc_path(),
                code2wav_model_path,
                code2wav_enable_torch_compile,
                code2wav_enable_torch_compile_first_chunk,
                code2wav_odeint_method,
                code2wav_odeint_method_relaxed,
                code2wav_batched_chunk,
                code2wav_frequency,
                visible_devices=([
                    self.code2wav_visible_devices[i % len(
                        self.code2wav_visible_devices)]
                ] if self.code2wav_visible_devices else None),
                code2wav_dynamic_batch=code2wav_dynamic_batch,
            )
            self._code2wav_pids.append(pid)
            self._code2wav_processes.append(process)
            self._code2wav_engine_clients.append(engine_client)

        # code2wav voice types
        if self.code2wav_model_path:
            self.code2wav_voice_types = self._load_code2wav_voice_types_legacy(
            )
            if not self.code2wav_voice_types:
                self.code2wav_voice_types = self._load_code2wav_voice_types()
        else:
            self.code2wav_voice_types = []
        # for prefix-caching
        if self.code2wav_voice_types:
            self.code2wav_voice_types.append('prefix_caching')
        logger.info("Code2Wav voice types: %s", self.code2wav_voice_types)

        # code2wav parameters
        self.code2wav_bs_mel = 24 if code2wav_frequency == "50hz" else 32
        self.code2wav_future_cache_size = self.code2wav_bs_mel * 1
        self.code2wav_chunk_size = self.code2wav_bs_mel * self.code2wav_batched_chunk

        # code2wav result listener
        self._code2wav_listener_thread = []
        for i in range(self.code2wav_data_parallelism):
            self._code2wav_listener_thread.append(
                Thread(target=self._listen_code2wav, args=(i, ), daemon=True))
            self._code2wav_listener_thread[i].start()

        # request id -> [union(request_output, (waveform, finished))]
        self.output_queue: Dict[str, queue.Queue[Union[RequestOutput, Tuple[
            np.ndarray, int]]]] = collections.defaultdict(queue.Queue)

    def _load_code2wav_voice_types_legacy(self):
        return [
            _MQOmniCode2WavEngine.parse_key(os.path.basename(f), 'spk_emb.npy')
            for f in sorted(
                glob.glob(
                    os.path.join(self.code2wav_model_path, 'inputs',
                                 '*spk_emb.npy')) +
                glob.glob(
                    os.path.join(self.code2wav_model_path, 'inputs_sft4spks',
                                 '*spk_emb.npy')))
        ]

    def _load_code2wav_voice_types(self):
        voice_type_mapping = {
            'Ethan': 'm02',
            'Chelsie': 'f030',
        }
        code2wav_voice_types = []

        if not os.path.exists(
                os.path.join(self.code2wav_model_path, 'spk_dict.pt')):
            return code2wav_voice_types

        for key, value in torch.load(
                os.path.join(self.code2wav_model_path, 'spk_dict.pt')).items():
            code2wav_voice_types.append(voice_type_mapping.get(key, key))
        return code2wav_voice_types

    def _start_async_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _start_thinker_engine(
        self,
        engine_args: AsyncEngineArgs,
        visible_devices: Optional[List[int]] = None,
    ) -> AsyncLLMEngine:
        if visible_devices:
            device_ctx = envvars(
                'CUDA_VISIBLE_DEVICES',
                ','.join(str(device) for device in visible_devices))
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            return AsyncLLMEngine.from_engine_args(
                engine_args,
                start_engine_loop=True,
            )

    def _start_talker_engine(
        self,
        talker_type: str,
        engine_args: AsyncEngineArgs,
        visible_devices: Optional[List[int]] = None,
    ) -> Tuple[int, multiprocessing.Process, MQLLMEngineClient, Thread]:
        ipc_path = get_open_zmq_ipc_path()
        logger.info(
            "Multiprocessing frontend to use %s for IPC Path for model %s.",
            ipc_path, engine_args.model)

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        if visible_devices:
            device_ctx = envvars(
                'CUDA_VISIBLE_DEVICES',
                ','.join(str(device) for device in visible_devices))
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            usage_context = UsageContext.API_SERVER
            vllm_config = engine_args.create_engine_config(
                usage_context=usage_context)
            engine_process = self.context.Process(
                target=run_mp_engine,
                args=(vllm_config, usage_context, ipc_path,
                      engine_args.disable_log_stats,
                      engine_args.disable_log_requests, engine_alive))
            engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        # Build RPCClient, which conforms to EngineClient Protocol.
        engine_config = engine_args.create_engine_config()
        mq_engine_client: MQLLMEngineClient = MQLLMEngineClient(
            ipc_path, engine_config, engine_pid)

        while True:
            try:
                asyncio.run_coroutine_threadsafe(
                    mq_engine_client.setup(),
                    self._loop,
                ).result()
                break
            except TimeoutError:
                if (not engine_process.is_alive() or not engine_alive.value):
                    raise RuntimeError(
                        "Engine process failed to start. See stack "
                        "trace for the root cause.") from None

        logger.info("Engine process with PID %d has started", engine_pid)

        thread = Thread(
            target=self._thinker_to_talker_loop,
            args=(self.thinker_outputs[talker_type], mq_engine_client),
            daemon=True,
        )
        thread.start()
        return engine_pid, engine_process, mq_engine_client, thread

    def _start_code2wav_engine(
        self,
        ipc_path: str,
        model_path: str,
        enable_torch_compile: bool,
        enable_torch_compile_first_chunk: bool,
        odeint_method: str = "euler",
        odeint_method_relaxed: bool = False,
        batched_chunk: int = 2,
        frequency: str = "50hz",
        device: Union[int, str] = 'cuda',
        visible_devices: Optional[List[int]] = None,
        code2wav_dynamic_batch: Optional[bool] = False,
    ) -> Tuple[int, multiprocessing.Process, _MQCode2WavEngineClient]:
        logger.info(
            "Multiprocessing frontend to use %s for IPC Path for model %s.",
            ipc_path, model_path)

        # The Process can raise an exception during startup, which may
        # not actually result in an exitcode being reported. As a result
        # we use a shared variable to communicate the information.
        engine_alive = multiprocessing.Value('b', True, lock=False)
        if visible_devices:
            device_ctx = envvars(
                'CUDA_VISIBLE_DEVICES',
                ','.join(str(device) for device in visible_devices))
        else:
            device_ctx = contextlib.nullcontext()
        with device_ctx:
            engine_process = self.context.Process(
                target=_MQOmniCode2WavEngine.run_code2wav_engine,
                args=(ipc_path, model_path, enable_torch_compile,
                      enable_torch_compile_first_chunk, odeint_method,
                      odeint_method_relaxed, batched_chunk, frequency,
                      engine_alive, device, code2wav_dynamic_batch))
            engine_process.start()
        engine_pid = engine_process.pid
        assert engine_pid is not None, "Engine process failed to start."
        logger.info("Started engine process with PID %d", engine_pid)

        mq_engine_client: _MQCode2WavEngineClient = _MQCode2WavEngineClient(
            ipc_path, engine_pid)

        while True:
            try:
                asyncio.run_coroutine_threadsafe(
                    mq_engine_client.setup(),
                    self._loop,
                ).result()
                break
            except TimeoutError:
                if (not engine_process.is_alive() or not engine_alive.value):
                    raise RuntimeError(
                        "Engine process failed to start. See stack "
                        "trace for the root cause.") from None

        logger.info("Engine process with PID %d has started", engine_pid)
        return engine_pid, engine_process, mq_engine_client

    def _shutdown_mp_engine(
        self,
        engine_pid: int,
        engine_process: multiprocessing.Process,
        mq_engine_client: MQLLMEngineClient,
    ):
        engine_process.terminate()
        mq_engine_client.close()
        engine_process.join(5)
        if engine_process.is_alive():
            logger.warning("Engine process did not terminate in time.")
            engine_process.kill()
        logger.info("Engine process with PID %d has stopped", engine_pid)

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self._closed = True
        # shutdown thinker engine
        self.thinker_engine.shutdown_background_loop()
        # shutdown talker engines
        for pid, process, engine_client in zip(
                self._talker_pids,
                self._talker_processes,
                self._talker_engine_clients.values(),
        ):
            self._shutdown_mp_engine(pid, process, engine_client)
        # shutdown code2wav engines
        for pid, process, engine_client in zip(
                self._code2wav_pids,
                self._code2wav_processes,
                self._code2wav_engine_clients,
        ):
            self._shutdown_mp_engine(pid, process, engine_client)

    def _load_model_embedding(
            self,
            engine_args: EngineArgs,
            kind: str,  # thinker or talker
    ) -> torch.nn.Embedding:
        model_loader = get_model_loader(
            LoadConfig(
                load_format=engine_args.load_format,
                download_dir=engine_args.download_dir,
                model_loader_extra_config=engine_args.
                model_loader_extra_config,
                ignore_patterns=engine_args.ignore_patterns,
            ))
        for key, value in model_loader._get_all_weights(
                ModelConfig(
                    model=engine_args.model,
                    task=engine_args.task,
                    tokenizer=engine_args.tokenizer,
                    tokenizer_mode=engine_args.tokenizer_mode,
                    processor=engine_args.processor,
                    trust_remote_code=engine_args.trust_remote_code,
                    dtype=engine_args.dtype,
                    seed=engine_args.seed,
                ),
                model=None,
        ):
            if key in [
                    "model.embed_tokens.weight",
                    *(["thinker.model.embed_tokens.weight"]
                      if kind == 'thinker' else []),
                    *(["talker.model.embed_tokens.weight"]
                      if kind == 'talker' else []),
            ]:
                return torch.nn.Embedding.from_pretrained(value).to(
                    self.device)
        raise ValueError(
            "Embedding weight 'model.embed.weight' not found in model weights."
        )

    def _init_special_tokens_embeddings(
        self,
        thinker_engine_args: AsyncEngineArgs,
        talker_engine_args: AsyncEngineArgs,
    ):
        # thinker and talker embeddings
        self.thinker_embedding = self._load_model_embedding(
            thinker_engine_args, 'thinker')
        self.talker_embedding = self._load_model_embedding(
            talker_engine_args, 'talker')

        # embed_text_bos_token
        self.tts_text_spk_token_ids = {
            # M02：我是个会说标准普通话、带部分北方口音的男声
            'm02': 151870,
            'Ethan': 151870,

            # F030：我是你的二次元虚拟女友
            'f030': 151872,
            'Chelsie': 151872,
        }
        self.default_tts_text_spk_type = list(
            self.tts_text_spk_token_ids.keys())[0]
        self.tts_text_spk_token_ids['prefix_caching'] = 151870

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        self.embed_text_bos_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_start_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_text_spk_tokens = {
            key:
            self.thinker_embedding(
                torch.tensor(
                    [value],
                    dtype=torch.long,
                    device=self.device,
                ))
            for key, value in self.tts_text_spk_token_ids.items()
        }
        self.embed_text_eos_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_end_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_text_pad_token = self.thinker_embedding(
            torch.tensor(
                [talker_hf_config.tts_text_pad_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_codec_bos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_start_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_codec_eos_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_end_token_id],
                dtype=torch.long,
                device=self.device,
            ))
        self.embed_codec_pad_token = self.talker_embedding(
            torch.tensor(
                [talker_hf_config.tts_codec_pad_token_id],
                dtype=torch.long,
                device=self.device,
            ))

    def _get_embed_text_spk_token(self, voice_type: str):
        if voice_type not in self.embed_text_spk_tokens:
            return self.embed_text_bos_token
        return self.embed_text_spk_tokens[voice_type]

    def _get_text_spk_token_id(self, voice_type: str):
        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        if voice_type not in self.tts_text_spk_token_ids:
            return talker_hf_config.tts_text_start_token_id
        return self.tts_text_spk_token_ids[voice_type]

    def _listen_thinker(
        self,
        request_id: str,
        voice_type: str,
        generator: AsyncGenerator[RequestOutput, None],
        thinker_outputs: queue.Queue[Tuple[str, str, RequestOutput]],
    ):
        include_voice = not self._ignore_voice(voice_type)

        def _listen():
            last_output = None
            for output in SynchronizedGenerator(generator, self._loop):
                if include_voice and self.talker_engine_args:
                    if output.finished:
                        thinker_output = copy.deepcopy(output)
                        if len(output.outputs[0].token_ids) > 1:
                            thinker_output.finished = False
                    else:
                        thinker_output = output
                    thinker_outputs.put(
                        (request_id, voice_type, thinker_output))
                last_output = output
                with suppress_output_queue_exception():
                    self.output_queue[request_id].put(output)
            if not include_voice or not self.talker_engine_args:
                with suppress_output_queue_exception():
                    self.output_queue[request_id].put(None)
                    self.output_queue.pop(request_id)
            logger.info(
                "Thinker request %s finished, reason: (%r, %r), outputs: %r, output tokens: %r",
                request_id, last_output.outputs[0].finish_reason,
                last_output.outputs[0].stop_reason,
                last_output.outputs[0].text, last_output.outputs[0].token_ids)

            if include_voice and self.talker_engine_args and len(
                    last_output.outputs[0].token_ids) > 1:
                # add two special tokens for talker model.
                special_token_embeds = [
                    self.embed_text_eos_token,
                    self.embed_text_pad_token,
                ]
                for i, token_embed in enumerate(special_token_embeds):
                    thinker_outputs.put(
                        (request_id, voice_type,
                         RequestOutput(
                             request_id=request_id,
                             prompt=last_output.prompt,
                             prompt_token_ids=last_output.prompt_token_ids,
                             prompt_logprobs=last_output.prompt_logprobs,
                             outputs=[
                                 CompletionOutput(
                                     index=0,
                                     text='',
                                     token_ids=[],
                                     cumulative_logprob=None,
                                     logprobs=None,
                                     prompt_embeds=token_embed,
                                 )
                             ],
                             finished=i == len(special_token_embeds) - 1,
                         )))

            with contextlib.suppress(Exception):
                del self.thinker_listener[request_id]

        self.thinker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.thinker_listener[request_id].start()

    def _listen_talker(
        self,
        talker_client: MQLLMEngineClient,
        request_id: str,
        voice_type: str,
        generator: AsyncGenerator[RequestOutput, None],
        output_queue: queue.Queue[Union[RequestOutput, np.ndarray]],
        talker_input_queue: queue.Queue[Optional[Tuple[torch.Tensor, bool]]],
    ):
        include_code2wav = not self._ignore_code2wav(voice_type)

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        def _listen():
            last_output, finished = None, False
            for output in SynchronizedGenerator(generator, self._loop):
                if not self._code2wav_engine_clients or not include_code2wav:
                    output_queue.put(output)
                elif last_output is None:
                    if output.finished:
                        self._talker_to_code2wav(output.request_id, voice_type,
                                                 [], True)
                else:
                    last_code = list(last_output.outputs[0].token_ids)
                    if output.finished:
                        curr_code = list(output.outputs[0].token_ids)
                        if curr_code[-1] in [
                                talker_hf_config.tts_codec_end_token_id,
                                talker_hf_config.tts_codec_start_token_id
                        ]:
                            self._talker_to_code2wav(output.request_id,
                                                     voice_type, last_code,
                                                     False)
                            self._talker_to_code2wav(output.request_id,
                                                     voice_type,
                                                     curr_code[:-1], True)
                        else:
                            self._talker_to_code2wav(output.request_id,
                                                     voice_type, last_code,
                                                     False)
                            self._talker_to_code2wav(output.request_id,
                                                     voice_type, curr_code,
                                                     True)
                    else:
                        self._talker_to_code2wav(output.request_id, voice_type,
                                                 last_code, False)
                last_output = output

                if not finished and not output.finished:
                    prompt_embeds, finished = talker_input_queue.get()
                    if prompt_embeds is not None:
                        asyncio.run_coroutine_threadsafe(
                            talker_client.resume_request(
                                request_id, prompt_embeds, finished),
                            self._loop,
                        ).result()

            if not self._code2wav_engine_clients or not include_code2wav:
                output_queue.put(None)

            logger.info(
                "Talker request %s finished, reason: (%r, %r), output tokens: %r",
                request_id, last_output.outputs[0].finish_reason,
                last_output.outputs[0].stop_reason,
                last_output.outputs[0].token_ids)

            with contextlib.suppress(Exception):
                del self.talker_listener[request_id]
                del self.talker_inputs[request_id]

        self.talker_listener[request_id] = Thread(target=_listen, daemon=True)
        self.talker_listener[request_id].start()

    def _listen_code2wav(self, code2wav_engine_client_id):
        while True:
            try:
                request_id, audio, output_tokens = asyncio.run_coroutine_threadsafe(
                    self._code2wav_engine_clients[code2wav_engine_client_id].
                    get_chunk(),
                    self._loop,
                ).result()
                if audio is None:
                    logger.info("Code2Wav request %s finished", request_id)
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(None)
                        self.output_queue.pop(request_id)
                else:
                    with suppress_output_queue_exception():
                        self.output_queue[request_id].put(
                            (audio, output_tokens))
            except:
                logger.exception("Error in code2wav listener")

    def _thinker_to_talker_loop(
        self,
        thinker_outputs: queue.Queue[Tuple[str, str, RequestOutput]],
        talker_client: MQLLMEngineClient,
    ):
        while True:
            try:
                request_id, voice_type, output = thinker_outputs.get()
                self._thinker_to_talker(request_id, voice_type, output,
                                        talker_client)
            except:
                logger.exception("Error in thinker to talker loop")

    def _thinker_to_talker(
        self,
        request_id: str,
        voice_type: str,
        output: RequestOutput,
        talker_client: MQLLMEngineClient,
    ):
        # structure of prompt tokens, embeddings and thinker_reply_part:
        #
        #   tokens: [input_tokens] + [codec_pad_token] + [codec_bos_token]
        #   embeddings: [input_embeds] + [text_bos_token] + [thinker_reply_part[0]]
        #   thinker_reply_part: [thinker_reply_part[1:]] + [text_eos_token] + [text_pad_token]

        if output == None:
            asyncio.run_coroutine_threadsafe(
                talker_client.abort(request_id),
                self._loop,
            ).result()
            with suppress_output_queue_exception():
                self.output_queue[request_id].put(None)
                self.output_queue.pop(request_id)
            return

        if len(output.outputs[0].token_ids) == 1 and output.finished:
            # don't involve talker model.
            with suppress_output_queue_exception():
                self.output_queue[request_id].put(None)
                self.output_queue.pop(request_id)
            return

        output_prompt_embeds = output.outputs[0].prompt_embeds.to(self.device)

        if len(output.outputs[0].token_ids) == 1:
            self.thinker_prompt_embeds[
                output.request_id] = output_prompt_embeds
            return

        talker_hf_config = self.talker_config
        if hasattr(talker_hf_config, 'talker_config'):
            talker_hf_config = talker_hf_config.talker_config

        if len(output.outputs[0].token_ids) == 2:
            # issue request
            prompt_embeds = torch.cat([
                self.thinker_prompt_embeds.pop(output.request_id),
                self._get_embed_text_spk_token(voice_type) +
                self.embed_codec_pad_token,
                output_prompt_embeds + self.embed_codec_bos_token,
            ],
                                      dim=0)
            prompt, sampling_params = self.talker_requests.pop(request_id)
            if isinstance(prompt, str):
                prompt_token_ids = self.thinker_tokenizer.encode(prompt)
            else:
                if 'prompt_token_ids' in prompt:
                    prompt_token_ids = prompt['prompt_token_ids']
                else:
                    prompt_token_ids = self.thinker_tokenizer.encode(
                        prompt['prompt'])
            prompt_token_ids += [
                # input_text_ids:
                # the first token should be: self._get_text_spk_token_id(voice_type),
                # but it will be ignored in the detokenize-tokenize round
                # during preprocessing, so we use tts_codec_pad_token_id instead.
                # self._get_text_spk_token_id(voice_type),
                talker_hf_config.tts_codec_pad_token_id,
                output.outputs[0].token_ids[0],

                # input_ids (will be replaced in model_runner):
                # talker_hf_config.tts_codec_pad_token_id,
                # talker_hf_config.tts_codec_start_token_id,
            ]
            generator = talker_client.generate(
                request_id=output.request_id,
                prompt=TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    prompt_embeds=prompt_embeds,
                    multi_modal_data=prompt.get('multi_modal_data', None),
                    mm_processor_kwargs=prompt.get('mm_processor_kwargs',
                                                   None),
                ),
                sampling_params=sampling_params,
                resumable=not bool(output.finished),
            )
            self.talker_inputs[output.request_id] = queue.Queue()
            self._listen_talker(
                talker_client,
                output.request_id,
                voice_type,
                generator,
                self.output_queue[output.request_id],
                self.talker_inputs[output.request_id],
            )
        elif output.request_id in self.talker_inputs:
            self.talker_inputs[output.request_id].put(
                (output_prompt_embeds, output.finished))
        else:
            logger.warning("Warning: talker has been over for request_id: %s",
                           request_id)

    def _talker_to_code2wav(
        self,
        request_id: str,
        voice_type: str,
        code: List[int],
        finished: bool,
    ):
        if finished:
            logger.info("Talker request %s finished with %d tokens",
                        request_id, len(code))
        chunk_code_length = len(code) * (2 if self.code2wav_frequency == "50hz"
                                         else
                                         4) - self.code2wav_future_cache_size
        if (chunk_code_length > 0 and
                chunk_code_length % self.code2wav_chunk_size == 0) or finished:

            code2wav_engine_id = 0
            if self.code2wav_data_parallelism > 0:
                request_hash = hash(request_id) if len(
                    request_id) > 1 else ord(request_id)
                code2wav_engine_id = request_hash % self.code2wav_data_parallelism
            asyncio.run_coroutine_threadsafe(
                self._code2wav_engine_clients[code2wav_engine_id].
                process_chunk(request_id=request_id,
                              voice_type=voice_type,
                              code=code,
                              finished=finished),
                self._loop,
            ).result()

    def add_request(
        self,
        request_id: str,
        prompt: Optional[PromptType] = None,
        params: Optional[Union[SamplingParams, PoolingParams]] = None,
        lora_request: Optional[LoRARequest] = None,
        trace_headers: Optional[Mapping[str, str]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
        talker_params: Optional[Union[SamplingParams, PoolingParams]] = None,
        *,
        inputs: Optional[PromptType] = None,  # DEPRECATED
        talker_type: str = "default",
        voice_type: str = "default",
    ) -> queue.Queue[Union[RequestOutput, np.ndarray]]:
        # normalize voice_type
        if voice_type:
            voice_type = voice_type.lower()

        if (self.talker_engine_args and voice_type
                and (not self._ignore_voice(voice_type))
                and self.code2wav_voice_types
                and voice_type not in self.code2wav_voice_types):
            raise ValueError(
                f"Voice type '{voice_type}' not found in available voice types: {self.code2wav_voice_types}"
            )

        if voice_type == "default":
            voice_type = self.default_tts_text_spk_type

        if not params:
            params = SamplingParams(
                top_k=self.thinker_generation_config.top_k,
                top_p=self.thinker_generation_config.top_p,
                temperature=self.thinker_generation_config.temperature,
                repetition_penalty=self.thinker_generation_config.
                repetition_penalty,
                stop_token_ids=self.thinker_config.eos_token_id,
            )

        generator = asyncio.run_coroutine_threadsafe(
            self.thinker_engine.add_request(
                request_id=request_id,
                prompt=prompt,
                params=params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority,
                inputs=inputs,
            ),
            self._loop,
        ).result()
        self._listen_thinker(request_id, voice_type, generator,
                             self.thinker_outputs[talker_type])

        if self.talker_engine_args and not self._ignore_voice(voice_type):
            talker_hf_config = self.talker_config
            if hasattr(talker_hf_config, 'talker_config'):
                talker_hf_config = talker_hf_config.talker_config

            if not talker_params:
                talker_params = SamplingParams(
                    top_k=self.talker_generation_config.top_k,
                    top_p=self.talker_generation_config.top_p,
                    temperature=self.talker_generation_config.temperature,
                    repetition_penalty=self.talker_generation_config.
                    repetition_penalty,
                    max_tokens=8192,
                )
            if not talker_params.stop_token_ids:
                talker_params.stop_token_ids = []
            talker_params.stop_token_ids.extend([
                talker_hf_config.tts_codec_start_token_id,
                talker_hf_config.tts_codec_end_token_id,
            ])

            # talker: chunked output to reduce overhead of zmq
            talker_params.chunked_return = True
            talker_params.chunked_return_first_size = self.code2wav_future_cache_size // (
                2 if self.code2wav_frequency == "50hz" else 4)
            talker_params.chunked_return_size = self.code2wav_chunk_size // (
                2 if self.code2wav_frequency == "50hz" else 4)

            if params and params.max_tokens:
                factor = 1 if self._ignore_code2wav(voice_type) else 40
                talker_params.max_tokens = min(talker_params.max_tokens,
                                               params.max_tokens * factor)

            self.talker_requests[request_id] = (prompt, talker_params)

        return self.output_queue[request_id]

    def abort_request(self, request_id: str, talker_type: str = "default"):
        # cancel thinker
        asyncio.run_coroutine_threadsafe(
            self.thinker_engine.abort(request_id),
            self._loop,
        ).result()
        # cancel talker
        self.thinker_outputs[talker_type].put((request_id, None, None))

    def _ignore_voice(self, voice_type: str = "default") -> bool:
        return (not voice_type or voice_type == "null" or voice_type == "none")

    def _ignore_code2wav(self, voice_type: str = "default") -> bool:
        return voice_type and voice_type.lower() in ["prefix_caching"]
