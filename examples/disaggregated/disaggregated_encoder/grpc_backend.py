# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Rust Inference gRPC transport for the Python EPD proxy."""

from __future__ import annotations

import asyncio
import copy
import sys
import time
import uuid
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import grpc
import msgspec
from fastapi import HTTPException
from google.protobuf.json_format import MessageToDict, ParseDict

from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from vllm.entrypoints.scale_out.token_in_token_out.protocol import (
    GenerateTokensChoice,
    GenerateTokensResponse,
    GenerateTokensStreamChoice,
    GenerateTokensStreamResponse,
)
from vllm.entrypoints.serve.engine.protocol import ErrorResponse, UsageInfo
from vllm.entrypoints.serve.utils.api_utils import get_max_tokens
from vllm.renderers import renderer_from_config
from vllm.renderers.online_derenderer import OnlineDerenderer
from vllm.renderers.online_renderer import OnlineRenderer
from vllm.v1.serial_utils import MsgpackEncoder


@lru_cache(maxsize=1)
def load_protocol():
    """Generate Python bindings from this checkout, without duplicating protos."""
    root = str(Path(__file__).resolve().parents[3])
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        pb, inference = grpc.protos_and_services("rust/proto/inference.proto")
        control_pb, control = grpc.protos_and_services("rust/proto/control.proto")
    except NotImplementedError as error:
        raise RuntimeError(
            "gRPC proxy requires grpcio-tools; install it in the proxy environment"
        ) from error
    return pb, inference, control_pb, control


def transfer_params(value):
    """Recover integral JSON numbers lost in protobuf Struct's double values."""

    def normalize(item):
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    return normalize(MessageToDict(value))


class GrpcBackend:
    def __init__(
        self,
        model: str | None = None,
        *,
        trust_remote_code: bool = False,
        trust_request_mm_kwargs: bool = False,
        reasoning_parser: str | None = None,
        request_timeout: float = 300,
    ):
        self.pb, self.inference, self.control_pb, self.control = load_protocol()
        self.model = model
        self.trust_remote_code = trust_remote_code
        self.trust_request_mm_kwargs = trust_request_mm_kwargs
        self.reasoning_parser = reasoning_parser
        self.request_timeout = request_timeout
        self.channels = {}
        self.frontends = {}
        self._frontend_lock = asyncio.Lock()

    def channel(self, url: str):
        target = urlsplit(url).netloc
        if not target:
            raise HTTPException(400, "Expected grpc://host:port")
        if target not in self.channels:
            self.channels[target] = grpc.aio.insecure_channel(
                target,
                options=[
                    ("grpc.max_send_message_length", -1),
                    ("grpc.max_receive_message_length", -1),
                ],
            )
        return self.channels[target]

    async def close(self):
        await asyncio.gather(*(channel.close() for channel in self.channels.values()))
        for render, _ in self.frontends.values():
            render.renderer.shutdown()
        self.channels.clear()
        self.frontends.clear()

    async def healthy(self, url: str, timeout: float = 2) -> bool:
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        response = await health_pb2_grpc.HealthStub(self.channel(url)).Check(
            health_pb2.HealthCheckRequest(service="vllm.Inference"), timeout=timeout
        )
        return response.status == health_pb2.HealthCheckResponse.SERVING

    async def models(self, url: str) -> dict:
        info = await self.control.ControlStub(self.channel(url)).GetModelInfo(
            self.control_pb.GetModelInfoRequest(), timeout=self.request_timeout
        )
        return {
            "object": "list",
            "data": [
                {"id": info.served_model_name or info.model_id, "object": "model"}
            ],
        }

    async def frontend(self, model: str):
        source = self.model or model
        async with self._frontend_lock:
            if source not in self.frontends:
                self.frontends[source] = await asyncio.to_thread(
                    self._load_frontend, source
                )
        return self.frontends[source]

    def _load_frontend(self, model: str):
        model_config = ModelConfig(
            model=model,
            trust_remote_code=self.trust_remote_code,
            enable_mm_embeds=True,
            mm_processor_cache_gb=0,
        )
        config = VllmConfig(
            model_config=model_config, device_config=DeviceConfig(device="cpu")
        )
        # This is a renderer, not an engine with a connector of its own.
        if model_config.multimodal_config is not None:
            model_config.multimodal_config.allow_missing_mm_embeddings = True
        renderer = renderer_from_config(config)
        options = dict(
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            reasoning_parser=self.reasoning_parser,
        )
        return (
            OnlineRenderer(
                model_config,
                renderer,
                trust_request_mm_kwargs=self.trust_request_mm_kwargs,
                **options,
            ),
            OnlineDerenderer(model_config, renderer, **options),
        )

    async def render_prompt(self, render, request):
        """Leave raw media fetching, processing and expansion to the worker."""
        messages = copy.deepcopy(request.messages)
        media = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                kind = part.get("type")
                if kind not in ("image_url", "audio_url", "video_url", "input_audio"):
                    continue
                modality = (
                    "audio" if kind == "input_audio" else kind.removesuffix("_url")
                )
                if kind == "input_audio":
                    audio = part["input_audio"]
                    source = f"data:audio/{audio['format']};base64,{audio['data']}"
                else:
                    source = part[kind]
                    if isinstance(source, dict):
                        source = source.get("url")
                if not isinstance(source, str) or not source:
                    raise HTTPException(
                        400, "gRPC raw media requires a URL or data URI"
                    )
                item = self.pb.MediaItem(
                    modality=getattr(self.pb, f"MODALITY_{modality.upper()}"),
                    uuid=part.get("uuid") or "",
                )
                if source.startswith("data:"):
                    item.data_uri = source
                elif source.startswith(("http://", "https://")):
                    item.url = source
                else:
                    raise HTTPException(400, "gRPC media requires http(s) or data URLs")
                media.append(item)
                # The existing renderer accepts absent media for UUID references.
                # Stop before process_for_engine: the worker expands placeholders.
                part.clear()
                part.update(
                    type=f"{modality}_url",
                    uuid=item.uuid or f"grpc-media-{len(media)}",
                    **{f"{modality}_url": None},
                )
        if not media:
            result = await render.render_chat(request, skip_mm_cache=True)
            if isinstance(result, ErrorResponse):
                raise HTTPException(result.error.code, result.model_dump())
            _, inputs = result
            if len(inputs) != 1 or inputs[0]["type"] not in ("token", "multimodal"):
                raise HTTPException(
                    400, "gRPC EPD requires a decoder-only token prompt"
                )
            return inputs[0], media
        if request.mm_processor_kwargs or request.media_io_kwargs:
            raise HTTPException(
                400, "gRPC raw media does not support per-request processing options"
            )
        if request.tools or request.tool_choice not in (None, "none"):
            raise HTTPException(400, "gRPC raw media does not support tool calls")
        error = render.validate_chat_template(
            request.chat_template, request.chat_template_kwargs, False
        )
        if error is not None:
            raise HTTPException(error.error.code, error.model_dump())
        chat_params = request.build_chat_params(
            render.chat_template,
            render.chat_template_content_format,
            render.reasoning_effort_map,
        ).with_defaults(render.default_chat_template_kwargs)
        _, prompt = await render.renderer.render_messages_async(messages, chat_params)
        mm_data = prompt.pop("multi_modal_data", {})
        if any(any(item is not None for item in items) for items in mm_data.values()):
            raise HTTPException(400, "gRPC cannot mix raw media and embedding inputs")
        prompts = await render.renderer.tokenize_prompts_async(
            [prompt], request.build_tok_params(render.model_config)
        )
        return prompts[0], media

    async def prepare(self, data: dict, request_id: str):
        request = ChatCompletionRequest.model_validate(data)
        if (request.n or 1) != 1:
            raise HTTPException(400, "The gRPC EPD proxy currently requires n=1")
        if request.stream and request.stop:
            raise HTTPException(400, "Streaming stop strings are not supported on gRPC")
        if (
            request.prompt_logprobs is not None
            or request.echo
            or request.use_beam_search
        ):
            raise HTTPException(
                400, "gRPC does not support prompt logprobs, echo or beams"
            )
        render, derender = await self.frontend(request.model)
        engine_input, raw_media = await self.render_prompt(render, request)
        ids = engine_input["prompt_token_ids"]
        defaults = render.model_config.get_diff_sampling_param()
        max_tokens = get_max_tokens(
            render.model_config.max_model_len,
            request.max_completion_tokens
            if request.max_completion_tokens is not None
            else request.max_tokens,
            len(ids),
            defaults,
            defaults.get("max_new_tokens"),
        )
        params = request.to_sampling_params(max_tokens, defaults)
        params.update_from_tokenizer(render.renderer.get_tokenizer())
        proto = self.pb.GenerateRequest(
            request_id=request_id,
            model=request.model,
            token_ids=self.pb.TokenIds(ids=ids),
            temperature=params.temperature,
            sampling=self.pb.RandomSampling(
                num_sequences=1,
                top_k=max(0, params.top_k),
                top_p=params.top_p,
                min_p=params.min_p,
                **({"seed": params.seed} if params.seed is not None else {}),
            ),
            decoding=self.pb.DecodingParameters(
                presence_penalty=params.presence_penalty,
                frequency_penalty=params.frequency_penalty,
                repetition_penalty=params.repetition_penalty,
                logit_bias=params.logit_bias or {},
                allowed_token_ids=params.allowed_token_ids or [],
                bad_words_token_ids=[
                    self.pb.TokenIds(ids=bad)
                    for bad in params.bad_words_token_ids or []
                ],
            ),
            stopping=self.pb.StoppingCriteria(
                max_new_tokens=params.max_tokens,
                min_new_tokens=params.min_tokens,
                stop_token_ids=params.stop_token_ids,
                stop_strings=params.stop,
                ignore_eos=params.ignore_eos,
                include_stop_strings=params.include_stop_str_in_output,
            ),
            response=self.pb.ResponseOptions(
                prompt_token_ids=True,
                output_text=bool(params.stop),
                output_token_ids=True,
                output_logprobs=bool(request.logprobs),
                skip_special_tokens=params.skip_special_tokens,
            ),
            priority=request.priority,
            session_id=request.session_id,
            watermarking=params.watermarking,
            media=raw_media,
        )
        if params.structured_outputs is not None:
            for key in ("json", "regex", "grammar", "json_object", "structural_tag"):
                value = getattr(params.structured_outputs, key)
                if value is not None:
                    if key == "json" and isinstance(value, dict):
                        value = msgspec.json.encode(value).decode()
                    setattr(proto.decoding, key, value)
            if params.structured_outputs.choice is not None:
                proto.decoding.choice.choices.extend(params.structured_outputs.choice)
        if request.logprobs:
            proto.response.output_candidates.top_n = request.top_logprobs or 0
        proto.kv.cache_salt = request.cache_salt or ""
        proto.kv.bypass_prefix_cache = bool(params.skip_reading_prefix_cache)
        ParseDict(
            render.effective_chat_template_kwargs(request),
            proto.engine_reasoning_gate.chat_template_kwargs,
        )
        reasoning_ended = request.resolve_reasoning_ended(None, ids)
        if reasoning_ended is not None:
            proto.engine_reasoning_gate.reasoning_ended = reasoning_ended
        if params.thinking_token_budget is not None:
            proto.stopping.thinking_token_budget = params.thinking_token_budget
        for key in ("ec_transfer_params", "kv_transfer_params"):
            if value := (params.extra_args or {}).get(key):
                ParseDict(value, getattr(proto.kv, key))
        if engine_input.get("type") == "multimodal":
            encoder = MsgpackEncoder(size_threshold=2**62)
            for modality, hashes in engine_input["mm_hashes"].items():
                for index, mm_hash in enumerate(hashes):
                    position = engine_input["mm_placeholders"][modality][index]
                    item = engine_input["mm_kwargs"][modality][index]
                    if item is None:
                        raise HTTPException(400, "gRPC requires inline multimodal data")
                    proto.media.add(
                        modality=getattr(self.pb, f"MODALITY_{modality.upper()}"),
                        features=self.pb.PreprocessedMediaFeatures(
                            kwargs=bytes(encoder.encode(item)[0]),
                            identifier=mm_hash,
                            mm_hash=mm_hash,
                            offset=position.offset,
                            length=position.length,
                            is_embed=(
                                position.is_embed.tolist()
                                if position.is_embed is not None
                                else []
                            ),
                        ),
                    )
            if sum(len(item.features.kwargs) for item in proto.media) > 16 * 1024**2:
                raise HTTPException(
                    413, "Multimodal features exceed the Rust gRPC 16 MiB inline limit"
                )
        return request, proto, derender

    @asynccontextmanager
    async def call(self, url: str, proto, rank: str | None, *, stream: bool):
        channel = self.channel(url)
        metadata = [("x-data-parallel-rank", rank)] if rank is not None else []
        stub = self.inference.InferenceStub(channel)
        rpc = (stub.GenerateStream if stream else stub.Generate)(
            proto, timeout=self.request_timeout, metadata=metadata
        )
        completed = False
        try:
            yield rpc
            completed = True
        except grpc.aio.AioRpcError as error:
            status = {
                grpc.StatusCode.INVALID_ARGUMENT: 400,
                grpc.StatusCode.NOT_FOUND: 404,
                grpc.StatusCode.RESOURCE_EXHAUSTED: 503,
                grpc.StatusCode.DEADLINE_EXCEEDED: 504,
            }.get(error.code(), 502)
            raise HTTPException(status, error.details()) from error
        finally:
            if not completed:
                rpc.cancel()

                # Shield cleanup from request cancellation; bound it independently.
                async def abort():
                    with suppress(grpc.aio.AioRpcError):
                        await self.control.ControlStub(channel).Abort(
                            self.control_pb.AbortRequest(
                                request_ids=[proto.request_id]
                            ),
                            timeout=2,
                        )

                task = asyncio.create_task(abort())
                await asyncio.shield(task)

    def output(self, reply):
        output = reply.outputs
        finish = output.finish_info if output.HasField("finish_info") else None
        reason = None
        stop_reason = None
        if finish is not None:
            reason = {1: "length", 2: "stop"}.get(finish.finish_reason)
            if reason is None:
                raise HTTPException(502, "gRPC worker aborted the request")
            which = finish.WhichOneof("stop_reason")
            if which is not None:
                stop_reason = getattr(finish, which)
        logprobs = None
        if output.logprobs:
            logprobs = {
                "content": [
                    {
                        "token_id": token,
                        "logprob": score,
                        "rank": output.ranks[i] if i < len(output.ranks) else 1,
                        "top_logprobs": [
                            {"token_id": c.id, "logprob": c.logprob, "rank": c.rank}
                            for c in (
                                output.candidate_tokens[i].tokens
                                if i < len(output.candidate_tokens)
                                else []
                            )
                        ],
                    }
                    for i, (token, score) in enumerate(
                        zip(output.token_ids, output.logprobs)
                    )
                ]
            }
        return dict(
            index=output.index,
            token_ids=list(output.token_ids),
            finish_reason=reason,
            stop_reason=stop_reason,
            logprobs=logprobs,
        )

    async def request(self, url: str, data: dict, headers: dict):
        # A retry must not share its engine id with an earlier timed-out attempt.
        request_id = f"{headers.get('x-request-id', 'epd')}-{uuid.uuid4().hex}"
        try:
            request, proto, derender = await self.prepare(data, request_id)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        response = GrpcResponse(self, url, request, proto, derender, headers)
        if not request.stream:
            await response.read()
        return response


class GrpcResponse:
    """The small response surface used by the existing HTTP forwarding path."""

    status = 200
    headers = {"Content-Type": "application/json"}

    def __init__(self, backend, url, request, proto, derender, headers):
        self.backend = backend
        self.url = url
        self.request = request
        self.proto = proto
        self.derender = derender
        self.rank = headers.get("X-data-parallel-rank")
        self.content = self
        self._body = None
        self._iterator = None
        self.prompt_ids = list(proto.token_ids.ids)
        self.prompt_tokens = len(self.prompt_ids)

    def update_prompt(self, reply):
        if reply.HasField("prompt_info"):
            self.prompt_tokens = reply.prompt_info.num_prompt_tokens
            if reply.prompt_info.token_ids:
                self.prompt_ids = list(reply.prompt_info.token_ids)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        if self._iterator is not None:
            await self._iterator.aclose()
        return False

    def raise_for_status(self):
        pass

    async def read(self):
        if self._body is None:
            async with self.backend.call(
                self.url, self.proto, self.rank, stream=False
            ) as rpc:
                reply = await rpc
                self.update_prompt(reply)
                if not reply.outputs.HasField("finish_info"):
                    raise HTTPException(502, "gRPC unary reply has no terminal output")
                choice = self.backend.output(reply)
            gen = GenerateTokensResponse(
                request_id=self.proto.request_id,
                choices=[GenerateTokensChoice(**choice)],
            )
            # Encoder-only workers finish without sampling a token.
            if choice["token_ids"]:
                choices, _ = await self.derender.derender_chat(
                    gen, self.request, self.prompt_ids
                )
            else:
                choices = []
            prompt_tokens = self.prompt_tokens
            completion_tokens = reply.outputs.finish_info.num_output_tokens
            response = ChatCompletionResponse(
                id=self.proto.request_id,
                model=self.request.model,
                choices=choices,
                usage=UsageInfo(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            ).model_dump(exclude_none=True)
            finish = reply.outputs.finish_info
            for key in ("ec_transfer_params", "kv_transfer_params"):
                if finish.HasField(key):
                    response[key] = transfer_params(getattr(finish, key))
            self._body = msgspec.json.encode(response)
        return self._body

    async def text(self):
        return (await self.read()).decode()

    def iter_any(self):
        self._iterator = self._stream()
        return self._iterator

    async def _stream(self):
        state = None
        created = int(time.time())
        completed = False
        tokens = 0
        async with self.backend.call(
            self.url, self.proto, self.rank, stream=True
        ) as rpc:
            async for reply in rpc:
                self.update_prompt(reply)
                if not reply.HasField("outputs"):
                    continue
                choice = self.backend.output(reply)
                tokens += len(choice["token_ids"])
                chunk, state = await self.derender.derender_chat_stream(
                    self.request.model,
                    GenerateTokensStreamResponse(
                        request_id=self.proto.request_id,
                        choices=[GenerateTokensStreamChoice(**choice)],
                    ),
                    state,
                    self.request,
                    self.prompt_tokens,
                    self.prompt_ids,
                )
                chunk.created = created
                yield (
                    b"data: "
                    + chunk.model_dump_json(exclude_none=True).encode()
                    + b"\n\n"
                )
                completed = choice["finish_reason"] is not None
            if not completed:
                raise HTTPException(502, "gRPC stream ended without a terminal output")
        if self.request.stream_options and self.request.stream_options.include_usage:
            prompt_tokens = self.prompt_tokens
            yield (
                b"data: "
                + msgspec.json.encode(
                    {
                        "id": self.proto.request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self.request.model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": tokens,
                            "total_tokens": prompt_tokens + tokens,
                        },
                    }
                )
                + b"\n\n"
            )
        yield b"data: [DONE]\n\n"
