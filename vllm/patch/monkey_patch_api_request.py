from typing import Dict, List, Union, Optional, Literal
from pydantic import BaseModel, Field, validator
from fastapi.responses import JSONResponse
from packaging import version
from functools import cache
from http import HTTPStatus
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, UsageInfo
from vllm.entrypoints.openai import cli_args
import importlib
import time

try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template

    _fastchat_available = True
except ImportError:
    _fastchat_available = False

# vllm:0.2.7
try:
    from vllm.entrypoints.openai.serving_engine import OpenAIServing
except ImportError:
    OpenAIServing = None


class ChatMessageEx(BaseModel):
    role: Optional[str] = "assistant"
    content: Optional[str] = ""

    @validator("role")
    @classmethod
    def validate_ignore_resource(cls, value: str) -> str:
        if not value:
            return "assistant"
        return value


class ChatCompletionResponseChoiceEx(BaseModel):
    index: int
    message: ChatMessageEx
    finish_reason: Optional[Literal["stop", "length"]] = None


class ChatCompletionResponseEx(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{random_uuid()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoiceEx]
    usage: UsageInfo


def get_conversation_stop(conv: Conversation):
    conv_name = conv.name
    if conv_name == "qwen-7b-chat":
        return ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
    if not hasattr(conv, 'stop_str'):
        return
    return conv.stop_str


def get_conversation_stop_token_ids(conv: Conversation):
    conv_name = conv.name
    # if conv_name == "Yi-34b-chat":
    #     return [7, ]

    if not hasattr(conv, 'stop_token_ids'):
        return
    return conv.stop_token_ids


@cache
def get_default_model_template():
    from vllm.entrypoints.openai.cli_context_args import cliContext
    args = cliContext.args
    if not args:
        return
    default_model_template = getattr(args, 'default_model_template', None)
    print('default_model_template', default_model_template)
    return default_model_template


async def patch_check_model(request) -> Optional[JSONResponse]:
    from vllm.entrypoints.openai.api_server import origin_check_model
    ret = await origin_check_model(request)
    reset_default_request(request=request)
    return ret

origin_make_arg_parser = cli_args.make_arg_parser

def patch_make_arg_parser():
    parser = origin_make_arg_parser()
    parser.add_argument("--default-model-template",
                        type=str,
                        default=None,
                        help="model template")
    return parser.parse_args()
    
async def origin_serving_self_check_model(self, request):
    # if request.model == self.served_model:
    #     return
    # return self.create_error_response(
    #     message=f"The model `{request.model}` does not exist.",
    #     err_type="NotFoundError",
    #     status_code=HTTPStatus.NOT_FOUND)

    if request.model in self.served_model_names:
        return None
    # if request.model in [lora.lora_name for lora in self.lora_requests]:
    #     return None
    return self.create_error_response(
        message=f"The model `{request.model}` does not exist.",
        err_type="NotFoundError",
        status_code=HTTPStatus.NOT_FOUND)


async def patch_serving_self_check_model(self, request):
    ret = await self.origin_serving_check_model(request)
    reset_default_request(request=request)
    return ret


def reset_default_request(request):
    """
    重置 request default
    Args:
        request (_type_): _description_
    """
    if not _fastchat_available:
        return

    default_model_template = get_default_model_template()
    model_name = default_model_template if default_model_template else request.model
    conv = get_conversation_template(model_name)

    # Set default stop
    if not request.stop and not request.stop_token_ids:
        print("Set default stop")
        _stop = get_conversation_stop(conv)
        # print(_stop)
        if _stop:
            request.stop = _stop
        _stop_token_ids = get_conversation_stop_token_ids(conv)
        # print(_stop_token_ids)
        if _stop_token_ids:
            request.stop_token_ids = list(_stop_token_ids)
    # print("end")


async def patch_get_gen_prompt(request) -> str:
    if not _fastchat_available:
        raise ModuleNotFoundError(
            "fastchat is not installed. Please install fastchat to use "
            "the chat completion and conversation APIs: `$ pip install fschat`"
        )
    if version.parse(fastchat.__version__) < version.parse("0.2.23"):
        raise ImportError(
            f"fastchat version is low. Current version: {fastchat.__version__} "
            "Please upgrade fastchat to use: `$ pip install -U fschat`")
    default_model_template = get_default_model_template()
    model_name = default_model_template if default_model_template else request.model
    conv = get_conversation_template(model_name)

    conv = Conversation(
        name=conv.name,
        system_template=conv.system_template,
        system_message=conv.system_message,
        roles=conv.roles,
        messages=list(conv.messages),  # prevent in-place modification
        offset=conv.offset,
        sep_style=SeparatorStyle(conv.sep_style),
        sep=conv.sep,
        sep2=conv.sep2,
        stop_str=conv.stop_str,
        stop_token_ids=conv.stop_token_ids,
    )

    if isinstance(request.messages, str):
        prompt = request.messages
    else:
        for message in request.messages:
            msg_role = message["role"]
            if msg_role == "system":
                conv.system_message = message["content"]
            elif msg_role == "user":
                conv.append_message(conv.roles[0], message["content"])
            elif msg_role == "assistant":
                conv.append_message(conv.roles[1], message["content"])
            else:
                raise ValueError(f"Unknown role: {msg_role}")

        # Add a blank message for the assistant.
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

    # Set default stop
    if not request.stop:
        request.stop = get_conversation_stop(conv)

    return prompt


def patch_api_server():
    # print("load monkey_patch_api_request_v4")
    from vllm.entrypoints.openai import protocol
    protocol.ChatMessage = ChatMessageEx
    protocol.ChatCompletionResponseChoice = ChatCompletionResponseChoiceEx
    protocol.ChatCompletionResponse = ChatCompletionResponseEx
    try:
        from vllm.entrypoints.openai.api_server import check_model as origin_check_model
    except ImportError:
        origin_check_model = None
    from vllm.entrypoints.openai import api_server

    api_server.get_gen_prompt = patch_get_gen_prompt
    if origin_check_model:
        api_server.origin_check_model = origin_check_model
        api_server.check_model = patch_check_model

    if OpenAIServing:
        OpenAIServing.origin_serving_check_model = origin_serving_self_check_model
        OpenAIServing._check_model = patch_serving_self_check_model
        
    cli_args.make_arg_parser = patch_make_arg_parser

