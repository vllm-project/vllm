 
from typing import Dict, List
from packaging import version
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
try:
    import fastchat
    from fastchat.conversation import Conversation, SeparatorStyle
    from fastchat.model.model_adapter import get_conversation_template
    _fastchat_available = True
except ImportError:
    _fastchat_available = False
    
def get_conversation_stop(conv: Conversation):
    
    conv_name = conv.name
    if conv_name == "qwen-7b-chat":
        return ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
    
    return conv.stop_str
    
def get_default_model_template():
    from vllm.entrypoints.openai.api_server import parser
    
    args = parser.parse_args()
    default_model_template = getattr(args, 'default_model_template', None)
    print('default_model_template', default_model_template)
    return default_model_template



async def patch_get_gen_prompt(request: ChatCompletionRequest) -> str:
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
    from vllm.entrypoints.openai import api_server
    api_server.get_gen_prompt = patch_get_gen_prompt
