import copy
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizer, GenerationConfig
from transformers import AutoTokenizer

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""

IMEND = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"

HistoryType = List[Tuple[str, str]]
TokensType = List[int]
BatchTokensType = List[List[int]]

def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


class vLLMWrapper:
    def __init__(self,
               model_dir: str,
               trust_remote_code: bool = True,
               tensor_parallel_size: int = 1,
               gpu_memory_utilization: float = 0.90,
               dtype: str = "bfloat16",
               **kwargs):

        if dtype not in ("bfloat16", "float16", "float32"):
            print("now not support {}!".format(dtype))
            raise Exception

        #set the 'device_map' attr to make big model run faster
        device_map = kwargs.get('device_map', 'auto')

        # build generation_config
        self.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=trust_remote_code)

        # build tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.eos_token_id = self.generation_config.eos_token_id

        self.stop_words_ids = []

        from vllm import LLM

        quantization = kwargs.get('quantization', None)

        self.model = LLM(model=model_dir,
                            tokenizer=model_dir,
                            tensor_parallel_size=tensor_parallel_size,
                            trust_remote_code=trust_remote_code,
                            quantization=quantization,
                            gpu_memory_utilization=gpu_memory_utilization,
                            dtype=dtype)

        for stop_id in get_stop_words_ids(self.generation_config.chat_format, self.tokenizer):
            self.stop_words_ids.extend(stop_id)
        self.stop_words_ids.extend([self.generation_config.eos_token_id])

    def chat(self,
        query: str,
        history: Optional[HistoryType],
        tokenizer: PreTrainedTokenizer = None,
        system: str = "You are a helpful assistant.",
        generation_config: Optional[GenerationConfig] = None,
        **kwargs):
        generation_config = generation_config if generation_config is not None else self.generation_config
        tokenizer = self.tokenizer if tokenizer is None else tokenizer

        assert generation_config.chat_format == 'chatml', _ERROR_BAD_CHAT_FORMAT

        if history is None:
            history = []
        else:
            # make a copy of the user's input such that is is left untouched
            history = copy.deepcopy(history)

        extra_stop_words_ids = kwargs.get('stop_words_ids', None)
        if extra_stop_words_ids is None:
            extra_stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size

        from vllm.sampling_params import SamplingParams
        sampling_params = SamplingParams(stop_token_ids=self.stop_words_ids,
                                            early_stopping=False,
                                            top_p=self.generation_config.top_p,
                                            top_k=-1 if self.generation_config.top_k == 0 else self.generation_config.top_k,
                                            temperature=self.generation_config.temperature,
                                            repetition_penalty=self.generation_config.repetition_penalty,
                                            max_tokens=self.generation_config.max_new_tokens,
                                        )

        raw_text, context_tokens = make_context(
            self.tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=self.generation_config.chat_format,
        )

        req_outputs = self.model.generate([query],
                                            sampling_params=sampling_params,
                                            prompt_token_ids=[context_tokens])
        req_output = req_outputs[0]

        prompt_str = req_output.prompt
        prompt_ids = req_output.prompt_token_ids
        req_sample_output_ids = []
        req_sample_output_strs = []
        for sample in req_output.outputs:
            output_str = sample.text
            output_ids = sample.token_ids
            if IMEND in output_str:
                output_str = output_str[:-len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[:-len(ENDOFTEXT)]
            req_sample_output_ids.append(prompt_ids + output_ids)
            req_sample_output_strs.append(prompt_str + output_str)
        assert len(req_sample_output_strs) == 1
        response = req_sample_output_strs[0][len(prompt_str):]
        history.append((prompt_str, response))

        return response, history

    def generate(self,
                 inputs: Optional[torch.Tensor] = None,
                 stop_words_ids: Optional[List[List[int]]] = None,
                 generation_config: Optional[GenerationConfig] = None,
                 **kwargs):
        # Process stop_words_ids.
        if stop_words_ids is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
            self.sampling_params.stop_token_ids = stop_words_ids
        query = kwargs.get("query", None)
        if inputs is None or query is None:
            print("*********must input context_tokens (prompt before encoding) and prompt**********")
            return None

        req_outputs = self.model.generate([query],
                                            sampling_params=self.sampling_params,
                                            prompt_token_ids=[inputs])
        req_output = req_outputs[0]

        prompt_str = req_output.prompt
        prompt_ids = req_output.prompt_token_ids
        req_sample_output_ids = []
        req_sample_output_strs = []
        for sample in req_output.outputs:
            output_str = sample.text
            output_ids = sample.token_ids
            if IMEND in output_str:
                output_str = output_str[:-len(IMEND)]
            if ENDOFTEXT in output_str:
                output_str = output_str[:-len(ENDOFTEXT)]
            req_sample_output_ids.append(prompt_ids + output_ids)
            req_sample_output_strs.append(prompt_str + output_str)
        assert len(req_sample_output_ids) == 1
        result = req_sample_output_ids

        return result