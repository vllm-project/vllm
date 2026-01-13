# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import math
from functools import cached_property
from typing import Any, Optional, Union

from transformers import Qwen2TokenizerFast

from vllm.logger import init_logger

logger = init_logger(__name__)


class StepAudio2Tokenizer(Qwen2TokenizerFast):

    _tts_start_token: str = "<tts_start>"  #151693
    _tts_end_token: str = "<tts_end>"  #151694
    _first_audio_token: str = "<audio_0>"
    _tts_pad_token: str = "<tts_pad>"
    _audio_pad_token: str = "<audio_6561>"

    @cached_property
    def max_token_id(self) -> int:
        return len(self.vocab) - 1

    @cached_property
    def tts_start_token_id(self):
        return self.vocab.get(self._tts_start_token)

    @cached_property
    def tts_end_token_id(self):
        return self.vocab.get(self._tts_end_token)

    @cached_property
    def first_audio_token_id(self):
        return self.vocab.get(self._first_audio_token)

    @cached_property
    def tts_pad_token_id(self):
        return self.vocab.get(self._tts_pad_token)

    @cached_property
    def audio_pad_token_id(self):
        return self.vocab.get(self._audio_pad_token)

    def is_step_audio_token(self, token_id: int):
        return token_id >= self.first_audio_token_id

    def apply_chat_template_no_trans(
            self,
            conversation: Union[list[dict[str, str]], list[list[dict[str,
                                                                     str]]]],
            tools: Optional[list[dict]] = None,
            documents: Optional[list[dict[str, str]]] = None,
            chat_template: Optional[str] = None,
            add_generation_prompt: bool = False,
            continue_final_message: bool = False,
            tokenize: bool = True,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_dict: bool = False,
            return_assistant_tokens_mask: bool = False,
            tokenizer_kwargs: Optional[dict[str, Any]] = None,
            **kwargs) -> list[int]:
        """Convert chat messages to token IDs sequence.
        
        Args:
            conversation: list of chat messages
            tools: Tool configurations (optional)
            
        Returns:
            list[int]: Sequence of token IDs
        """
        result = []
        messages = conversation
        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."  # noqa: E501
            )

        if tools:
            result.append('<|BOT|>system\n')
            if messages and messages[0]['role'] == 'system':
                result.append(messages[0]['content'] + '<|EOT|>')
            result.append('<|BOT|>')
            result.append("tool_json_schemas\n")
            result.append(json.dumps(tools, ensure_ascii=False) + '<|EOT|>')
        elif messages and messages[0]['role'] == 'system':
            result.append('<|BOT|>system\n' + messages[0]['content'] +
                          '<|EOT|>')

        for i, message in enumerate(messages):
            if message["role"] == "user":
                result.append('<|BOT|>human\n' + message["content"] +
                              '<|EOT|>')
            elif message["role"] == "system":
                if i != 0:
                    result.append('<|BOT|>system\n' + message["content"] +
                                  '<|EOT|>')
            elif message["role"] == "assistant":
                result.append('<|BOT|>' + message["role"] + '\n')
                if message["content"]:
                    result.append(message["content"])
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        result.append('<tool_call>' + 'function\n' +
                                      tool_call["name"] + '\n')
                        result.append(
                            json.dumps(tool_call["arguments"],
                                       ensure_ascii=False))
                        result.append('</tool_call>')
                result.append('<|EOT|>')
            elif message["role"] == "tool":
                result.append('<|BOT|>')
                function_name = "tool"
                if message.get("tool_call_id"):
                    for prev_msg in messages:
                        if prev_msg["role"] == "assistant" and prev_msg.get(
                                "tool_calls"):
                            for tool_call in prev_msg["tool_calls"]:
                                if tool_call["id"] == message[
                                        "tool_call_id"] and "function" in tool_call:  # noqa: E501
                                    function_name = tool_call["function"][
                                        "name"]
                result.append('function_output\n' + function_name + '\n')
                result.append(message["content"])
                result.append('<|EOT|>')
            elif message["role"] == "function_output":
                result.append('<|BOT|>' + 'input\n' + message["content"] +
                              '<|EOT|>')
            else:
                result.append('<|BOT|>' + message["role"] + '\n' +
                              message["content"] + '<|EOT|>')

        if add_generation_prompt:
            result.append('<|BOT|>assistant\n')

        if continue_final_message:
            final_message = message["content"]
            if isinstance(final_message, (list, tuple)):
                final_message = final_message[-1]["text"]
            final_message = final_message.strip()
            last_index = -1
            for i in range(len(result) - 1, -1, -1):
                if final_message in result[i]:
                    last_index = i
                    break  # 找到后立即退出循环
            result = result[:last_index + 1]

        return ''.join(result)

    def apply_chat_template_trans_ta4(
            self,
            conversation: Union[list[dict[str, str]], list[list[dict[str,
                                                                     str]]]],
            tools: Optional[list[dict]] = None,
            documents: Optional[list[dict[str, str]]] = None,
            chat_template: Optional[str] = None,
            add_generation_prompt: bool = False,
            continue_final_message: bool = False,
            tokenize: bool = True,
            padding: bool = False,
            truncation: bool = False,
            max_length: Optional[int] = None,
            return_dict: bool = False,
            return_assistant_tokens_mask: bool = False,
            tokenizer_kwargs: Optional[dict[str, Any]] = None,
            tts_content: Optional[list[dict]] = None,
            **kwargs) -> list[int]:
        """Convert chat messages to token IDs sequence.
        
        Args:
            conversation: list of chat messages
            tools: Tool configurations (optional)
            
        Returns:
            list[int]: Sequence of token IDs
        """
        result = []
        messages = conversation
        if continue_final_message and add_generation_prompt:
            raise ValueError(
                "continue_final_message and add_generation_prompt are not compatible. Use continue_final_message when you want the model to continue the final message, and add_generation_prompt when you want to add a header that will prompt it to start a new assistant message instead."  # noqa: E501
            )
        if tools:
            result.append('<|BOT|>system\n')
            if messages and messages[0]['role'] == 'system':
                result.append(messages[0]['content'] + '<|EOT|>')
            result.append('<|BOT|>')
            result.append("tool_json_schemas\n")
            result.append(json.dumps(tools, ensure_ascii=False) + '<|EOT|>')
        elif messages and messages[0]['role'] == 'system':
            result.append('<|BOT|>system\n' + messages[0]['content'] +
                          '<|EOT|>')

        for i, message in enumerate(messages):
            message_content = message["content"]
            if message["role"] == "user":
                result.append('<|BOT|>human\n' + message_content + '<|EOT|>')
            elif message["role"] == "system":
                if i != 0:
                    result.append('<|BOT|>system\n' + message_content +
                                  '<|EOT|>')
            elif message["role"] == "assistant":
                result.append('<|BOT|>' + message["role"] + '\n')
                if 'tts_content' not in message:
                    result.append(message_content)
                else:
                    tts_content = message['tts_content']
                    if "tts_text" not in tts_content or "tts_audio" not in tts_content:  # noqa: E501
                        raise ValueError(
                            "tts_text/tts_audio must in tts_content keys.")
                    tts_content['text'] = message_content
                    result.append(tts_content)
                if message.get("tool_calls"):
                    for tool_call in message["tool_calls"]:
                        if "function" in tool_call:
                            tool_call = tool_call["function"]
                        result.append('<tool_call>' + 'function\n' +
                                      tool_call["name"] + '\n')
                        result.append(
                            json.dumps(tool_call["arguments"],
                                       ensure_ascii=False))
                        result.append('</tool_call>')
                result.append('<|EOT|>')
            elif message["role"] == "tool":
                result.append('<|BOT|>')
                function_name = "tool"
                if message.get("tool_call_id"):
                    for prev_msg in messages:
                        if prev_msg["role"] == "assistant" and prev_msg.get(
                                "tool_calls"):
                            for tool_call in prev_msg["tool_calls"]:
                                if tool_call["id"] == message[
                                        "tool_call_id"] and "function" in tool_call:  # noqa: E501
                                    function_name = tool_call["function"][
                                        "name"]
                result.append('function_output\n' + function_name + '\n')
                result.append(message_content)
                result.append('<|EOT|>')
            elif message["role"] == "function_output":
                result.append('<|BOT|>' + 'input\n' + message["content"] +
                              '<|EOT|>')
            else:
                result.append('<|BOT|>' + message["role"] + '\n' +
                              message_content + '<|EOT|>')

        if add_generation_prompt:
            result.append('<|BOT|>assistant\n')

        if continue_final_message:
            final_message = message_content
            if isinstance(final_message, (list, tuple)):
                final_message = final_message[-1]["text"]
            final_message = final_message.strip()
            last_index = -1
            for i in range(len(result) - 1, -1, -1):
                if final_message in result[i]:
                    last_index = i
                    break  # 找到后立即退出循环
            result = result[:last_index + 1]
        trans_token_ids = self.trans_text_audio_to_ta4(result)
        return trans_token_ids

    def build_tts_interleave_data(self,
                                  text_token_ids,
                                  audio_token_ids,
                                  chunk=4):

        text_token_ids_pad = text_token_ids
        chunk_nums = max(math.ceil(len(audio_token_ids) / chunk),
                         len(text_token_ids))
        ta4_content = []
        text_token_ids_pad = text_token_ids + [self.tts_pad_token_id] * (
            chunk_nums - len(text_token_ids))
        audio_token_ids_pad = audio_token_ids + [self.audio_pad_token_id] * (
            chunk_nums - len(audio_token_ids))
        for idx in range(chunk_nums):
            ta4_content += text_token_ids_pad[idx:(idx + 1)]
            ta4_content += audio_token_ids_pad[idx * chunk:(idx + 1) * chunk]

        all_token_ids = [self.tts_start_token_id
                         ] + ta4_content + [self.tts_end_token_id]
        return all_token_ids

    def trans_text_audio_to_ta4(self, content_list: list[str]):
        result = []
        for content in content_list:
            if isinstance(content, str):
                content_tokens = self.tokenize(content)
                content_token_ids = self.convert_tokens_to_ids(content_tokens)
                result += content_token_ids

            elif isinstance(content, dict):
                tts_text_tokens = self.tokenize(content['tts_text'])
                tts_audio_tokens = self.tokenize(content['tts_audio'])
                text_tokens = self.tokenize(content['text'])

                tts_text_tokens_ids = self.convert_tokens_to_ids(
                    tts_text_tokens)
                tts_audio_tokens_ids = self.convert_tokens_to_ids(
                    tts_audio_tokens)
                text_tokens_ids = self.convert_tokens_to_ids(text_tokens)
                trans_token_ids = self.build_tts_interleave_data(
                    tts_text_tokens_ids, tts_audio_tokens_ids)
                result += trans_token_ids + text_tokens_ids

        return result
