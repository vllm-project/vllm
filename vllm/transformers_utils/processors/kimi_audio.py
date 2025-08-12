# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from https://github.com/MoonshotAI/Kimi-Audio/tree/master/kimia_infer/api/prompt_manager.py
from typing import Union

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch import nn
from subprocess import CalledProcessError, run
from typing import Dict, List
from dataclasses import dataclass
import os
from functools import lru_cache
from typing import Optional, Union
from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs
from transformers import AutoProcessor
from ...transformers_utils.tokenizers import Glm4Tokenizer

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 120
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = ["ffmpeg", "-nostdin", "-threads", "0", "-i", file, "-f", "s16le",
           "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr), "-"]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int = 128) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    with np.load(
        os.path.join(os.path.dirname(__file__), "mel_filters.npz")  # todo
        # os.path.join("assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 128,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(
        audio, N_FFT, HOP_LENGTH, window=window, return_complex=True
    )
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperEncoder(nn.Module):
    def __init__(
        self, model_path, mel_batch_size=40, unfreeze_online_whisper_model=False
    ):
        super().__init__()
        self.speech_encoder = WhisperModel.from_pretrained(model_path).encoder
        self.unfreeze_online_whisper_model = unfreeze_online_whisper_model
        if not self.unfreeze_online_whisper_model:
            self.speech_encoder.eval()
        self.mel_batch_size = mel_batch_size

    def forward(self, audio, kimia_whisper_clip_silence=False):
        if isinstance(audio, torch.Tensor):
            audio = audio[0]
            audio = audio.cpu().numpy()
        time_step = 0
        audios = []
        while time_step * 16000 < audio.shape[0]:
            audio_segment = audio[time_step * 16000 : (time_step + 30) * 16000]
            audios.append(audio_segment)
            time_step += 30

        final_audio_embedding = []

        for audio_segment in audios:
            # import pdb; pdb.set_trace()
            assert audio_segment.shape[0] <= 480000
            L = audio_segment.shape[0]
            # to match huggingface logic, with use attention mask to control the
            # length and the slice with mask[:, ::160], also match the glm4 12.5 logic
            token_len = (L - 1) // (160 * 8) + 1

            pad_audio = pad_or_trim(audio_segment.flatten())
            mel = log_mel_spectrogram(pad_audio)  # torch.Size([80, 3000])
            assert mel.shape[1] == 3000
            if kimia_whisper_clip_silence:
                input_seq_lens_list = [token_len * 4]
                input_seq_lens = torch.LongTensor(input_seq_lens_list).to(
                    self.speech_encoder.conv1.weight.device
                )
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0)
                    .to(self.speech_encoder.conv1.weight.device)
                    .to(torch.bfloat16),
                    return_dict=True,
                    input_seq_lens=input_seq_lens,
                ).last_hidden_state
            else:
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0)
                    .to(self.speech_encoder.conv1.weight.device)
                    .to(torch.bfloat16),
                    return_dict=True,
                ).last_hidden_state
                # audio_embedding: [1, 3000, 1280]
                audio_embedding = audio_embedding[:, : token_len * 4, :]
            final_audio_embedding.append(audio_embedding)

        final_audio_embedding = torch.cat(final_audio_embedding, dim=1)
        return final_audio_embedding

    @torch.no_grad()
    def tokenize_waveform(self, audio, kimia_whisper_clip_silence=False):
        audio_embedding = self.forward(audio, kimia_whisper_clip_silence)
        return audio_embedding.cpu()


class KimiAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {},
    }


@dataclass
class ExtraTokens:
    msg_end: int
    user_msg_start: int
    assistant_msg_start: int

    media_begin: int
    media_end: int

    kimia_text_blank: int
    kimia_text_eos: int

    kimia_user_msg_start: int
    kimia_assistant_msg_start: int

    kimia_speech_ct_id: int
    kimia_speech_ctd_id: int

    pad: int


def instantiate_extra_tokens(tokenizer):
    if hasattr(tokenizer, "special_tokens"):
        map_fn = lambda x: tokenizer.special_tokens[x]
    elif hasattr(tokenizer, "convert_tokens_to_ids"):
        map_fn = lambda x: tokenizer.convert_tokens_to_ids(x)
    else:
        raise ValueError(f"Invalid tokenizer type: {type(tokenizer)}")
    return ExtraTokens(
        msg_end=map_fn("<|im_msg_end|>"),  # 0
        user_msg_start=map_fn("<|im_user_msg_start|>"),  # 1
        assistant_msg_start=map_fn("<|im_assistant_msg_start|>"),  # 2
        media_begin=map_fn("<|im_media_begin|>"),  # 13
        media_end=map_fn("<|im_media_end|>"),  # 15
        kimia_text_blank=map_fn("<|im_kimia_text_blank|>"),  # 18
        kimia_text_eos=map_fn("<|im_kimia_text_eos|>"),  # 19
        kimia_user_msg_start=map_fn("<|im_kimia_user_msg_start|>"),  # 22
        kimia_assistant_msg_start=map_fn(
            "<|im_kimia_assistant_msg_start|>"
        ),  # 23
        kimia_speech_ct_id=map_fn("<|im_kimia_speech_ct_id|>"),  # 27
        kimia_speech_ctd_id=map_fn("<|im_kimia_speech_ctd_id|>"),  # 28
        pad=tokenizer.pad_id,
    )


class KimiAContent:
    def __init__(
        self,
        audio_token_ids=None,
        text_token_ids=None,
        is_continuous_mask=None,
        audio_token_loss_mask=None,
        text_token_loss_mask=None,
    ):
        self.audio_token_ids: list[int] = audio_token_ids or []
        self.text_token_ids: list[int] = text_token_ids or []
        self.is_continuous_mask: list[int] = is_continuous_mask or []

        self.audio_token_loss_mask: list[int] = audio_token_loss_mask or []
        self.text_token_loss_mask: list[int] = text_token_loss_mask or []

        self.continuous_feature = []

    def audio_append(
        self,
        index: int,
        is_continuous: bool = False,
        audio_token_loss_mask: bool = False,
    ):
        self.audio_token_ids.append(index)
        self.is_continuous_mask.append(is_continuous)
        self.audio_token_loss_mask.append(audio_token_loss_mask)

    def text_append(self, index: int, text_token_loss_mask: bool = False):
        self.text_token_ids.append(index)
        self.text_token_loss_mask.append(text_token_loss_mask)

    def audio_extend(
        self,
        ids: list[int],
        is_continuous: bool = False,
        audio_token_loss_mask: bool = False,
    ):
        self.audio_token_ids.extend(ids)
        self.is_continuous_mask.extend([is_continuous] * len(ids))
        self.audio_token_loss_mask.extend([audio_token_loss_mask] * len(ids))

    def text_extend(self, ids: list[int], text_token_loss_mask: bool = False):
        self.text_token_ids.extend(ids)
        self.text_token_loss_mask.extend([text_token_loss_mask] * len(ids))

    def audio_prepend(
        self,
        index: int,
        is_continuous: bool = False,
        audio_token_loss_mask: bool = False,
    ):
        self.audio_token_ids = [index] + self.audio_token_ids
        self.is_continuous_mask = [is_continuous] + self.is_continuous_mask
        self.audio_token_loss_mask = [
            audio_token_loss_mask
        ] + self.audio_token_loss_mask

    def text_prepend(self, index: int, text_token_loss_mask: bool = False):
        self.text_token_ids = [index] + self.text_token_ids
        self.text_token_loss_mask = [
            text_token_loss_mask
        ] + self.text_token_loss_mask

    def audio_pretend(
        self,
        ids: list[int],
        is_continuous: bool = False,
        audio_token_loss_mask: bool = False,
    ):
        self.audio_token_ids = ids + self.audio_token_ids
        self.is_continuous_mask = [is_continuous] * len(
            ids
        ) + self.is_continuous_mask
        self.audio_token_loss_mask = [audio_token_loss_mask] * len(
            ids
        ) + self.audio_token_loss_mask

    def text_pretend(self, ids: list[int], text_token_loss_mask: bool = False):
        self.text_token_ids = ids + self.text_token_ids
        self.text_token_loss_mask = [text_token_loss_mask] * len(
            ids
        ) + self.text_token_loss_mask

    def merge(self, other: "KimiAContent"):
        self.audio_token_ids.extend(other.audio_token_ids)
        self.text_token_ids.extend(other.text_token_ids)
        self.is_continuous_mask.extend(other.is_continuous_mask)
        self.audio_token_loss_mask.extend(other.audio_token_loss_mask)
        self.text_token_loss_mask.extend(other.text_token_loss_mask)
        self.continuous_feature.extend(other.continuous_feature)

    def to_tensor(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.tensor([self.audio_token_ids], dtype=torch.long),
            torch.tensor([self.text_token_ids], dtype=torch.long),
            torch.tensor([self.is_continuous_mask], dtype=torch.bool),
            torch.tensor([self.audio_token_loss_mask], dtype=torch.bool),
            torch.tensor([self.text_token_loss_mask], dtype=torch.bool),
        )

    def is_valid(self) -> bool:
        return (
            len(self.audio_token_ids)
            == len(self.text_token_ids)
            == len(self.is_continuous_mask)
            == len(self.audio_token_loss_mask)
            == len(self.text_token_loss_mask)
        )


class KimiAudioProcessor:
    r"""
    Lightweight processor:
    - audio_tokenizer: has method `tokenize(audio_path=...)` -> torch.Tensor (1, N) of discrete audio token ids (before offset)
    - text_tokenizer: huggingface tokenizer-like, has `encode` or `encode_plus` and `convert_tokens_to_ids`
    """

    attributes = ["audio_tokenizer", "text_tokenizer"]
    audio_tokenizer_class = "Glm4Tokenizer"
    text_tokenizer_class = "TikTokenTokenizer"

    def __init__(
        self,
        kimia_text_audiodelaytokens,
        kimia_token_offset,
        audio_tokenizer=None,
        text_tokenizer=None,
        chat_template=None,
    ):
        self.audio_tokenizer = audio_tokenizer or Glm4Tokenizer(
            "THUDM/glm-4-voice-tokenizer"
        )
        self.text_tokenizer = text_tokenizer
        self.chat_template = chat_template
        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.kimia_token_offset = kimia_token_offset

    def _tokenize_audio(self, wav_path) -> List[int]:
        wav_tokens = self.audio_tokenizer.tokenize(audio_path=wav_path)
        wav_tokens = wav_tokens + self.kimia_token_offset
        wav_tokens_list = wav_tokens.squeeze(0).cpu().numpy().tolist()
        return wav_tokens_list

    def _tokenize_text(self, text: str) -> List[int]:
        if text is None:
            return None
        if hasattr(self.text_tokenizer, "encode"):
            return self.text_tokenizer.encode(text, bos=False, eos=False)
        else:
            return self.text_tokenizer(text, add_special_tokens=False)[
                "input_ids"
            ]

    def extract_whisper_feat(self, wav: torch.Tensor | str) -> torch.Tensor:
        if isinstance(wav, str):
            wav = librosa.load(wav, sr=16000)[0]

            wav_tensor = torch.tensor(wav).unsqueeze(0)[:, :]
        elif isinstance(wav, torch.Tensor):
            wav_tensor = wav
        else:
            raise ValueError(f"Invalid wav type: {type(wav)}")
        return wav_tensor

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
        extract_whisper_feature=False,
        output_type: str = "text",
    ) -> KimiAContent:
        kimia_content_msg = KimiAContent()

        role = message["role"]

        has_loss = role == "assistant"

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_user_msg_start
                )
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_blank
                )
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start
                )
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_blank
                )
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens)
            )

            if role == "assistant":
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_eos, has_loss
                )  # eos for text stream
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_text_blank,
                    audio_token_loss_mask=False,
                )

        elif message["message_type"] == "audio":
            if "audio_tokens" in message:
                speech_tokens = message["audio_tokens"]
            else:
                audio_path = message["content"]
                speech_tokens = self._tokenize_audio(audio_path)

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(
                speech_tokens,
                is_continuous=True,
                audio_token_loss_mask=has_loss,
            )
            kimia_content_msg.audio_append(
                self.extra_tokens.media_end, audio_token_loss_mask=has_loss
            )  # EOS for audio stream
            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] * (len(speech_tokens) + 2)
            )

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ct_id
                    )
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id
                    )
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_blank
                )

            if extract_whisper_feature:
                whisper_feature = self.extract_whisper_feat(audio_path)
                kimia_content_msg.continuous_feature.append(whisper_feature)
        elif message["message_type"] == "audio-text":
            audio_path, text = message["content"]
            speech_tokens = self._tokenize_audio(audio_path)
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank]
                * self.kimia_text_audiodelaytokens
            )
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
            kimia_content_msg.text_extend(text_tokens)
            text_pad_tokens = (
                self.kimia_text_audiodelaytokens
                + len(speech_tokens)
                - len(text_tokens)
            ) * [self.extra_tokens.kimia_text_blank]
            kimia_content_msg.text_extend(text_pad_tokens)

        elif message["message_type"] == None:
            pass
        else:
            raise NotImplementedError(
                f"message_type: \
                                      {message['message_type']}"
            )

        if has_msg_end_token:
            kimia_content_msg.audio_append(
                self.extra_tokens.msg_end, audio_token_loss_mask=False
            )
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert (
            kimia_content_msg.is_valid()
        ), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def __call__(
        self,
        messages: List[Dict],
        output_type: str = "text",
        add_assistant_start_msg: bool = True,
    ) -> Dict:
        """
        Build the same outputs that KimiAContent.to_tensor() currently returns:
        returns dict {
            "audio_input_ids": torch.LongTensor (B, S),
            # "text_input_ids": torch.LongTensor (B, S),
            "is_continuous_mask": List[torch.BoolTensor (B, S)],
            "continuous_feature": List[ndarray or torch.Tensor]
        }
        """
        assert output_type in ["text", "both"]

        msgs: List[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None
        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            if previous_role is None:
                tokenize_role = True
            else:
                if message["role"] == previous_role:
                    tokenize_role = False
                else:
                    tokenize_role = True

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                if messages[msg_idx + 1]["role"] != message["role"]:
                    has_ct_token = True
                    has_msg_end_token = True
                else:
                    has_ct_token = False
                    has_msg_end_token = False

            previous_role = message["role"]

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
                extract_whisper_feature=True,
                output_type=output_type,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                message={
                    "role": "assistant",
                    "message_type": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )

            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]

        for msg in msgs[1:]:
            ret_msg.merge(msg)

        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = (
            ret_msg.to_tensor()
        )
        audio_features = ret_msg.continuous_feature

        return dict(
            audio_input_ids=audio_input_ids,
            text_input_ids=text_input_ids,
            is_continuous_mask=is_continuous_mask,
            whisper_input_feature=audio_features,
            text_input_ids=text_input_ids,
        )

    @property
    # NOTE: we don't have default templates anymore, and the below is kept only because the hub config is not yet updated!
    def default_chat_template(self):
        """
        This default vicuna template formats inputs in the form of a chat history. For each message in the chat history:
        * the template will output the role of the speaker followed by the content of the message.
        * content is a list of strings and audios.
        * If the content element is an audio, the template will output a sequence of <|AUDIO|> tokens

        Example:

        ```python
        messages = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
                {"type": "text", "text": "What's that sound?"},
            ]},
            {"role": "assistant", "content": "It is the sound of glass shattering."},
            {"role": "user", "content": [
                {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"},
                {"type": "text", "text": "How about this one?"},
            ]},
        ]

        result = template.render(messages=messages, add_generation_prompt=True)
        ```
        """
        # fmt: off
        return (
            "{% set audio_count = namespace(value=0) %}"
            "{% for message in messages %}"
                "{% if loop.first and message['role'] != 'system' %}"
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "{% endif %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{% if message['content'] is string %}"
                    "{{ message['content'] }}<|im_end|>\n"
                "{% else %}"
                    "{% for content in message['content'] %}"
                        "{% if 'audio' in content or 'audio_url' in content or message['type'] == 'audio' or content['type'] == 'audio' %}"
                            "{% set audio_count.value = audio_count.value + 1 %}"
                            "Audio {{ audio_count.value }}: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
                        "{% elif 'text' in content %}"
                            "{{ content['text'] }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<|im_end|>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "<|im_start|>assistant\n"
            "{% endif %}"
        )
        # fmt: on


AutoProcessor.register("KimiAudioProcessor", KimiAudioProcessor)
