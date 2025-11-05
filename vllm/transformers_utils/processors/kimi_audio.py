# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from https://github.com/MoonshotAI/Kimi-Audio/tree/master/kimia_infer/api/prompt_manager.py
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from subprocess import CalledProcessError, run
from typing import Optional, Union, cast

import numpy as np
import torch
import torch.nn.functional as F
from vllm.multimodal.inputs import NestedTensors
from torch import nn
from transformers import AutoConfig, AutoProcessor, ProcessorMixin
from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import TextInput

from vllm.utils import PlaceholderModule

try:
    import librosa
except ImportError:
    librosa = PlaceholderModule("librosa")  # type: ignore[assignment]

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 120
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk

AUDIO_TYPE = Union[np.ndarray, torch.Tensor]


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
            array = array.index_select(dim=axis,
                                       index=torch.arange(length,
                                                          device=array.device))

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array,
                          [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@cache
def mel_filters(device: Union[str, torch.device],
                n_mels: int = 128) -> torch.Tensor:
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
        The path to audio or either a NumPy array or Tensor containing
        the audio waveform in 16 kHz

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
        audio = torch.from_numpy(np.asarray(audio))

    audio = cast(torch.Tensor, audio)
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio,
                      N_FFT,
                      HOP_LENGTH,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs()**2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec


class WhisperEncoder(nn.Module):

    def __init__(self,
                 model_path,
                 mel_batch_size=40,
                 unfreeze_online_whisper_model=False):
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
            audio_segment = audio[time_step * 16000:(time_step + 30) * 16000]
            audios.append(audio_segment)
            time_step += 30

        final_audio_embedding = []

        for audio_segment in audios:
            # import pdb; pdb.set_trace()
            assert audio_segment.shape[0] <= 480000
            L = audio_segment.shape[0]
            # to match huggingface logic, with use attention mask to
            # control the length and the slice with mask[:, ::160],
            # also match the glm4 12.5 logic
            token_len = (L - 1) // (160 * 8) + 1

            pad_audio = pad_or_trim(audio_segment.flatten())
            mel = log_mel_spectrogram(pad_audio)  # torch.Size([80, 3000])
            assert mel.shape[1] == 3000
            if kimia_whisper_clip_silence:
                input_seq_lens_list = [token_len * 4]
                input_seq_lens = torch.LongTensor(input_seq_lens_list).to(
                    self.speech_encoder.conv1.weight.device)
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0).to(
                        self.speech_encoder.conv1.weight.device).to(
                            torch.bfloat16),
                    return_dict=True,
                    input_seq_lens=input_seq_lens,
                ).last_hidden_state
            else:
                audio_embedding = self.speech_encoder(
                    mel.unsqueeze(0).to(
                        self.speech_encoder.conv1.weight.device).to(
                            torch.bfloat16),
                    return_dict=True,
                ).last_hidden_state
                # audio_embedding: [1, 3000, 1280]
                audio_embedding = audio_embedding[:, :token_len * 4, :]
            final_audio_embedding.append(audio_embedding)

        final_audio_embedding = torch.cat(final_audio_embedding, dim=1)
        return final_audio_embedding

    @torch.no_grad()
    def tokenize_waveform(self, audio, kimia_whisper_clip_silence=False):
        audio_embedding = self.forward(audio, kimia_whisper_clip_silence)
        return audio_embedding.cpu()


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
    assert tokenizer is not None, (
        "tokenizer is None, you must set the text tokenizer manually.")
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
            "<|im_kimia_assistant_msg_start|>"),  # 23
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

        # Storage raw audio consists of sampling rate and raw audio data
        self.audios: list[np.ndarray] = []
        self.continuous_feature: list[np.ndarray] = []

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
        self.audio_token_loss_mask = [audio_token_loss_mask
                                      ] + self.audio_token_loss_mask

    def text_prepend(self, index: int, text_token_loss_mask: bool = False):
        self.text_token_ids = [index] + self.text_token_ids
        self.text_token_loss_mask = [text_token_loss_mask
                                     ] + self.text_token_loss_mask

    def audio_pretend(
        self,
        ids: list[int],
        is_continuous: bool = False,
        audio_token_loss_mask: bool = False,
    ):
        self.audio_token_ids = ids + self.audio_token_ids
        self.is_continuous_mask = [is_continuous
                                   ] * len(ids) + self.is_continuous_mask
        self.audio_token_loss_mask = [audio_token_loss_mask
                                      ] * len(ids) + self.audio_token_loss_mask

    def text_pretend(self, ids: list[int], text_token_loss_mask: bool = False):
        self.text_token_ids = ids + self.text_token_ids
        self.text_token_loss_mask = [text_token_loss_mask
                                     ] * len(ids) + self.text_token_loss_mask

    def merge(self, other: "KimiAContent"):
        self.audio_token_ids.extend(other.audio_token_ids)
        self.text_token_ids.extend(other.text_token_ids)
        self.is_continuous_mask.extend(other.is_continuous_mask)
        self.audio_token_loss_mask.extend(other.audio_token_loss_mask)
        self.text_token_loss_mask.extend(other.text_token_loss_mask)
        self.audios.extend(other.audios)
        self.continuous_feature.extend(other.continuous_feature)

    def to_tensor(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        return (
            torch.tensor([self.audio_token_ids], dtype=torch.long),
            torch.tensor([self.text_token_ids], dtype=torch.long),
            torch.tensor([self.is_continuous_mask], dtype=torch.bool),
            torch.tensor([self.audio_token_loss_mask], dtype=torch.bool),
            torch.tensor([self.text_token_loss_mask], dtype=torch.bool),
        )

    def is_valid(self) -> bool:
        return (len(self.audio_token_ids) == len(self.text_token_ids) == len(
            self.is_continuous_mask) == len(self.audio_token_loss_mask) == len(
                self.text_token_loss_mask))


class KimiAudioProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "audio_kwargs": {},
    }


def ndarray_to_int_list(arr: np.ndarray) -> list[int]:
    """
    Convert a 1-D numpy array that encodes integer token ids (possibly float 
    dtype but integer-valued) into a Python list[int]. Raise if array looks 
    like audio waveform.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError("expected numpy ndarray")
    if arr.ndim != 1:
        arr = arr.ravel()

    if np.issubdtype(arr.dtype, np.floating):
        # check all values are exact integers (x % 1 == 0)
        if not np.all(np.equal(np.mod(arr, 1), 0)):
            raise ValueError(
                "ndarray looks like waveform/floats, not integer token ids")
    return arr.astype(np.int64).tolist()


# TODO(HelloWorldU): Add batch inference support
class KimiAudioProcessor(ProcessorMixin):
    r"""
    Lightweight processor:
    - audio_tokenizer: has method `tokenize(audio_path=...)` ->
    torch.Tensor (1, N) of discrete audio token ids (before offset)
    - text_tokenizer: huggingface tokenizer-like,
    has `encode` or `encode_plus` and `convert_tokens_to_ids`
    """

    attributes = ["audio_tokenizer", "text_tokenizer"]
    audio_tokenizer_class = "Glm4Tokenizer"
    text_tokenizer_class = "TikTokenTokenizer"

    def __init__(
        self,
        kimia_text_audiodelaytokens=5,
        kimia_token_offset=152064,
        audio_tokenizer="THUDM/glm-4-voice-tokenizer",
        text_tokenizer=None,
        chat_template=None,
    ):
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.chat_template = chat_template
        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.kimia_token_offset = kimia_token_offset
        self.pad_token_id = self.extra_tokens.kimia_text_blank
        # Just for simple distinguish
        self.audio_token_id = -1
        self._whisper_config = None
        # Make sure no conflict with text tokenizer
        self.audio_placeholder_id = self.text_tokenizer.vocab_size + 10

    def _tokenize_text(self, text: str) -> list[int]:
        if text is None:
            return None
        if hasattr(self.text_tokenizer, "encode"):
            return self.text_tokenizer.encode(text, bos=False, eos=False)
        else:
            return self.text_tokenizer(text,
                                       add_special_tokens=False)["input_ids"]

    def _get_raw_audio(self,
                       audio_path: str) -> dict[str, Union[str, np.ndarray]]:
        audios = {"path": audio_path}
        audio_data: np.ndarray = librosa.load(**audios, sr=16000)[0]
        audios["data"] = audio_data
        return audios

    def _tokenize_audio(self, audio_path: str) -> list[int]:
        # handle audio placeholder here, using sequence
        # with same length as actual audio tensor
        if self._whisper_config is None:
            self._whisper_config = AutoConfig.from_pretrained(
                self.audio_tokenizer
            )

        count = self.get_num_audio_tokens(audio_path,
                                          self._whisper_config)
        wav_tokens = torch.full((count, ),
                                self.audio_token_id,
                                dtype=torch.long)
        wav_tokens = wav_tokens.cpu().tolist()
        return wav_tokens

    def get_num_audio_tokens(self,
                             audio_path: str,
                             model_config,
                             hop_length=160) -> int:
        import soundfile as sf
        audio_info = sf.info(audio_path)
        audio_length = int(audio_info.frames)

        if model_config is None:
            raise ValueError("model_config for WhisperVQEncoder is None")

        pooling_kernel_size = getattr(model_config, "pooling_kernel_size", 4)
        conv1_stride = getattr(model_config, "conv1.stride", [1])[0]
        conv2_stride = getattr(model_config, "conv2.stride", [2])[0]
        total_downsample = conv1_stride * conv2_stride * pooling_kernel_size

        total_tokens = 0
        stride = total_downsample * hop_length

        for i in range(0, audio_length, 30 * 16000):
            segment_length = min(30 * 16000, audio_length - i)
            padded_length = ((segment_length + stride - 1) // stride) * stride
            n_frames = padded_length // hop_length
            n_tokens = n_frames // total_downsample
            total_tokens += n_tokens

        return total_tokens

    def get_audio_waves(self, audio: Union[str, AUDIO_TYPE]) -> np.ndarray:
        if isinstance(audio, str):
            wav_array = librosa.load(audio, sr=16000)[0]
        elif isinstance(audio, AUDIO_TYPE):
            wav_array = audio
        else:
            raise ValueError(f"Invalid wav type: {type(audio)}")
        wav_array = cast(np.ndarray, wav_array)
        return wav_array

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
                    self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start)
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        if message["message_type"] == "text":
            text = message["content"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens))

            if role == "assistant":
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos,
                                              has_loss)  # eos for text stream
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_text_blank,
                    audio_token_loss_mask=False,
                )

        elif message["message_type"] == "audio":
            # Only support audio_path for input
            if "content" not in message:
                raise ValueError("Expected 'content' key in message")

            audio_path = message["content"]
            if not isinstance(audio_path, str):
                raise ValueError("Expected 'content' to be a string")

            audio = self._get_raw_audio(audio_path)
            speech_tokens = self._tokenize_audio(audio["path"])

            kimia_content_msg.audios.append(audio["data"])
            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(
                speech_tokens,
                is_continuous=True,
                audio_token_loss_mask=has_loss,
            )
            kimia_content_msg.audio_append(
                self.extra_tokens.media_end,
                audio_token_loss_mask=has_loss)  # EOS for audio stream
            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] *
                (len(speech_tokens) + 2))

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id)
                kimia_content_msg.text_append(
                    self.extra_tokens.kimia_text_blank)

            if extract_whisper_feature:
                audio_waves = self.get_audio_waves(audio["data"])
                kimia_content_msg.continuous_feature.append(audio_waves)

        # Not support currently, as official library does utilize this branch
        elif message["message_type"] == "audio-text":
            audio_path, text = message["content"]
            audio = self._get_raw_audio(audio_path)
            speech_tokens = self._tokenize_audio(audio)
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] *
                self.kimia_text_audiodelaytokens)
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
            kimia_content_msg.text_extend(text_tokens)
            text_pad_tokens = (
                self.kimia_text_audiodelaytokens + len(speech_tokens) -
                len(text_tokens)) * [self.extra_tokens.kimia_text_blank]
            kimia_content_msg.text_extend(text_pad_tokens)

        elif message["message_type"] is None:
            pass
        else:
            raise NotImplementedError(
                f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end,
                                           audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert (kimia_content_msg.is_valid()
                ), f"kimia_content_msg is not valid: {kimia_content_msg}"

        return kimia_content_msg

    def handle_prompt(self, messages: KimiAContent) -> dict:
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = (
            messages.to_tensor())
        audio_input_items: list[np.ndarray] = messages.audios
        audio_features: list[np.ndarray] = messages.continuous_feature

        # Consturct text prompt ids, mm_data and mm_processor_kwargs
        text_input_ids = text_input_ids.cpu().tolist()
        mm_data = audio_input_items
        mm_processor_kwargs = dict(
            audio_input_ids=audio_input_ids.cpu().tolist(),
            text_input_ids=text_input_ids,
            is_continuous_mask=is_continuous_mask.cpu().tolist(),
            whisper_input_feature=[torch.as_tensor(f) for f in audio_features],
        )

        return {
            "prompt_token_ids": text_input_ids,
            "multi_modal_data": {
                "audio": mm_data
            },
            "mm_processor_kwargs": mm_processor_kwargs,
        }

    def get_prompt(
        self,
        messages: list[dict],
        output_type: str = "text",
        add_assistant_start_msg: bool = True,
    ) -> dict:
        assert output_type in ["text", "both"]

        msgs: list[KimiAContent] = []
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

        return self.handle_prompt(ret_msg)

    def __call__(
        self,
        text: Optional[Union[TextInput, list[TextInput]]] = None,
        audios: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        **kwargs,
    ) -> Mapping[str, NestedTensors]:
        audio_input_ids = kwargs.get("audio_input_ids")
        text_input_ids = kwargs.get("prompt_token_ids")
        is_continuous_mask = kwargs.get("is_continuous_mask")
        whisper_input_feature = kwargs.get("whisper_input_feature")

        # Generally, we expect the text tokens came from external process
        # in base processor rather than the method in this processor.
        if text_input_ids is None:
            text_input_ids = []
        text_tokens = torch.tensor(text_input_ids, dtype=torch.long)

        if audio_input_ids is None:
            audio_tokens = None
        if not isinstance(audio_input_ids, list):
            audio_input_ids = [audio_input_ids]
        audio_tokens = None

        # TODO(HelloWorldU): Add support for padding in a batch,
        # currently we utilize single tensor as a batch input
        assert isinstance(audio_input_ids, list)
        assert len(audio_input_ids) == 1
        audio_input_ids = audio_input_ids[0]
        for audio_list in audio_input_ids:
            assert isinstance(audio_list, list)

            start_idx = None
            end_idx = None

            for idx, tok in enumerate(audio_list):
                if tok == self.extra_tokens.media_begin:
                    start_idx = idx
                elif tok == self.extra_tokens.media_end:
                    end_idx = idx
                if (start_idx is not None and end_idx is not None
                        and start_idx < end_idx):
                    assert hasattr(self, "audio_placeholder_id"), (
                        "audio_placeholder_id is not set in KimiAudioProcessor"
                    )
                    wav_tokens_list = [self.audio_placeholder_id]
                    audio_tokens = (audio_list[:start_idx] + wav_tokens_list +
                                    audio_list[end_idx:])
                    break

        if audio_tokens:
            audio_tokens = torch.tensor(audio_tokens, dtype=torch.long)

        return {
            "audio_input_ids": audio_tokens,
            "input_ids": text_tokens,
            "is_continuous_mask": is_continuous_mask,
            "whisper_input_feature": whisper_input_feature,
        }


AutoProcessor.register("KimiAudioProcessor", KimiAudioProcessor)
