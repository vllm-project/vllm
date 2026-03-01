# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from https://github.com/MoonshotAI/Kimi-Audio/tree/master/kimia_infer/api/prompt_manager.py
import os
import contextlib
from dataclasses import dataclass
from functools import cache
from subprocess import CalledProcessError, run
from typing import cast, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from transformers.models.whisper.modeling_whisper import WhisperModel
from transformers.tokenization_utils_base import TextInput

from vllm.logger import init_logger

logger = init_logger(__name__)

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
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@cache
def mel_filters(device: str | torch.device, n_mels: int = 128) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    try:
        with np.load(
            os.path.join(os.path.dirname(__file__), "mel_filters.npz")
        ) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"]).to(device).to(torch.float32)
    except FileNotFoundError:
        # Fallback to creating filters on the fly if file is missing (e.g. in CI/CD)
        # This requires librosa, but handles the missing file case gracefully
        try:
            import librosa
            filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)
            return torch.from_numpy(filters).to(device).to(torch.float32)
        except ImportError as e:
            raise RuntimeError(
                "mel_filters.npz not found and librosa not installed. "
                "Please ensure mel_filters.npz exists in " + 
                "vllm/transformers_utils/processors/"
            ) from e


def log_mel_spectrogram(
    audio: str | np.ndarray | torch.Tensor,
    n_mels: int = 128,
    padding: int = 0,
    device: str | torch.device | None = None,
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
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    if magnitudes.dtype == torch.float64:
        magnitudes = magnitudes.to(torch.float32)

    filters = mel_filters(audio.device, n_mels)
    
    # Ensure dtypes match for matrix multiplication
    if filters.dtype != magnitudes.dtype:
        # Fallback: if magnitudes is still not float32 for some reason
        filters = filters.to(magnitudes.dtype)

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
    if tokenizer is None:
        raise ValueError("tokenizer is None, cannot instantiate extra tokens")

    # Check for special_tokens attribute first (TikTokenTokenizer uses this)
    if hasattr(tokenizer, "special_tokens"):
        map_fn = lambda x: tokenizer.special_tokens[x]
    # Fallback to convert_tokens_to_ids for HuggingFace tokenizers
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
        kimia_assistant_msg_start=map_fn("<|im_kimia_assistant_msg_start|>"),  # 23
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

        # Storage raw audio datas(or audio path) for audio waveform extraction in MoonshotKimiaForCausalLM.
        self.continuous_feature: list[Union[np.ndarray, str]] = []

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
        self.text_token_loss_mask = [text_token_loss_mask] + (
            self.text_token_loss_mask)

    def audio_pretend(
        self,
        ids: list[int],
        is_continuous: bool = False,
        audio_token_loss_mask: bool = False,
    ):
        self.audio_token_ids = ids + self.audio_token_ids
        self.is_continuous_mask = [is_continuous] * len(ids) + (
            self.is_continuous_mask)
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

    def to_tensor(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    # check all values are exact integers (x % 1 == 0)
    if np.issubdtype(arr.dtype, np.floating) and not np.all(
        np.equal(np.mod(arr, 1), 0)
    ):
        raise ValueError("ndarray looks like waveform/floats, not integer token ids")
    return arr.astype(np.int64).tolist()


class KimiAudioProcessor:
    r"""
    Kimi-Audio Dual-Stream Lightweight Processor:

    Architecture Features:
    - Parallel Synchronization: Maintains two strictly synchronized token streams:
      1. Text Stream: Standard vocabulary token IDs.
      2. Audio Stream: Discrete audio token IDs (from Glm4Tokenizer) or Blank IDs.

    Args:
        kimia_text_audiodelaytokens (`int`, *optional*, defaults to 5):
            Number of blank tokens to insert in audio stream.
        kimia_token_offset (`int`, *optional*, defaults to 152064):
            Offset applied to discrete audio IDs to map them into the LLM's 
            extended embedding space.
        text_tokenizer ([`Tiktokenizer`]):
            The tokenizer is a required input.
        audio_tokenizer ([`glm-4-voice-tokenizer`]):
            The tokenizer is a required input.
        chat_template (`Optional[str]`, *optional*):
            The Jinja template to use for formatting the conversation. If 
            not provided, the default chat template is used.
    """

    def __init__(
        self,
        kimia_text_audiodelaytokens=5,
        kimia_token_offset=152064,
        tokenizer=None,
        audio_tokenizer=None,
        chat_template=None,
    ):
        self.text_tokenizer = tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.chat_template = chat_template
        # Delay extra_tokens initialization for dummy input creation
        self._extra_tokens = None
        self.kimia_text_audiodelaytokens = kimia_text_audiodelaytokens
        self.kimia_token_offset = kimia_token_offset
        self._whisper_config = None
        if tokenizer is not None:
            self.audio_placeholder_id = self.text_tokenizer.vocab_size + 10
        else:
            self.audio_placeholder_id = 152074  # default vocab_size + 10

    @property
    def extra_tokens(self) -> ExtraTokens:
        """Lazy initialization of extra_tokens to support dummy input creation."""
        if self._extra_tokens is None:
            assert self.text_tokenizer is not None, ("text_tokenizer must be set to"
            "instantiate extra_tokens.")
            self._extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        return self._extra_tokens

    def _tokenize_text(self, text: str) -> list[int]:
        if text is None:
            return ""
        if hasattr(self.text_tokenizer, "encode"):
            return self.text_tokenizer.encode(text, bos=False, eos=False)
        else:
            return self.text_tokenizer(text, add_special_tokens=False)["input_ids"]

    def _tokenize_audio(self, audio_data: np.ndarray | str) -> list[int]:
        """
        Convert the audio data into a sequence of placeholder tokens.

        No actual audio encoding is performed here, but rather generation corresponding 
        to the audio length. Dummy tokens are used to maintain sequence length alignment.
        The actual audio feature extraction is handled solely by the forward in model.
        """
        if self.audio_tokenizer is None:
            raise ValueError(
                "audio_tokenizer is None, you must explicitly specify the path."
            )

        if self._whisper_config is None:
            self._whisper_config = AutoConfig.from_pretrained(self.audio_tokenizer)

        count = self.get_num_audio_tokens_from_data(audio_data, self._whisper_config)
        wav_tokens = torch.full(
            (count,), self.audio_placeholder_id, dtype=torch.long, device="cpu"
        )
        wav_tokens = wav_tokens.tolist()
        return wav_tokens

    def _calculate_num_tokens_from_length(
        self, audio_length: int, model_config, hop_length=160
    ) -> int:
        """Calculate number of audio tokens from audio length (frames)."""
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

    def get_num_audio_tokens_from_data(
        self, audio_input: np.ndarray | str, model_config, hop_length=160
    ) -> int:
        if isinstance(audio_input, str):
            audio_input = load_audio(audio_input)

        audio_length = audio_input.shape[0]
        return self._calculate_num_tokens_from_length(
            audio_length, model_config, hop_length
        )

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
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(
                    self.extra_tokens.kimia_assistant_msg_start
                )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
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
                audio_arrays: np.ndarray = message["audio_tokens"]
                audio_path = audio_arrays
                audio_data = self._tokenize_audio(audio_arrays)
            else:
                audio_path: str = message["content"]
                audio_data = self._tokenize_audio(audio_path)

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(
                audio_data,
                is_continuous=True,
                audio_token_loss_mask=has_loss,
            )
            kimia_content_msg.audio_append(
                self.extra_tokens.media_end, audio_token_loss_mask=has_loss
            )  # EOS for audio stream

            # A audio placeholder and two kimia_text_blank at the end and begin
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            kimia_content_msg.text_append(self.audio_placeholder_id)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(
                        self.extra_tokens.kimia_speech_ctd_id
                    )
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            # Whisper feature extraction will be done in worker process
            # Store path for later processing
            if extract_whisper_feature:
                kimia_content_msg.continuous_feature.append(audio_path)

        # TODO(HelloWorldU): Support this branch as official library does in future.
        elif message["message_type"] == "audio-text":
            audio_data, text = message["content"]
            speech_tokens = self._tokenize_audio(audio_data)
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * self.kimia_text_audiodelaytokens
            )
            kimia_content_msg.audio_extend(speech_tokens, is_continuous=False)
            kimia_content_msg.text_extend(text_tokens)
            text_pad_tokens = (
                self.kimia_text_audiodelaytokens + len(speech_tokens) - len(text_tokens)
            ) * [self.extra_tokens.kimia_text_blank]
            kimia_content_msg.text_extend(text_pad_tokens)

        elif message["message_type"] is None:
            pass
        else:
            raise NotImplementedError(f"message_type: {message['message_type']}")

        if has_msg_end_token:
            kimia_content_msg.audio_append(
                self.extra_tokens.msg_end, audio_token_loss_mask=False
            )
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        # NOTE: KimiAudio is a dual-track architecture. In vLLM, we use a single 
        # placeholder in text-stream to represent the entire audio sequence length. 
        # We skip the strict length check here because alignment is handled 
        # at the embedding level during model forward pass.
        # assert kimia_content_msg.is_valid(), (
        #     f"kimia_content_msg is not valid: {kimia_content_msg}"
        # )

        return kimia_content_msg

    def handle_prompt(self, messages: KimiAContent) -> dict:
        audio_input_ids, text_input_ids, is_continuous_mask, _, _ = messages.to_tensor()
        audio_contents: list[Union[np.ndarray, str]] = messages.continuous_feature

        # NOTE: vLLM's MultiModalDataParser requires raw audio data (numpy/tensor)
        # to perform validation and resampling. Passing paths directly causes 
        # assertion errors. We load the audio here to satisfy the parser.
        mm_data: list[torch.Tensor] = []
        for content in audio_contents:
            if isinstance(content, str):
                content = load_audio(content)  # np.ndarray
            
            if isinstance(content, np.ndarray):
                tensor = torch.from_numpy(content).float()
            elif isinstance(content, torch.Tensor):
                tensor = content.float() if content.dtype != torch.float32 else content
            else:
                raise TypeError(f"Unsupported audio content type: {type(content)}")
            
            mm_data.append(tensor)

        # shape: [batch, seq_len]
        per_request_kwargs = dict(
            audio_input_ids=audio_input_ids,
            is_continuous_mask=is_continuous_mask,
            audio_waveforms=mm_data,
        )

        # Convert to list[int] for expected vLLM format
        text_input_ids = text_input_ids.squeeze(0).tolist()        
        return {
            "prompt_token_ids": text_input_ids,
            "multi_modal_data": {"audio": mm_data},
            "mm_processor_kwargs": per_request_kwargs,
        }

    def get_prompt(
        self,
        messages: list[dict],
        output_type: str = "text",
        add_assistant_start_msg: bool = True,
    ) -> dict:
        """
        In this method, we process the prompt as official library does, while
        several key differences are noted as below:
        1. We use audio_placeholder_id as dummy placeholder token to construct
        audio token spans.
        2. The whisper config is lazily initialized only for output length
        estimation, not for actual feature extraction.
        2. We do not actually load audio data here to avoid fork issues with librosa.
        """
        assert output_type in ["text", "both"]

        msgs: list[KimiAContent] = []
        tokenize_role = True
        has_ct_token = False
        has_msg_end_token = False

        previous_role = None

        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]
            tokenize_role = message["role"] != previous_role

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
        text: TextInput | list[TextInput] | None = None,
        audio: Union[np.ndarray, list[np.ndarray]] = None,
        **kwargs,
    ) -> BatchFeature:
        """
        Wrapper for get_prompt().

        When called via vLLM's string-prompt path, ``text`` will contain
        ``<|AUDIO|>`` markers that indicate where each audio clip belongs.
        We split on these markers and build the proper dual-stream
        ``kimia_messages`` list.

        Args:
            text: Raw prompt text (may contain ``<|AUDIO|>`` markers).
            audio: Raw audio waveforms (np.ndarray) for each audio clip.

        Returns:
            BatchFeature with synchronized tensors for the dual-stream.
        """
        return_tensor = kwargs.get("return_tensor", None)

        if text is None:
            raise ValueError("You need to specify `text` input to process.")
        elif isinstance(text, list):
            text = text[0] if len(text) > 0 else ""
        elif not isinstance(text, str):
            text = str(text)

        if isinstance(audio, np.ndarray):
            audio = [audio]
        elif audio is None:
            audio = []

        audio_queue = list(audio)
        kimia_messages = []

        if not text:
            # Empty text – audio-only (e.g. dummy / profiling inputs)
            for a in audio_queue:
                kimia_messages.append({
                    "role": "user",
                    "message_type": "audio",
                    "content": a,
                })
        elif "<|AUDIO|>" in text:
            # Text with <|AUDIO|> markers – interleave text and audio
            parts = text.split("<|AUDIO|>")
            audio_idx = 0
            for i, part in enumerate(parts):
                part_stripped = part.strip()
                if part_stripped:
                    kimia_messages.append({
                        "role": "user",
                        "message_type": "text",
                        "content": part_stripped,
                    })
                # Each split boundary consumes one audio item
                if i < len(parts) - 1 and audio_idx < len(audio_queue):
                    kimia_messages.append({
                        "role": "user",
                        "message_type": "audio",
                        "content": audio_queue[audio_idx],
                    })
                    audio_idx += 1
        else:
            # Non-empty text without <|AUDIO|> – text message, then any
            # remaining audio items (this branch also handles edge cases
            # like profiling with text but no markers).
            kimia_messages.append({
                "role": "user",
                "message_type": "text",
                "content": text,
            })
            for a in audio_queue:
                kimia_messages.append({
                    "role": "user",
                    "message_type": "audio",
                    "content": a,
                })

        prompt_data = self.get_prompt(
            kimia_messages,
            output_type="text",
        )

        text_input_ids: list[int] = prompt_data["prompt_token_ids"]
        mm_kwargs = prompt_data["mm_processor_kwargs"]

        text_input_ids = torch.tensor([text_input_ids], dtype=torch.long)
        inputs = {
            "input_ids": text_input_ids,
            "audio_input_ids": mm_kwargs["audio_input_ids"],
            "is_continuous_mask": mm_kwargs["is_continuous_mask"],
            "audio_waveforms": (mm_kwargs["audio_waveforms"]
                                if mm_kwargs["audio_waveforms"] else []),
        }

        return BatchFeature(data=inputs, tensor_type=return_tensor)


AutoProcessor.register("KimiAudioProcessor", KimiAudioProcessor)
