# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import json
import os

import sentencepiece as spm
from transformers import SPIECE_UNDERLINE, PreTrainedTokenizer
from transformers.models.whisper.english_normalizer import (
    BasicTextNormalizer,
    EnglishTextNormalizer,
)
from transformers.processing_utils import ProcessorMixin
from transformers.utils import cached_file, is_offline_mode
from transformers.utils.import_utils import requires

CMD_ASR_BOS = "<|startoftranscript|>"
CMD_ASR_EOS = "<|endoftext|>"
CMD_ASR_PAD = "<pad>"
CMD_ASR_UNK = "<unk>"


@requires(backends=("sentencepiece",))
class CohereASRTokenizer(PreTrainedTokenizer, ProcessorMixin):
    # class CohereASRTokenizer(TokenizerLike, PreTrainedTokenizer, ProcessorMixin):
    """
    Minimal HF 'slow' tokenizer wrapper around a SentencePiece BPE model.
    Preserves SPM behaviors like --byte_fallback

    # TODO: check additional_special_tokens in SPM model
    """

    def __init__(
        self,
        spm_model_file: str,
        bos_token=CMD_ASR_BOS,
        eos_token=CMD_ASR_EOS,
        unk_token=CMD_ASR_UNK,
        pad_token=CMD_ASR_PAD,
        additional_special_tokens=None,
        split_special_tokens=False,
        add_prefix_space=False,
        sp_model_kwargs: dict | None = None,
        **kwargs,
    ):
        self.spm_model_file = spm_model_file
        self.sp_model_kwargs = sp_model_kwargs or {}
        self.sp_model = self.get_spm_processor()
        self.vocab_files_names = {"spm_file": os.path.basename(spm_model_file)}
        self.model_input_names = ["input_ids"]

        self.add_prefix_space = add_prefix_space

        super().__init__(
            unk_token=unk_token,
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            additional_special_tokens=additional_special_tokens or [],
            split_special_tokens=split_special_tokens,
            add_prefix_space=add_prefix_space,
            **kwargs,
        )
        self.legacy = kwargs.get("legacy", False)
        self.init_kwargs["sp_model_kwargs"] = dict(self.sp_model_kwargs)

        # normalizer_file is in same directory as tokenizer.model
        normalizer_file = os.path.join(
            os.path.dirname(spm_model_file), "normalizer.json"
        )
        assert os.path.isfile(normalizer_file), (
            f"Expected normalizer.json next to {spm_model_file}"
        )
        with open(normalizer_file, encoding="utf-8") as vocab_handle:
            self.english_spelling_normalizer = json.load(vocab_handle)

        self._max_token_id = self.vocab_size - 1
        self._max_chars_per_token = max(
            len(self.sp_model.id_to_piece(i)) for i in range(self.vocab_size)
        )

    @property
    def unk_token_length(self):
        return len(self.sp_model.encode(str(self.unk_token)))

    def get_vocab(self):
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        added_tokens = self.added_tokens_decoder
        for token_id, added_token in added_tokens.items():
            if added_token.content not in vocab:
                vocab[added_token.content] = token_id
            else:
                assert vocab[added_token.content] == token_id, (
                    f"Conflict when trying to add the token {added_token.content} "
                    f"to the vocabulary. This token is already associated with "
                    f"the id {vocab[added_token.content]}, while you are trying "
                    f"to associate it with the id {token_id}."
                )
        return vocab

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    def _tokenize(self, text, **kwargs):
        pieces = self.sp_model.encode(text, out_type=str)

        if text and text[0] == " " and (not pieces or pieces[0] != SPIECE_UNDERLINE):
            pieces = [SPIECE_UNDERLINE] + pieces

        return pieces

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.id_to_piece(index)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return (
            [self.bos_token_id]
            + token_ids_0
            + [self.eos_token_id]
            + token_ids_1
            + [self.eos_token_id]
        )

    def normalize(self, text):
        """
        Normalize a given string using the `EnglishTextNormalizer` class, which
        performs common transformations on english text.
        """
        normalizer = EnglishTextNormalizer(self.english_spelling_normalizer)
        return normalizer(text)

    @staticmethod
    def basic_normalize(text, remove_diacritics=False):
        """
        Normalize a given string using the `BasicTextNormalizer` class, which
        performs common transformations on multilingual text.
        """
        normalizer = BasicTextNormalizer(remove_diacritics=remove_diacritics)
        return normalizer(text)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        os.makedirs(save_directory, exist_ok=True)
        out_name = (
            filename_prefix + "-" if filename_prefix else ""
        ) + "tokenizer.model"
        dest = os.path.join(save_directory, out_name)

        if not os.path.exists(dest):
            with open(dest, "wb") as d:
                proto = self.sp_model.serialized_model_proto()
                d.write(proto)
        return (dest,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        # Try to resolve the tokenizer.model file whether local or remote (HF Hub)
        spm_path = None

        # Case 1: Local directory or file
        local_path = os.path.join(pretrained_model_name_or_path, "tokenizer.model")
        if os.path.exists(local_path):
            spm_path = local_path
        else:
            # Case 2: Hugging Face Hub repo — download or cache the file
            try:
                spm_path = cached_file(
                    pretrained_model_name_or_path,
                    "tokenizer.model",
                    _raise_exceptions_for_missing_entries=True,
                )
            except OSError as e:
                if is_offline_mode():
                    raise ValueError(
                        f"Offline mode: couldn't find tokenizer.model for "
                        f"{pretrained_model_name_or_path} in local cache."
                    ) from e
                raise ValueError(
                    f"tokenizer.model not found in "
                    f"{pretrained_model_name_or_path} (local or remote): {e}"
                ) from e

        return super().from_pretrained(
            pretrained_model_name_or_path,
            *init_inputs,
            spm_model_file=spm_path,
            **kwargs,
        )

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        if already_has_special_tokens:
            # assume caller already inserted specials; mark any special ids as 1
            def is_special(tid):
                return tid in {
                    self.bos_token_id,
                    self.eos_token_id,
                    self.pad_token_id,
                    self.unk_token_id,
                    *[
                        self.convert_tokens_to_ids(t)
                        for t in (self.additional_special_tokens or [])
                    ],
                }

            return [1 if is_special(t) else 0 for t in token_ids_0]

        if token_ids_1 is None:
            return [1] + [0] * len(token_ids_0) + [1]
        return [1] + [0] * len(token_ids_0) + [1] + [0] * len(token_ids_1) + [1]

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False, (
            f"Pair sequences not supported for {self.__class__.__name__}."
        )
        return 2

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        if not tokens:
            return ""
        # since we manually add the prefix space, we have to remove it when decoding
        if self.add_prefix_space and tokens[0].startswith(SPIECE_UNDERLINE):
            tokens = [tokens[0][1:]] + tokens[1:]
        out = []
        buf: list[str] = []
        prev_was_special = False

        def flush_buf():
            nonlocal buf, prev_was_special
            if not buf:
                return
            # If a special preceded this buffer and the buffer starts with ▁,
            # we must emit a real space.
            if prev_was_special and buf[0].startswith(SPIECE_UNDERLINE):
                out.append(" ")
            out.append(self.sp_model.decode(buf))
            buf = []
            prev_was_special = False

        for tok in tokens:
            if tok in self.all_special_tokens:
                # finish any pending normal text (adding a boundary space if needed)
                flush_buf()
                out.append(tok)
                prev_was_special = True
            else:
                buf.append(tok)

        flush_buf()

        return "".join(out)

    def get_spm_processor(self, from_slow=False):
        tokenizer = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        tokenizer.Load(self.spm_model_file)
        return tokenizer

    @property
    def max_token_id(self) -> int:
        return self._max_token_id

    @property
    def max_chars_per_token(self) -> int:
        return self._max_chars_per_token

    # Copied from transformers.models.t5.tokenization_t5.T5Tokenizer.__getstate__
    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d

        self.sp_model = self.get_spm_processor()
