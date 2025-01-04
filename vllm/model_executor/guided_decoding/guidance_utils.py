import logging
import re
import textwrap
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
import transformers as transformers_package
from pydantic import (BaseModel, Field, NonNegativeInt, RootModel,
                      model_validator)
from typing_extensions import Annotated


class ChatTemplate:
    """Contains template for all chat and instruct tuned models."""

    def get_role_start(self, role_name: str, **kwargs):
        raise NotImplementedError(
            "You need to use a ChatTemplate subclass that \
                overrides the get_role_start method")

    def get_role_end(self, role_name: Union[str, None] = None):
        raise NotImplementedError("You need to use a ChatTemplate subclass \
                that overrides the get_role_start method")


class Tokenizer:
    """This is the standardized tokenizer interface used by guidance models.

    This class should be subclassed by 
    specific implementations and then used as the
    tokenizer in the corresponding Engine subclass.
    """

    def __init__(
        self,
        tokens: Union[Sequence[bytes], np.ndarray],
        chat_template: Union[str, ChatTemplate, None],
        bos_token_id: Union[int, None] = None,
        eos_token_id: Union[int, None] = None,
    ):

        # a numpy array of token byte strings indexed by their token id
        if isinstance(tokens, list):
            # note that we need np.bytes_
            # so zero bytes are not treated as null terminations
            self._tokens = np.array(tokens, dtype="object")

        # a numpy array of token byte strings indexed by their token id
        elif isinstance(tokens, np.ndarray):
            self._tokens = tokens

        else:
            raise ValueError("Unknown tokenizer was passed!")

        assert isinstance(self.tokens[0],
                          bytes), "The tokens need to be provided as bytes!"

        # This method supports None, a huggingface style jinja2_template_str,
        # or a ChatTemplate subclass
        # Defaults to ChatML if nothing is found
        # self._chat_template = load_template_class(chat_template)

        self._bos_token_id: Union[int, None] = bos_token_id
        self._bos_token: Union[bytes,
                               None] = (None if self.bos_token_id is None else
                                        self.tokens[self.bos_token_id])
        self._eos_token_id: Union[
            int,
            None] = eos_token_id if eos_token_id is not None else bos_token_id
        self._eos_token: Union[bytes,
                               None] = (None if self.eos_token_id is None else
                                        self.tokens[self.eos_token_id])

        # track which tokens are duplicates
        self._duplicate_tokens: list[tuple[int, int]] = []
        found: Dict[bytes, int] = {}
        for i, t in enumerate(self.tokens):
            if t in found:
                self._duplicate_tokens.append((i, found[t]))
            else:
                found[t] = i

    @property
    def tokens(self) -> np.ndarray:
        return self._tokens

    @property
    def bos_token_id(self) -> Union[int, None]:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> Union[int, None]:
        return self._eos_token_id

    @property
    def bos_token(self) -> Union[bytes, None]:
        return self._bos_token

    @property
    def eos_token(self) -> Union[bytes, None]:
        return self._eos_token

    def __call__(self, byte_string: bytes):
        return self.encode(byte_string)

    def encode(self, byte_string: bytes) -> list[int]:
        """Returns a list of tokens that represent the given byte string."""
        raise NotImplementedError("You need to use a Tokenize subclass \
                that overrides the encode method")

    def decode(self, tokens: Sequence[int]) -> bytes:
        """Returns the bytes represented by the given list of tokens."""
        tokens_string = "".join([str(self.tokens[t]) for t in tokens])
        return tokens_string.encode()

    def recode(self, tokens: Sequence[int]) -> list[int]:
        """Redoes a tokenisation.

        Encoding a string into tokens does not distribute over concatenation.
        That is, in general, `encode(A)+encode(B) != encode(A+B)` (although it
        it may in some cases).
        An LLM will generate token-by-token, \
            but it is possible (even likely) that
        when the generation is considered as a whole, \
            a better tokenisation may
        be possible.
        This method takes in a sequence of tokens, \
            and returns an 'improved' sequence.
        """

        # This is the notional behaviour
        # It may need to be overridden in particular cases because
        # we are dealing with LLM ecosystems in the real world
        return self.encode(self.decode(tokens))

    def clean_duplicate_tokens(self, probs):
        """This moves all the probability mass \
            from duplicate positions on \
            to their primary index."""
        for i, j in self._duplicate_tokens:
            probs[j] += probs[i]
            probs[i] = 0


class ByteDecoderError(Exception):
    pass


class ByteTokensError(Exception):
    pass


class TransformersTokenizer(Tokenizer):

    def __init__(
        self,
        model: Union[str, "transformers_package.PreTrainedModel"],
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast", None, ],
        chat_template=None,
        ignore_bos_token=False,
        **kwargs,
    ):
        if transformers_tokenizer is None:
            if isinstance(model, str):
                transformers_tokenizer, byte_tokens = self._tokenizer(
                    model, **kwargs)
            else:
                raise ValueError("A model object was passed in, \
                        but no tokenizer was provided. \
                        Please provide a tokenizer.")
        else:
            is_ptt = isinstance(transformers_tokenizer,
                                transformers_package.PreTrainedTokenizer)
            is_ptt_fast = isinstance(
                transformers_tokenizer,
                transformers_package.PreTrainedTokenizerFast)
            assert is_ptt or is_ptt_fast
            byte_tokens = self._byte_tokens(transformers_tokenizer)

        self._orig_tokenizer = transformers_tokenizer

        # Chat Template logic
        if chat_template is None and hasattr(self._orig_tokenizer,
                                             "chat_template"):
            chat_template = self._orig_tokenizer.chat_template

        # the superclass does most of the work once we have the tokens
        super().__init__(
            byte_tokens,
            chat_template,
            None if ignore_bos_token else transformers_tokenizer.bos_token_id,
            transformers_tokenizer.eos_token_id,
        )

    def _tokenizer(
        self, model: str, **kwargs
    ) -> tuple[Union["transformers_package.PreTrainedTokenizer",
                     "transformers_package.PreTrainedTokenizerFast", ],
               list[bytes], ]:

        try:
            tokenizer = transformers_package.AutoTokenizer.from_pretrained(
                model, use_fast=False, **kwargs)
            byte_tokens = self._byte_tokens(tokenizer)
        except ImportError:
            # Raise on ImportError because it's likely
            # a missing dependency that the user can install
            raise
        except ByteTokensError as e:
            # Give a specific warning for ByteTokensError
            # and fall back to fast tokenizer
            error_log = f"Falling back to fast tokenizer. \
                Could not build byte tokens for model {model!r} \
                due to exception {e.__class__.__name__}: {e}"

            logging.warning(error_log)
        except Exception as e:
            # Fall back for other exceptions
            error_log = f"Falling back to fast tokenizer. \
                Could not load tokenizer for model {model!r} \
                due to exception {e.__class__.__name__}: {e}"

            logging.warning(error_log)
        else:
            return tokenizer, byte_tokens

        tokenizer = transformers_package.AutoTokenizer.from_pretrained(
            model, use_fast=True, **kwargs)
        try:
            byte_tokens = self._byte_tokens(tokenizer)
        except ByteTokensError as e:
            raise ValueError(
                f"Fallback to fast tokenizer failed for model {model!r}"
            ) from e
        return tokenizer, byte_tokens

    def _byte_tokens(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast", ],
    ) -> list[bytes]:

        if hasattr(transformers_tokenizer, "byte_decoder"):
            try:
                self._check_byte_decoder(transformers_tokenizer.byte_decoder,
                                         transformers_tokenizer)
            except ByteDecoderError as e:
                error_log = f"Tokenizer has a byte_decoder, \
                    but it can't be used to construct byte_tokens: {e}"

                logging.warning(error_log)
                pass
            else:
                return self._byte_tokens_from_byte_decoder(
                    transformers_tokenizer.byte_decoder,
                    transformers_tokenizer)

        if hasattr(transformers_tokenizer, "sp_model"):
            return self._byte_tokens_from_sp_model(transformers_tokenizer)

        try:
            return self._byte_tokens_by_encoding_token_strings(
                transformers_tokenizer)
        except ValueError as e:
            error_log = f"Could not build byte tokens from the \
                            tokenizer by encoding token strings: {e}"

            logging.warning(error_log)
            pass

        fallback_byte_decoder = self._fallback_byte_decoder()
        try:
            self._check_byte_decoder(fallback_byte_decoder,
                                     transformers_tokenizer)
        except ByteDecoderError as e:
            # Should be the only exception that is raised in _byte_tokens
            raise ByteTokensError(
                "Could not build byte tokens from the tokenizer, \
                    and falling back to a standard gpt2 byte_decoder failed"
            ) from e
        return self._byte_tokens_from_byte_decoder(fallback_byte_decoder,
                                                   transformers_tokenizer)

    def _byte_tokens_from_byte_decoder(
        self,
        byte_decoder: dict[str, int],
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast", ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        for i in range(len(transformers_tokenizer)):
            byte_coded = bytes([
                byte_decoder[c]
                for c in transformers_tokenizer.convert_ids_to_tokens(i)
            ])
            byte_tokens[i] = byte_coded
        return byte_tokens

    def _byte_tokens_from_sp_model(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast", ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        special_tokens_map = {
            id: token
            for token, id in transformers_tokenizer.get_added_vocab().items()
        }
        space_prefix = "▁".encode()
        for i in range(len(transformers_tokenizer)):
            if i in special_tokens_map:
                byte_coded = special_tokens_map[i].encode()
            else:
                byte_coded = re.sub(
                    rb"<0x(..)>",
                    lambda x: bytes.fromhex(x[1].decode()),
                    transformers_tokenizer.sp_model.id_to_piece(i).encode(),
                )
            byte_tokens[i] = byte_coded.replace(space_prefix, b" ")
        return byte_tokens

    def _byte_tokens_by_encoding_token_strings(
        self,
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast", ],
    ) -> list[bytes]:
        byte_tokens = [b""] * len(transformers_tokenizer)
        special_tokens_map = {
            id: token
            for token, id in transformers_tokenizer.get_added_vocab().items()
        }
        byte_encoder = self._bytes_to_unicode()
        byte_decoder = {v: k for k, v in byte_encoder.items()}

        for i in range(len(transformers_tokenizer)):
            if i in special_tokens_map:
                byte_coded = special_tokens_map[i].encode()
            else:
                token = transformers_tokenizer.convert_ids_to_tokens(i)
                if isinstance(token, bytes):
                    byte_coded = token
                elif isinstance(token, str):
                    if hasattr(transformers_tokenizer,
                               "convert_tokens_to_string"):
                        token_str = \
                            transformers_tokenizer.convert_tokens_to_string([token])
                        encoded_str = transformers_tokenizer.encode(token_str)
                        if len(encoded_str) != 1:
                            raise ValueError(f"Round-trip encoding of tokens \
                                    [{token}] failed! Got {encoded_str}")
                        roundtrip_id = encoded_str[0]
                        if roundtrip_id == i:
                            byte_coded = token_str.encode()
                        else:
                            byte_coded = bytes(
                                [byte_decoder[c] for c in token])
                    else:
                        byte_coded = token.encode()
                else:
                    raise ValueError(f"Unexpected token type: {type(token)}")
            byte_tokens[i] = byte_coded
        return byte_tokens

    def _fallback_byte_decoder(self) -> dict[str, int]:
        byte_decoder = transformers_package.AutoTokenizer.from_pretrained(
            "gpt2", use_fast=False).byte_decoder  # fall back to gpt2 mapping

        # some special tokens may not have their whitespace encoded...
        byte_decoder[" "] = 32
        byte_decoder["\n"] = 10
        byte_decoder["\r"] = 13
        byte_decoder["\t"] = 9
        byte_decoder["▁"] = 32

        return byte_decoder

    def _check_byte_decoder(
        self,
        byte_decoder: dict[str, int],
        transformers_tokenizer: Union[
            "transformers_package.PreTrainedTokenizer",
            "transformers_package.PreTrainedTokenizerFast", ],
    ) -> None:

        def check_byte_decoder_has_all_bytes() -> None:
            # This is here because some tokenizers are bad
            # and don't have all the bytes
            # (I'm looking at you, microsoft/phi2)
            all_bytes = set()
            for x in transformers_tokenizer.get_vocab():
                for y in x:
                    all_bytes.add(y)
            if not set(byte_decoder.keys()) >= all_bytes:
                raise ByteDecoderError(f"Byte decoder is missing bytes: \
                        {all_bytes - set(byte_decoder.keys())}")

        def check_byte_decoder_complex_round_trip() -> None:
            # run a quick spot check to verify if
            # we can rebuild complex multi-token unicode symbols
            s = "’•¶∂ƒ˙∆£Ħ爨ൠᅘ∰፨"
            reconstructed = b""
            try:
                input_ids = transformers_tokenizer(s)["input_ids"]
                for i in input_ids:
                    nxt_bytes = []
                    token_str = transformers_tokenizer.convert_ids_to_tokens(i)
                    for c in token_str:
                        nxt_bytes.append(byte_decoder[c])
                    reconstructed += bytes(nxt_bytes)
                # Check if the tokenizer has a bos_token attribute,
                # and if it does,
                # check if it's at the start of the reconstructed bytes
                # Some tokenizers add this automatically
                # as part of the call function, so
                # we need to remove it to compare
                if hasattr(
                        transformers_tokenizer, "bos_token"
                ) and transformers_tokenizer.bos_token \
                  and reconstructed.startswith(
                        transformers_tokenizer.bos_token.encode()):
                    reconstructed = reconstructed[len(transformers_tokenizer.
                                                      bos_token):]
            # TODO: can we narrow this exception?
            except Exception as e:
                msg = textwrap.dedent(f"""
                    The tokenizer being used is unable \
                        to convert a special character in {s}.
                    For models with sentencepiece based tokenizers \
                        (e.g. llama, phi-3-mini),
                    installing sentencepiece often fixes this issue: \
                          (pip install sentencepiece).
                    """)
                raise ByteDecoderError(msg) from e
            if reconstructed.decode() != s:
                raise ByteDecoderError(f"Failed to reconstruct the string {s} \
                        from the tokenizer's byte_decoder: \
                        {reconstructed.decode()!r} != {s!r}")

        check_byte_decoder_has_all_bytes()
        check_byte_decoder_complex_round_trip()

    def _bytes_to_unicode(self):
        bs = (list(range(ord("!"),
                         ord("~") + 1)) + list(range(ord("¡"),
                                                     ord("¬") + 1)) +
              list(range(ord("®"),
                         ord("ÿ") + 1)))
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        _cs = [chr(n) for n in cs]
        return dict(zip(bs, _cs))

    def encode(self, byte_string: bytes) -> list[int]:
        assert isinstance(byte_string, bytes)
        # HF tokenizers take in strings apparently
        tokenization = self._orig_tokenizer(byte_string.decode(),
                                            add_special_tokens=False)
        return tokenization["input_ids"]

    def decode(self, tokens: Sequence[int]) -> bytes:
        decoded_str = self._orig_tokenizer.decode(tokens)
        return decoded_str.encode()

    def recode(self, tokens: Sequence[int]) -> list[int]:
        # the encode/decode cycle might not work
        # if we have partial unicode strings
        used_tokens = len(tokens)
        for _ in range(3):
            try:
                first_decode = self.decode(tokens).decode("utf8")
            except UnicodeDecodeError:
                if used_tokens == 0:
                    break
                else:
                    used_tokens -= 1

        new_ids = list(self.encode(first_decode.encode("utf-8")))
        if used_tokens < len(tokens):
            new_ids += tokens[used_tokens:]

        # HACK: check for a bug in the HuggingFace tokenizer
        # (that will just add extra spaces during an encode-decode cycle)
        second_decode = self._orig_tokenizer.decode(new_ids)
        if (second_decode != first_decode
                and len(second_decode) == len(first_decode) + 1
                and second_decode.startswith("<s>  ")):
            new_ids = new_ids[0:1] + new_ids[2:]

        return new_ids


class LLProgressCapture(BaseModel):
    object: Literal["capture"]
    name: str
    hex: str
    log_prob: float
    list_append: bool = False

    @model_validator(mode="before")
    def strip_list_append_prefix(cls, values):
        name = values["name"]
        if name.startswith("__LIST_APPEND:"):
            values["name"] = name[14:]
            # Override whatever was set
            values["list_append"] = True
        return values


class LLProgressText(BaseModel):
    object: Literal["text"]
    hex: str
    num_tokens: NonNegativeInt
    log_prob: float
    is_generated: bool


class LLProgressFinalText(BaseModel):
    object: Literal["final_text"]


LLProgressItem = Annotated[Union[LLProgressCapture, LLProgressText,
                                 LLProgressFinalText],
                           Field(discriminator="object"), ]


class EngineCallResponse(BaseModel):
    new_bytes: bytes
    is_generated: bool
    new_bytes_prob: float
    capture_groups: dict
    capture_group_log_probs: dict
    new_token_count: NonNegativeInt


class LLProgress(RootModel):
    root: list[LLProgressItem]

    def to_engine_call_response(self) -> EngineCallResponse:
        new_bytes = b""
        new_token_count = 0
        new_bytes_prob = 0.0
        is_generated = False
        capture_groups: dict[str, Any] = {}
        capture_group_log_probs: dict[str, Any] = {}
        num_text_entries = 0

        for j in self.root:
            if isinstance(j, LLProgressCapture):
                is_generated = True
                cname = j.name
                data = bytes.fromhex(j.hex)
                if j.list_append:
                    if cname not in capture_groups or not isinstance(
                            capture_groups[cname], list):
                        capture_groups[cname] = []
                        capture_group_log_probs[cname] = []
                    capture_groups[cname].append(data)
                    capture_group_log_probs[cname].append(j.log_prob)
                else:
                    capture_groups[cname] = data
                    capture_group_log_probs[cname] = j.log_prob
            elif isinstance(j, LLProgressText):
                # it actually should only happen once per round...
                new_bytes += bytes.fromhex(j.hex)
                new_token_count += j.num_tokens
                new_bytes_prob += j.log_prob
                is_generated |= j.is_generated
                num_text_entries += 1
        if num_text_entries > 0:
            new_bytes_prob /= num_text_entries

        return EngineCallResponse(
            new_bytes=new_bytes,
            new_token_count=new_token_count,
            new_bytes_prob=new_bytes_prob,
            is_generated=is_generated,
            capture_groups=capture_groups,
            capture_group_log_probs=capture_group_log_probs,
        )


class LLInterpreterResponse(BaseModel):
    progress: LLProgress
    stop: bool
    temperature: Optional[float]
