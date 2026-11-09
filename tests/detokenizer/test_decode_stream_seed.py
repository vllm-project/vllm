# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fast detokenizer primes its `DecodeStream` with a short prompt tail.

The tail must stream exactly the text that priming with the whole prompt
streams. Besides real tokenizers, one tokenizer is built in-process per
decoder that depends on context: byte-level and byte-fallback UTF-8, `Strip`
after `Fuse`, Metaspace, WordPiece, the BPE end-of-word suffix and CTC.
"""

import random

import pytest
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from transformers import AutoTokenizer, TokenizersBackend

from vllm.sampling_params import SamplingParams
from vllm.tokenizers.detokenizer_utils import (
    INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET,
)
from vllm.transformers_utils.repo_utils import with_retry
from vllm.v1.engine import EngineCoreRequest, detokenizer
from vllm.v1.engine.detokenizer import (
    FastIncrementalDetokenizer,
    get_decode_stream_seed,
)

TEXT = (
    "Hi.  café naïve e\u0301 我很感谢你的热情 🙂🌶️ 👩\u200d💻 𠜎 ပုံပြင် "
    "the is ok , isn't it ? i do not          know   "
)
CONTINUATION = " Sure 🙂 the thing , 我 isn't ."
EOS = "</s>"
CORPUS = [TEXT, CONTINUATION, "the is ok Sure the thing Hello world"] * 20
SPECIAL_RUNS = (0, 1, 2, 7, 40)


def _train(tok: Tokenizer, trainer) -> Tokenizer:
    tok.train_from_iterator(CORPUS, trainer)
    return tok


def _byte_level() -> Tokenizer:
    # ASCII-only merges: non-ASCII characters stay split into byte tokens.
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["Hi. the is ok Sure the thing Hello world"] * 20,
        trainers.BpeTrainer(
            vocab_size=330,
            special_tokens=[EOS],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return tok


def _byte_fallback(strip_start: int) -> Tokenizer:
    # SentencePiece-style BPE: byte tokens for anything outside the vocab and
    # a decoder that strips the leading space(s) of the fused text.
    vocab = {"<unk>": 0, "<s>": 1, EOS: 2}
    for i in range(256):
        vocab[f"<0x{i:02X}>"] = len(vocab)
    pieces = ["▁", *"abcdefghijklmnopqrstuvwxyz.,?'", "▁t", "▁th", "▁the"]
    for piece in [*pieces, "▁i", "▁is"]:
        vocab[piece] = len(vocab)
    merges = [("▁", "t"), ("▁t", "h"), ("▁th", "e"), ("▁", "i"), ("▁i", "s")]
    tok = Tokenizer(
        models.BPE(vocab=vocab, merges=merges, byte_fallback=True, unk_token="<unk>")
    )
    tok.add_special_tokens(["<s>", EOS])
    tok.normalizer = normalizers.Sequence(
        [normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")]
    )
    tok.decoder = decoders.Sequence(
        [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Strip(" ", strip_start, 0),
        ]
    )
    return tok


def _metaspace(prepend_scheme: str) -> Tokenizer:
    tok = Tokenizer(models.Unigram())
    tok.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme=prepend_scheme)
    tok.decoder = decoders.Metaspace(prepend_scheme=prepend_scheme)
    return _train(
        tok,
        trainers.UnigramTrainer(
            vocab_size=120, special_tokens=["<unk>", EOS], unk_token="<unk>"
        ),
    )


def _wordpiece() -> Tokenizer:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=False)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.decoder = decoders.WordPiece(cleanup=True)
    return _train(
        tok, trainers.WordPieceTrainer(vocab_size=150, special_tokens=["[UNK]", EOS])
    )


def _bpe_suffix() -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>", end_of_word_suffix="</w>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.decoder = decoders.BPEDecoder(suffix="</w>")
    return _train(
        tok,
        trainers.BpeTrainer(
            vocab_size=150, special_tokens=["<unk>", EOS], end_of_word_suffix="</w>"
        ),
    )


def _ctc() -> Tokenizer:
    chars = sorted(set("".join(CORPUS)) - {" "})
    vocab = {t: i for i, t in enumerate(["<pad>", "<unk>", EOS, "|", *chars])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.add_special_tokens([EOS])
    tok.normalizer = normalizers.Replace(" ", "|")
    tok.pre_tokenizer = pre_tokenizers.Split(Regex("."), "isolated")
    tok.decoder = decoders.CTC(pad_token="<pad>", word_delimiter_token="|")
    return tok


IN_PROCESS_TOKENIZERS = {
    "byte_level": _byte_level,
    "byte_fallback_strip1": lambda: _byte_fallback(1),
    "byte_fallback_strip8": lambda: _byte_fallback(8),
    "metaspace_first": lambda: _metaspace("first"),
    "metaspace_always": lambda: _metaspace("always"),
    "metaspace_never": lambda: _metaspace("never"),
    "wordpiece": _wordpiece,
    "bpe_suffix": _bpe_suffix,
    "ctc": _ctc,
}
HUB_TOKENIZERS = [
    "facebook/opt-125m",
    "openai-community/gpt2",
    "Qwen/Qwen3-0.6B",
    "bigcode/tiny_starcoder_py",
    "EleutherAI/pythia-70m",
    "bigscience/bloom-560m",
    "tiiuae/falcon-7b",
    "codellama/CodeLlama-7b-hf",
    "google-bert/bert-base-uncased",
    "google-t5/t5-small",
    "FacebookAI/xlm-roberta-base",
]


@pytest.fixture(scope="module", params=[*IN_PROCESS_TOKENIZERS, *HUB_TOKENIZERS])
def hf_tokenizer(request) -> TokenizersBackend:
    if request.param in IN_PROCESS_TOKENIZERS:
        return TokenizersBackend(
            tokenizer_object=IN_PROCESS_TOKENIZERS[request.param](), eos_token=EOS
        )
    tokenizer = with_retry(
        lambda: AutoTokenizer.from_pretrained(request.param),
        log_msg=f"Error loading tokenizer {request.param}",
    )
    assert isinstance(tokenizer, TokenizersBackend)
    return tokenizer


def _request(prompt_token_ids: list[int], skip_special_tokens: bool):
    return EngineCoreRequest(
        request_id="",
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=SamplingParams(skip_special_tokens=skip_special_tokens),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def _assert_streams_like_full_prompt(
    monkeypatch, hf_tokenizer, prompt: list[int], continuation: list[int], skip: bool
) -> None:
    backend = hf_tokenizer.backend_tokenizer
    if not backend.decode(prompt, skip_special_tokens=skip).endswith("\ufffd"):
        seed = get_decode_stream_seed(backend, prompt, skip)
        assert seed is not None
        seed_tokens_with_text = sum(
            bool(backend.decode([t], skip_special_tokens=skip)) for t in seed
        )
        assert seed_tokens_with_text <= INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET + 3
    request = _request(prompt, skip)
    tail = FastIncrementalDetokenizer(hf_tokenizer, request)
    with monkeypatch.context() as m:
        m.setattr(detokenizer, "get_decode_stream_seed", lambda tok, ids, skip: ids)
        full = FastIncrementalDetokenizer(hf_tokenizer, request)
    expected = [full.decode_next(t) for t in continuation]
    assert [tail.decode_next(t) for t in continuation] == expected, (prompt, skip)


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_prompt_tail_streams_same_text_as_full_prompt(
    monkeypatch, hf_tokenizer, skip_special_tokens
):
    """Every cut of the prompt, followed by runs of special tokens (chat
    end-of-turn markers, padding), streams the text of the full-prompt seed.
    Cuts inside multi-byte characters, combining marks, emoji sequences and
    whitespace runs are included."""
    ids = hf_tokenizer.encode(TEXT, add_special_tokens=False)
    continuation = hf_tokenizer.encode(CONTINUATION, add_special_tokens=False)
    special = hf_tokenizer.all_special_ids[0]
    for cut in range(1, len(ids) + 1):
        for num_specials in SPECIAL_RUNS:
            prompt = ids[:cut] + [special] * num_specials
            for cont in (continuation, [prompt[-1], *continuation]):
                _assert_streams_like_full_prompt(
                    monkeypatch, hf_tokenizer, prompt, cont, skip_special_tokens
                )


def test_random_token_ids_stream_same_text_as_full_prompt(monkeypatch, hf_tokenizer):
    """Token-id prompts may hold any token sequence, e.g. byte tokens that do
    not form valid UTF-8, or special tokens in the middle of a character."""
    rng = random.Random(0)
    backend = hf_tokenizer.backend_tokenizer
    pools = [
        range(len(hf_tokenizer)),
        hf_tokenizer.all_special_ids + list(hf_tokenizer.added_tokens_decoder),
        hf_tokenizer.encode(TEXT, add_special_tokens=False),
    ]
    compared = 0
    for _ in range(500):
        prompt = [rng.choice(rng.choice(pools)) for _ in range(rng.randint(1, 30))]
        skip = rng.random() < 0.5
        if backend.decode(prompt, skip_special_tokens=skip).endswith("\ufffd"):
            # Full-prompt seeding streams the whole prompt text here
            # (https://github.com/vllm-project/vllm/issues/48853).
            continue
        continuation = [rng.choice(rng.choice(pools)) for _ in range(6)]
        _assert_streams_like_full_prompt(
            monkeypatch, hf_tokenizer, prompt, continuation, skip
        )
        compared += 1
    assert compared >= 200


@pytest.mark.parametrize("skip_special_tokens", [True, False])
def test_seed_length_is_independent_of_prompt_length(hf_tokenizer, skip_special_tokens):
    """The first output token must not decode the whole prompt."""
    backend = hf_tokenizer.backend_tokenizer
    ids = hf_tokenizer.encode("the is ok. 我很 ", add_special_tokens=False)
    prompt = ids * (100_000 // len(ids))
    for specials in (0, 40):
        seed = get_decode_stream_seed(
            backend,
            prompt + [hf_tokenizer.all_special_ids[0]] * specials,
            skip_special_tokens,
        )
        assert seed is not None
        assert len(seed) <= specials + 2 * INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET


def test_empty_and_missing_prompt(hf_tokenizer):
    backend = hf_tokenizer.backend_tokenizer
    assert get_decode_stream_seed(backend, [], True) == []
    assert get_decode_stream_seed(backend, None, True) is None


def test_prompt_ending_mid_character_keeps_full_prompt_seed():
    """A prompt whose own text ends in U+FFFD cannot be windowed exactly."""
    hf_tokenizer = TokenizersBackend(tokenizer_object=_byte_level(), eos_token=EOS)
    ids = hf_tokenizer.encode("the is ok 𠜎", add_special_tokens=False)
    backend = hf_tokenizer.backend_tokenizer
    assert get_decode_stream_seed(backend, ids[:-1], True) == ids[:-1]
    assert len(get_decode_stream_seed(backend, ids, True)) <= (
        INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET + 3
    )
