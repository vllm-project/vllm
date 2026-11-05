import pytest
from vllm.parser.engine.streaming_parser_engine import StreamingParserEngine
from vllm.parser.engine.types import EventType


class DummyTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab
        self._id_to_text = {v: k for k, v in vocab.items()}

    def get_vocab(self):
        return self._vocab

    def decode(self, ids):
        return "".join(self._id_to_text.get(i, "") for i in ids)

    eos_token_id = None


def test_prelexed_terminal_discards_matching_buffered_subword_prefix():
    """
    Test for Issue #49205:
    Ensures that when subword text tokens form a prefix of a special token terminal,
    and the model then switches to emitting the special-token ID (PreLexedTerminal),
    the stale subword prefix in the lexer buffer is discarded rather than flushed
    as a spurious TEXT_CHUNK.
    """
    from vllm.parser.qwen3 import qwen3_config

    vocab = {
        "<tool_call>": 100,
        "</tool_call>": 101,
        "<": 1,
        "tool": 2,
        "_": 3,
    }
    tokenizer = DummyTokenizer(vocab)
    engine = StreamingParserEngine(qwen3_config(thinking=False), tokenizer)

    events = []
    # 1. Feed subword text tokens that form "<tool_"
    for text, tid in [("<", 1), ("tool", 2), ("_", 3)]:
        events.extend(engine.feed(text, [tid]))

    # 2. Feed the dedicated special token for <tool_call> (ID 100)
    events.extend(engine.feed("<tool_call>", [100]))

    # 3. Verify no leaked bare tag fragment (<tool_) was emitted in text chunks
    text_chunks = [e for e in events if e.type == EventType.TEXT_CHUNK]
    assert not any("<tool_" in chunk.value for chunk in text_chunks), (
        f"Spurious bare tag fragment leaked into TEXT_CHUNK: {text_chunks}"
    )

    # 4. Verify TOOL_CALL_START event was emitted correctly
    tool_starts = [e for e in events if e.type == EventType.TOOL_CALL_START]
    assert len(tool_starts) == 1
