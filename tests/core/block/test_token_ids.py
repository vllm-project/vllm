import pytest

from vllm.core.block.token_ids import TokenIds, TokenRangeAnnotation
from vllm.inputs.data import LLMInputs
from vllm.sequence import Sequence


@pytest.mark.parametrize(
    "value",
    [
        # Must be contained within the token IDs.
        [TokenRangeAnnotation(0, 0, -1, 2)],
        [TokenRangeAnnotation(0, 0, 0, 5)],
        [TokenRangeAnnotation(0, 0, 4, 1)],

        # Must not overlap.
        [
            TokenRangeAnnotation(000, 0, 0, 1),
            TokenRangeAnnotation(111, 0, 0, 1)
        ],
        [
            TokenRangeAnnotation(000, 0, 0, 2),
            TokenRangeAnnotation(111, 0, 1, 2)
        ],
        [
            TokenRangeAnnotation(000, 0, 2, 1),
            TokenRangeAnnotation(111, 0, 0, 4)
        ],

        # Must be sorted.
        [
            TokenRangeAnnotation(000, 0, 2, 1),
            TokenRangeAnnotation(111, 0, 0, 1)
        ],
        [
            TokenRangeAnnotation(000, 0, 0, 1),
            TokenRangeAnnotation(111, 0, 3, 1),
            TokenRangeAnnotation(222, 0, 2, 1)
        ],
    ])
def test_invalid_annotations_should_raise(value):
    with pytest.raises(ValueError):
        TokenIds(range(4), value)


@pytest.mark.parametrize("value", [
    TokenIds(()),
    TokenIds((1, 2, 3)),
    TokenIds((1, 2, 3), [TokenRangeAnnotation(0, 0, 0, 1)])
])
def test_token_ids_add_unit(value):
    assert value + TokenIds() == value
    assert TokenIds() + value == value


def test_token_ids_add_without_annotations():
    a = TokenIds((1, 2, 3))
    b = TokenIds((4, 5, 6))
    assert a + b == TokenIds((1, 2, 3, 4, 5, 6))


def test_token_ids_add_with_annotations():
    a = TokenIds((1, 2, 3))
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(0, 0, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6),
                             [TokenRangeAnnotation(0, 0, 3, 1)])
    assert b + a == TokenIds((4, 5, 6, 1, 2, 3),
                             [TokenRangeAnnotation(0, 0, 0, 1)])


def test_token_ids_add_can_coalesce():
    a = TokenIds((1, 2, 3), [TokenRangeAnnotation(111, 0, 1, 2)])
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(111, 2, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6),
                             [TokenRangeAnnotation(111, 0, 1, 3)])


def test_token_ids_add_cannot_coalesce_different_offsets():
    a = TokenIds((1, 2, 3), [TokenRangeAnnotation(111, 0, 1, 2)])
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(111, 4, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6), [
        TokenRangeAnnotation(111, 0, 1, 2),
        TokenRangeAnnotation(111, 4, 3, 1)
    ])


def test_token_ids_add_cannot_coalesce_different_hash():
    a = TokenIds((1, 2, 3), [TokenRangeAnnotation(111, 0, 1, 2)])
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(222, 2, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6), [
        TokenRangeAnnotation(111, 0, 1, 2),
        TokenRangeAnnotation(222, 2, 3, 1)
    ])


def test_annotation_adjustment():
    r = TokenRangeAnnotation(111, 0, 2, 3)
    # Overlapping windows
    assert r.adjusted(0, 1) is None
    assert r.adjusted(0, 2) is None
    assert r.adjusted(0, 3) == TokenRangeAnnotation(111, 0, 2, 1)
    assert r.adjusted(0, 4) == TokenRangeAnnotation(111, 0, 2, 2)
    assert r.adjusted(1, 5) == TokenRangeAnnotation(111, 0, 1, 3)
    assert r.adjusted(2, 6) == TokenRangeAnnotation(111, 0, 0, 3)
    assert r.adjusted(3, 7) == TokenRangeAnnotation(111, 1, 0, 2)
    assert r.adjusted(4, 8) == TokenRangeAnnotation(111, 2, 0, 1)
    assert r.adjusted(5, 9) is None

    # Interior windows
    assert r.adjusted(2, 3) == TokenRangeAnnotation(111, 0, 0, 1)
    assert r.adjusted(2, 4) == TokenRangeAnnotation(111, 0, 0, 2)
    assert r.adjusted(2, 5) == TokenRangeAnnotation(111, 0, 0, 3)
    assert r.adjusted(3, 4) == TokenRangeAnnotation(111, 1, 0, 1)
    assert r.adjusted(3, 5) == TokenRangeAnnotation(111, 1, 0, 2)
    assert r.adjusted(4, 5) == TokenRangeAnnotation(111, 2, 0, 1)
    assert r.adjusted(3, 3) is None


def test_token_id_chunks():
    token_ids = TokenIds(range(8), [
        TokenRangeAnnotation(111, 0, 2, 3),
        TokenRangeAnnotation(222, 0, 6, 1),
        TokenRangeAnnotation(333, 0, 7, 1)
    ])
    single_chunks = [
        TokenIds([0]),
        TokenIds([1]),
        TokenIds([2], [TokenRangeAnnotation(111, 0, 0, 1)]),
        TokenIds([3], [TokenRangeAnnotation(111, 1, 0, 1)]),
        TokenIds([4], [TokenRangeAnnotation(111, 2, 0, 1)]),
        TokenIds([5]),
        TokenIds([6], [TokenRangeAnnotation(222, 0, 0, 1)]),
        TokenIds([7], [TokenRangeAnnotation(333, 0, 0, 1)]),
    ]

    # Without overriding initial chunk size
    for chunk_size in range(1, len(single_chunks) + 1):
        chunks = list(token_ids.chunks(chunk_size))
        expected_chunks = [
            sum(single_chunks[i:i + chunk_size], start=TokenIds())
            for i in range(0, len(single_chunks), chunk_size)
        ]
        assert chunks == expected_chunks

    # With overriding first chunk size
    for first_chunk_size in range(1, len(single_chunks) + 1):
        first_chunk = sum(single_chunks[0:first_chunk_size], start=TokenIds())
        for chunk_size in range(1, len(single_chunks) + 1):
            chunks = list(
                token_ids.chunks(chunk_size,
                                 first_chunk_size=first_chunk_size))
            expected_chunks = [first_chunk] + [
                sum(single_chunks[i:i + chunk_size], start=TokenIds()) for i in
                range(first_chunk_size, len(single_chunks), chunk_size)
            ]
            assert chunks == expected_chunks

    # Slicing should be equivalent
    for i in range(len(single_chunks)):
        assert token_ids[i:] == sum(single_chunks[i:], start=TokenIds())


def test_token_ids_from_sequence():
    token_ids_list = list(range(4))
    sequence = Sequence(
        seq_id=0,
        inputs=LLMInputs(prompt_token_ids=token_ids_list,
                         token_annotations=[TokenRangeAnnotation(0, 0, 2, 2)]),
        block_size=16)
    whole_token_ids = TokenIds(token_ids_list,
                               [TokenRangeAnnotation(0, 0, 2, 2)])

    for i in range(len(token_ids_list)):
        assert TokenIds.from_sequence(sequence,
                                      offset=i) == whole_token_ids[i:]
