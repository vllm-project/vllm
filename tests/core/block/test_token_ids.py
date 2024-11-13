import pytest

from vllm.core.block.token_ids import TokenIds, TokenRangeAnnotation


@pytest.mark.parametrize(
    "value",
    [
        # Must be contained within the token IDs.
        [TokenRangeAnnotation(0, 0, -1, 2)],
        [TokenRangeAnnotation(0, 0, 0, 5)],
        [TokenRangeAnnotation(0, 0, 4, 5)],

        # Must not overlap.
        [
            TokenRangeAnnotation(000, 0, 0, 1),
            TokenRangeAnnotation(111, 0, 0, 1)
        ],
        [
            TokenRangeAnnotation(000, 0, 0, 2),
            TokenRangeAnnotation(111, 0, 1, 3)
        ],
        [
            TokenRangeAnnotation(000, 0, 2, 3),
            TokenRangeAnnotation(111, 0, 0, 4)
        ],

        # Must be sorted.
        [
            TokenRangeAnnotation(000, 0, 2, 3),
            TokenRangeAnnotation(111, 0, 0, 1)
        ],
        [
            TokenRangeAnnotation(000, 0, 0, 1),
            TokenRangeAnnotation(111, 0, 3, 4),
            TokenRangeAnnotation(222, 0, 2, 3)
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
                             [TokenRangeAnnotation(0, 0, 3, 4)])
    assert b + a == TokenIds((4, 5, 6, 1, 2, 3),
                             [TokenRangeAnnotation(0, 0, 0, 1)])


def test_token_ids_add_can_coalesce():
    a = TokenIds((1, 2, 3), [TokenRangeAnnotation(111, 0, 1, 3)])
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(111, 2, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6),
                             [TokenRangeAnnotation(111, 0, 1, 4)])


def test_token_ids_add_cannot_coalesce_different_offsets():
    a = TokenIds((1, 2, 3), [TokenRangeAnnotation(111, 0, 1, 3)])
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(111, 4, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6), [
        TokenRangeAnnotation(111, 0, 1, 3),
        TokenRangeAnnotation(111, 4, 3, 4)
    ])


def test_token_ids_add_cannot_coalesce_different_hash():
    a = TokenIds((1, 2, 3), [TokenRangeAnnotation(111, 0, 1, 3)])
    b = TokenIds((4, 5, 6), [TokenRangeAnnotation(222, 2, 0, 1)])

    assert a + b == TokenIds((1, 2, 3, 4, 5, 6), [
        TokenRangeAnnotation(111, 0, 1, 3),
        TokenRangeAnnotation(222, 2, 3, 4)
    ])


def test_annotation_clipping():
    r = TokenRangeAnnotation(111, 0, 2, 5)
    # Overlapping windows
    assert r.clipped_to_slice(0, 1) is None
    assert r.clipped_to_slice(0, 2) is None
    assert r.clipped_to_slice(0, 3) == TokenRangeAnnotation(111, 0, 2, 3)
    assert r.clipped_to_slice(0, 4) == TokenRangeAnnotation(111, 0, 2, 4)
    assert r.clipped_to_slice(1, 5) == TokenRangeAnnotation(111, 0, 1, 4)
    assert r.clipped_to_slice(2, 6) == TokenRangeAnnotation(111, 0, 0, 3)
    assert r.clipped_to_slice(3, 7) == TokenRangeAnnotation(111, 1, 0, 2)
    assert r.clipped_to_slice(4, 8) == TokenRangeAnnotation(111, 2, 0, 1)
    assert r.clipped_to_slice(5, 9) is None

    # Interior windows
    assert r.clipped_to_slice(2, 3) == TokenRangeAnnotation(111, 0, 0, 1)
    assert r.clipped_to_slice(2, 4) == TokenRangeAnnotation(111, 0, 0, 2)
    assert r.clipped_to_slice(2, 5) == TokenRangeAnnotation(111, 0, 0, 3)
    assert r.clipped_to_slice(3, 4) == TokenRangeAnnotation(111, 1, 0, 1)
    assert r.clipped_to_slice(3, 5) == TokenRangeAnnotation(111, 1, 0, 2)
    assert r.clipped_to_slice(4, 5) == TokenRangeAnnotation(111, 2, 0, 1)
    assert r.clipped_to_slice(3, 3) is None


def test_token_id_chunks():
    token_ids = TokenIds(range(8), [
        TokenRangeAnnotation(111, 0, 2, 5),
        TokenRangeAnnotation(222, 0, 6, 7),
        TokenRangeAnnotation(333, 0, 7, 8)
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
        chunks = list(token_ids.to_chunks(chunk_size))
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
                token_ids.to_chunks(chunk_size,
                                    first_chunk_size=first_chunk_size))
            expected_chunks = [first_chunk] + [
                sum(single_chunks[i:i + chunk_size], start=TokenIds()) for i in
                range(first_chunk_size, len(single_chunks), chunk_size)
            ]
            assert chunks == expected_chunks

    # Slicing should be equivalent
    for i in range(len(single_chunks)):
        assert token_ids[i:] == sum(single_chunks[i:], start=TokenIds())
