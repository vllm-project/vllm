from vllm.prefix import PrefixPool

import pytest


@pytest.fixture
def no_max_capacity_prefix_pool() -> PrefixPool:
    return PrefixPool(block_size=32, max_capacity_in_blocks=float('inf'))


def test_prefix_length_behaviours(no_max_capacity_prefix_pool: PrefixPool):
    """
    This test checks that prefixes of length less than pool.block_size are not created and are not added to the pool.
    It also checks that prefixes of length equal to or greater to pool.block_size are created and added to the pool.
    """
    prefix_1 = no_max_capacity_prefix_pool.add_or_get_prefix(
        list(range(no_max_capacity_prefix_pool.block_size - 1)))
    prefix_2 = no_max_capacity_prefix_pool.add_or_get_prefix(
        list(range(no_max_capacity_prefix_pool.block_size)))
    prefix_3 = no_max_capacity_prefix_pool.add_or_get_prefix(
        list(range(no_max_capacity_prefix_pool.block_size * 2)))
    assert prefix_1 is None
    assert prefix_2 is not None
    assert prefix_3 is not None
    assert len(no_max_capacity_prefix_pool) == 2


def test_same_prefix_added_twice(no_max_capacity_prefix_pool: PrefixPool):
    """
    Tests that when a prefix is added more than once to the pool, all subsequent additions
    return the same prefix object that was created the first time.
    """
    prefix_1 = no_max_capacity_prefix_pool.add_or_get_prefix(
        list(range(no_max_capacity_prefix_pool.block_size)))
    prefix_2 = no_max_capacity_prefix_pool.add_or_get_prefix(
        list(range(no_max_capacity_prefix_pool.block_size)))
    assert prefix_1 is prefix_2
    assert len(no_max_capacity_prefix_pool) == 1


def test_prefix_pool_max_capacity():
    """
    Tests that the pool is evicting prefixes when it reaches max capacity.
    """
    max_capacity_in_blocks = 2
    max_capacity_prefix_pool = PrefixPool(
        block_size=32, max_capacity_in_blocks=max_capacity_in_blocks)

    # Tests that on the third insertion, new object is created because capacity limits reached,
    # but that the newly created object is equal to the old object
    prefix_1 = max_capacity_prefix_pool.add_or_get_prefix(
        list(range(max_capacity_prefix_pool.block_size)))
    _ = max_capacity_prefix_pool.add_or_get_prefix(
        list(range(max_capacity_prefix_pool.block_size * 2)))
    prefix_3 = max_capacity_prefix_pool.add_or_get_prefix(
        list(range(max_capacity_prefix_pool.block_size)))
    assert prefix_1 is not prefix_3
    assert prefix_1 == prefix_3

    assert len(max_capacity_prefix_pool) == 1
    assert max_capacity_prefix_pool.current_block_usage == 1


def test_current_block_usage():
    """
    Tests that the current_block_usage property remains the same thorough the
    lifetime of the pool when adding prefixes that are always the same length equal
    to the max capacity.
    """
    max_capacity_in_blocks = 2
    max_capacity_prefix_pool = PrefixPool(
        block_size=32, max_capacity_in_blocks=max_capacity_in_blocks)

    for _ in range(10):
        _ = max_capacity_prefix_pool.add_or_get_prefix(
            list(
                range(max_capacity_prefix_pool.block_size *
                      max_capacity_in_blocks)))
        assert len(max_capacity_prefix_pool) == 1
        assert max_capacity_prefix_pool.current_block_usage == max_capacity_in_blocks


def test_prefix_truncation_1():
    """
    Tests that prefix is truncated if it exceeds the max capacity.
    """
    prefix_pool = PrefixPool(block_size=1, max_capacity_in_blocks=2)
    prefix = prefix_pool.add_or_get_prefix([1, 2, 3, 4])
    assert prefix.token_ids == (1, 2)


def test_prefix_truncation_2():
    """
    Testing truncation on non-block boundary
    """
    prefix_pool = PrefixPool(block_size=2, max_capacity_in_blocks=3)
    prefix = prefix_pool.add_or_get_prefix([1, 2, 3, 4, 5])
    assert prefix.token_ids == (1, 2, 3, 4)


def test_prefix_truncation_3():
    """
    Tests truncation because of both max capacity exceeded and no block boundary.
    """
    prefix_pool = PrefixPool(block_size=2, max_capacity_in_blocks=2)
    prefix = prefix_pool.add_or_get_prefix([1, 2, 3, 4, 5])
    assert prefix.token_ids == (1, 2, 3, 4)


def test_none_prefix_returned_1():
    """
    Tests that when the max capacity is zero, no prefix is created and None is returned.
    """
    prefix_pool = PrefixPool(block_size=32, max_capacity_in_blocks=0)
    prefix = prefix_pool.add_or_get_prefix(list(range(prefix_pool.block_size)))
    assert prefix is None
    assert len(prefix_pool) == 0


def test_none_prefix_returned_2():
    """
    Tests that when prefix length is less than block size, a None prefix is returned.
    """
    prefix_pool = PrefixPool(block_size=32, max_capacity_in_blocks=2)
    prefix = prefix_pool.add_or_get_prefix(
        list(range(prefix_pool.block_size - 1)))
    assert prefix is None
    assert len(prefix_pool) == 0


def test_assertion_raised_with_invalid_max_capacity():
    with pytest.raises(AssertionError):
        _ = PrefixPool(32, max_capacity_in_blocks=-1)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
