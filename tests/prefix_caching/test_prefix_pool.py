from vllm.prefix import PrefixPool

import pytest


@pytest.fixture
def no_max_capacity_prefix_pool() -> PrefixPool:
    return PrefixPool(block_size=32)


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
    max_capacity = 1
    max_capacity_prefix_pool = PrefixPool(block_size=32,
                                          max_capacity=max_capacity)

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

    # Tests that the max capacity remains the same
    for i in range(10):
        _ = max_capacity_prefix_pool.add_or_get_prefix(
            list(range(max_capacity_prefix_pool.block_size + i)))
        assert len(max_capacity_prefix_pool) == max_capacity


def test_assertion_raised_with_invalid_max_capacity():
    with pytest.raises(AssertionError):
        _ = PrefixPool(32, max_capacity=-1)

    with pytest.raises(AssertionError):
        _ = PrefixPool(32, max_capacity=0)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
