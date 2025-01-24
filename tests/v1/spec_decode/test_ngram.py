import pytest

from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.utils import ConstantList


@pytest.fixture
def proposer():
    return NgramProposer()


def test_kmp_lps_array(proposer):
    assert proposer._kmp_lps_array([]) == []
    assert proposer._kmp_lps_array([1]) == [0]
    assert proposer._kmp_lps_array([1, 1, 1]) == [0, 1, 2]
    assert proposer._kmp_lps_array([1, 2, 3, 4]) == [0, 0, 0, 0]
    assert proposer._kmp_lps_array([1, 2, 1, 2, 3]) == [0, 0, 1, 2, 0]


def test_find_subarray_kmp(proposer):
    X = ConstantList([1, 2, 3, 4, 1, 2, 3, 5, 6])

    assert proposer._find_subarray_kmp(X, [], 2) == [1, 2]
    assert proposer._find_subarray_kmp(X, [7, 8], 1) is None
    assert proposer._find_subarray_kmp(X, [1, 2, 3], 2) == [1, 2, 3, 4, 1]
    assert proposer._find_subarray_kmp(X, [5, 6], 1) == [5, 6]


def test_propose(proposer):
    context = ConstantList([1, 2, 3, 4, 1, 2, 3, 5, 6])
    assert proposer.propose(context, n=3, k=2) is None
    assert proposer.propose(context, n=2, k=1) is None
    assert proposer.propose(context, n=4, k=2) is None

    context = ConstantList([1, 2, 3, 4, 1, 2])
    assert proposer.propose(context, n=2, k=2) == [3, 4]
    assert proposer.propose(context, n=2, k=1) == [3]
    assert proposer.propose(context, n=3, k=2) is None
