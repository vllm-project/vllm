
import pytest
from typing import Generator, Optional

from vllm import LLM
from tests.anyscale.utils import SecretManager, cleanup


@pytest.fixture(scope="session", autouse=True)
def load_secrets():
    secrets = SecretManager()
    secrets.override_secret("HUGGING_FACE_HUB_TOKEN")


# pylint: disable=redefined-outer-name
@pytest.fixture(name="spec_decode_llm")
def create_spec_decode_llm(
        spec_decode_llm_generator: Generator[LLM, None, None]) -> LLM:
    for spec_decode_llm in spec_decode_llm_generator:
        yield spec_decode_llm
        del spec_decode_llm


@pytest.fixture
def spec_decode_llm_generator(
        target_model: str, draft_model: str, num_speculative_tokens: str,
        tensor_parallel_size: int, with_cuda_graph: bool,
        speculative_model_uses_tp_1: bool,
        disable_shared_memory: bool) -> Generator[LLM, None, None]:
    return create_spec_decode_llm_generator(target_model, draft_model,
                                            num_speculative_tokens,
                                            tensor_parallel_size,
                                            with_cuda_graph,
                                            speculative_model_uses_tp_1,
                                            disable_shared_memory)


@pytest.fixture
def max_model_len_spec_decode_generator(
        target_model: str, draft_model: str, num_speculative_tokens: str,
        tensor_parallel_size: int, with_cuda_graph: bool,
        disable_shared_memory: bool,
        max_model_len: int) -> Generator[LLM, None, None]:
    return create_spec_decode_llm_generator(
        target_model,
        draft_model,
        num_speculative_tokens,
        tensor_parallel_size,
        with_cuda_graph,
        speculative_model_uses_tp_1=False,
        disable_shared_memory=disable_shared_memory,
        max_model_len=max_model_len)


def create_spec_decode_llm_generator(
        target_model: str,
        draft_model: str,
        num_speculative_tokens: str,
        tensor_parallel_size: int,
        with_cuda_graph: bool,
        speculative_model_uses_tp_1: bool,
        disable_shared_memory: bool,
        max_model_len: Optional[int] = None) -> Generator[LLM, None, None]:

    def generator():
        addl_kwargs = {}
        if max_model_len is not None:
            addl_kwargs["max_model_len"] = max_model_len

        spec_decode_llm = LLM(
            model=target_model,
            speculative_model=draft_model,
            num_speculative_tokens=num_speculative_tokens,
            tensor_parallel_size=tensor_parallel_size,
            enable_cuda_graph=with_cuda_graph,
            disable_shared_memory=disable_shared_memory,
            speculative_model_uses_tp_1=speculative_model_uses_tp_1,
            worker_use_ray=True,
            **addl_kwargs,
        )

        yield spec_decode_llm

        del spec_decode_llm
        cleanup()

    return generator()


@pytest.fixture
def non_spec_decode_llm_generator(
        target_model: str, tensor_parallel_size: int,
        with_cuda_graph: bool) -> Generator[LLM, None, None]:

    return create_non_spec_decode_llm_generator(target_model,
                                                tensor_parallel_size,
                                                with_cuda_graph)


@pytest.fixture
def max_model_len_llm_generator(
        target_model: str, tensor_parallel_size: int, with_cuda_graph: bool,
        max_model_len: int) -> Generator[LLM, None, None]:
    return create_non_spec_decode_llm_generator(target_model,
                                                tensor_parallel_size,
                                                with_cuda_graph, max_model_len)


def create_non_spec_decode_llm_generator(
        target_model: str,
        tensor_parallel_size: int,
        with_cuda_graph: bool,
        max_model_len: Optional[int] = None) -> Generator[LLM, None, None]:

    def generator():
        addl_kwargs = {}
        if max_model_len is not None:
            addl_kwargs["max_model_len"] = max_model_len

        llm = LLM(
            model=target_model,
            tensor_parallel_size=tensor_parallel_size,
            enable_cuda_graph=with_cuda_graph,
            worker_use_ray=True,
            **addl_kwargs,
        )

        yield llm

        del llm
        cleanup()

    return generator()
