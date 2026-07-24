# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sanity tests for the small registry / CLI-arg / config additions made
by the Cohere v2 chat API:

* ``vllm/renderers/registry.py``: new ``"cohere"`` renderer entry.
* ``vllm/tokenizers/registry.py``: new ``"cohere"`` tokenizer entry
  (aliased to the cached HF tokenizer).
* ``vllm/entrypoints/openai/cli_args.py``: new
  ``cohere_is_reasoning_model`` field.
* ``vllm/config/model.py``: ``"cohere"`` added to ``TokenizerMode``
  Literal.
"""

import dataclasses
import typing

import pytest

from vllm.config.model import TokenizerMode
from vllm.entrypoints.openai.cli_args import BaseFrontendArgs, make_arg_parser
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry
from vllm.utils.argparse_utils import FlexibleArgumentParser


class TestRendererRegistry:
    def test_cohere_renderer_registered(self):
        # The registry resolves to the importable ``CohereRenderer`` class.
        cls = RENDERER_REGISTRY.load_renderer_cls("cohere")
        assert cls.__name__ == "CohereRenderer"
        # Sanity: class lives in the cohere renderer module.
        assert cls.__module__ == "vllm.renderers.cohere"


class TestTokenizerRegistry:
    def test_cohere_aliased_to_cached_hf_tokenizer(self):
        # ``cohere`` mode uses the standard HF tokenizer; only the
        # renderer stage is replaced. This test guards against accidental
        # divergence.
        cls = TokenizerRegistry.load_tokenizer_cls("cohere")
        assert cls.__name__ == "CachedHfTokenizer"


class TestTokenizerModeLiteral:
    def test_cohere_is_a_valid_tokenizer_mode(self):
        # The Literal must enumerate ``"cohere"`` so engine arg parsing
        # accepts ``--tokenizer-mode cohere``.
        modes = typing.get_args(TokenizerMode)
        assert "cohere" in modes


# ----------------------------------------------------------------------
# ``--cohere-is-reasoning-model`` CLI flag
# ----------------------------------------------------------------------


class TestCohereCliArg:
    """Verifies the new ``--cohere-is-reasoning-model`` flag end-to-end
    through ``make_arg_parser`` — mirrors the pattern used in
    :mod:`tests.entrypoints.openai.test_cli_args`.
    """

    def test_default_value_on_dataclass_is_true(self):
        fields = {f.name: f for f in dataclasses.fields(BaseFrontendArgs)}
        assert "cohere_is_reasoning_model" in fields
        field = fields["cohere_is_reasoning_model"]
        assert field.default is True
        assert field.type is bool

    @pytest.fixture
    def serve_parser(self) -> FlexibleArgumentParser:
        parser = FlexibleArgumentParser()
        return make_arg_parser(parser)

    def test_default_via_argparse_is_true(self, serve_parser: FlexibleArgumentParser):
        # No flag supplied → dataclass default (True) wins.
        args = serve_parser.parse_args(["--model", "m"])
        assert args.cohere_is_reasoning_model is True

    def test_explicit_false_via_argparse(self, serve_parser: FlexibleArgumentParser):
        # Boolean dataclass fields are wired up as ``--flag value`` /
        # ``--no-flag`` pairs by FlexibleArgumentParser.
        args = serve_parser.parse_args(
            ["--model", "m", "--no-cohere-is-reasoning-model"]
        )
        assert args.cohere_is_reasoning_model is False

    def test_explicit_true_via_argparse(self, serve_parser: FlexibleArgumentParser):
        args = serve_parser.parse_args(["--model", "m", "--cohere-is-reasoning-model"])
        assert args.cohere_is_reasoning_model is True
