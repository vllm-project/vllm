import pytest

from vllm.utils.argparse_utils import FlexibleArgumentParser


def _serve_parser() -> FlexibleArgumentParser:
    parser = FlexibleArgumentParser(description="test")
    parser.add_argument("model_tag", nargs="?")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def test_serve_model_flag_missing_value_errors():
    parser = _serve_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--model"])


def test_serve_model_flag_followed_by_option_errors():
    parser = _serve_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--model", "--port", "8001"])


def test_serve_model_flag_rewrites_to_positional():
    parser = _serve_parser()
    args = parser.parse_args(["serve", "--model", "facebook/opt-125m", "--port", "8001"])
    assert args.model_tag == "facebook/opt-125m"
    assert args.port == 8001
