"""Launch vLLM OpenAI-compatible API server for BitNet."""
import bitnet_vllm
bitnet_vllm.register()

from vllm.entrypoints.openai.api_server import run_server
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser


def strip_quant_config(config):
    if hasattr(config, 'quantization_config'):
        delattr(config, 'quantization_config')
    return config


if __name__ == '__main__':
    parser = FlexibleArgumentParser(description="BitNet vLLM Server")
    parser = make_arg_parser(parser)
    args = parser.parse_args([
        '--model', 'microsoft/bitnet-b1.58-2B-4T-bf16',
        '--dtype', 'bfloat16',
        '--max-model-len', '2048',
        '--enforce-eager',
        '--host', '0.0.0.0',
        '--port', '8000',
    ])
    # Inject hf_overrides to strip quantization_config
    args.hf_overrides = strip_quant_config

    import uvloop
    uvloop.run(run_server(args))
