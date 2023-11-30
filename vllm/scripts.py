# The CLI entrypoint to vLLM.
import argparse
import vllm.entrypoints.openai.api_server as api_server


def main():
    parser = argparse.ArgumentParser(description='vLLM CLI')
    subparsers = parser.add_subparsers()

    # Create the parser for the "serve" command
    serve_parser = subparsers.add_parser('serve', help='Start the vLLM API server')
    api_server.make_parser(serve_parser)
    serve_parser.set_defaults(func=lambda args: api_server.run_server(args))

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
