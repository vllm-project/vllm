import argparse
import logging

from disagg_proxy import ProxyServer

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Todo: allow more config
    parser = argparse.ArgumentParser("vLLM disaggregated proxy server.")
    parser.add_argument("--model",
                        "-m",
                        type=str,
                        required=True,
                        help="Model name")

    parser.add_argument(
        "--prefill",
        "-p",
        type=str,
        nargs="+",
        help="List of prefill node URLs (host:port)",
    )

    parser.add_argument(
        "--decode",
        "-d",
        type=str,
        nargs="+",
        help="List of decode node URLs (host:port)",
    )

    parser.add_argument(
        "--scheduling-policy",
        "--scheduling",
        "-S",
        type=str,
        choices=["round_robin"],
        default="round_robin",
        help="Proxy scheduling strategy",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number",
    )
    args = parser.parse_args()
    proxy_server = ProxyServer(args=args)
    proxy_server.run_server()