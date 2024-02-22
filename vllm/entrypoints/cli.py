from requests import request
import typer
import json
import os
from fastapi import FastAPI, Request, Response, HTTPStatus
from fastapi.middleware.cors import CORSMiddleware
from aioprometheus import MetricsMiddleware
from aioprometheus.asgi.starlette import metrics
from fastapi.responses import JSONResponse, StreamingResponse
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.protocol import CompletionRequest, ChatCompletionRequest, ErrorResponse
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.logger import init_logger
from vllm.engine.metrics import add_global_metrics_labels
import importlib
import inspect

app = typer.Typer()
HOST = "127.0.0.1"
PORT = 8000
logger = init_logger(__name__)

@app.command()
def serve(
    host: str = typer.Option(None, "--host", help="Host name."),
    port: int = typer.Option(8000, "--port", help="Port number."),
    allow_credentials: bool = typer.Option(False, "--allow-credentials", help="Allow credentials."),
    allowed_origins: str = typer.Option("['*']", "--allowed-origins", help="Allowed origins in JSON format."),
    allowed_methods: str = typer.Option("['*']", "--allowed-methods", help="Allowed methods in JSON format."),
    allowed_headers: str = typer.Option("['*']", "--allowed-headers", help="Allowed headers in JSON format."),
    api_key: str = typer.Option(None, "--api-key", help="API key for server authentication."),
    model_name: str = typer.Option(None, "--served-model-name", help="The model name used in the API."),
    chat_template: str = typer.Option(None, "--chat-template", help="Path or single-line form of the chat template."),
    response_role: str = typer.Option("assistant", "--response-role", help="Role name to return if `request.add_generation_prompt=true`."),
    ssl_keyfile: str = typer.Option(None, "--ssl-keyfile", help="SSL key file path."),
    ssl_certfile: str = typer.Option(None, "--ssl-certfile", help="SSL certificate file path."),
    root_path: str = typer.Option(None, "--root-path", help="FastAPI root_path for proxy routing."),
    middleware: List[str] = typer.Option(None, "--middleware", help="Additional ASGI middleware import paths.", callback=lambda x: x.split(',') if x else []),
):
    """
    Serve a specified model using the API server.
    """
    args = {
        "allowed_origins": json.loads(allowed_origins),
        "allow_credentials": allow_credentials,
        "allowed_methods": json.loads(allowed_methods),
        "allowed_headers": json.loads(allowed_headers),
        "api_key": api_key,
        "model_name": model_name,
        "chat_template": chat_template,
        "response_role": response_role,
        "ssl_keyfile": ssl_keyfile,
        "ssl_certfile": ssl_certfile,
        "root_path": root_path,
    }
    global HOST, PORT
    HOST, PORT = host, port
    api_server.start_server(args)

@app.command()
def query(operation: str, data: str):
    """
    Query the server with 'chat' or 'complete' operations.
    :param operation: 'chat' or 'complete'
    :param data: JSON string with request data
    """
    # Construct the API URL using the global HOST and PORT
    api_url = f"http://{HOST}:{PORT}"

    try:
        request_data = json.loads(data)
    except json.JSONDecodeError:
        typer.echo("Invalid JSON data.")
        raise typer.Exit(code=1)

    # Determine the endpoint based on the operation
    endpoint = "/v1/chat/completions" if operation.lower() == 'chat' else "/v1/completions"
    response = request.post(f"{api_url}{endpoint}", json=request_data)

    # Handle the response
    if response.status_code == 200:
        typer.echo(json.dumps(response.json(), indent=2))
    else:
        typer.echo(f"Error: {response.status_code} - {response.text}")


if __name__ == "__main__":
    app()
