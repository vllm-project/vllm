import typer
from typing import Optional, List
from vllm.entrypoints.openai.api_server import run_server
# Assuming `run_server` is a suitable entry function to start your API server

app = typer.Typer()

@app.command()
def serve(model_name: str):
    """
    Serve a specified model using the API server.
    """
    # Logic to start the server with the specified model
    

@app.command()
def query(operation: str, args: Optional[List[str]] = None):
    """
    Query the server with 'chat' or 'complete' operations.
    """
    if operation.lower() == 'chat':
        # Logic to handle chat operation, args can be passed as needed
        pass
    elif operation.lower() == 'complete':
        # Logic to handle complete operation, args can be passed as needed
        pass
    else:
        typer.echo("Invalid operation. Use 'chat' or 'complete'.")

if __name__ == "__main__":
    app()
