import argparse
import json
from typing import AsyncGenerator
from datetime import datetime
import subprocess
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn



TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()
engine = None


@app.post("/switch")
async def switch(request: Request) -> Response:
    """Switch the model.
    """
    request_dict = await request.json()
    current_type = request_dict.pop("modeltype")
    if current_type == args.modeltype:
         return {"model":current_type, "status":"finished"}
    args.modeltype = None
    command1 = "pkill -f 'python3 -m vllm.entrypoints.api_server_multi'"
    date_str = datetime.now().strftime("%Y%m%d")
    command2 = f"python3 -m vllm.entrypoints.api_server_multi --modeltype '{current_type}' --host 0.0.0.0 --port 12301 --trust-remote-code | tee './logs/{date_str}_{current_type}_output.log'"
    try:
        result = subprocess.run(command1, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("finish killed")
    except:
        pass
    try:
        process = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # 循环读取输出
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            if "Uvicorn running on http://140.210.91.219:12301" in line or "Uvicorn running on http://0.0.0.0:12301" in line:
                print("finish build")
                args.modeltype = current_type
                ret = {"model":args.modeltype, "status":"finished"}
                return JSONResponse(ret)
    except:
            ret = {"model":current_type, "status":"failed"}
            return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ssl-keyfile", type=str, default=None)
    parser.add_argument("--ssl-certfile", type=str, default=None)
    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="FastAPI root_path when app is behind a path based routing proxy")
    args = parser.parse_args()
    app.root_path = args.root_path
    args.modeltype = None
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                ssl_keyfile=args.ssl_keyfile,
                ssl_certfile=args.ssl_certfile)
