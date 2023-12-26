import os
import time
import traceback
from api.schemas.payload import BatchQueryBody
from api.schemas.response import BatchResponse, Response
from api.server import app
from api.services.detect import Engine
from starlette.requests import Request
import uvicorn
from uvicorn.loops.auto import auto_loop_setup

os.environ.setdefault('ATOM_API_PREFIX', '/omllava')
os.environ.setdefault('DEBUG', 'true')
from linker_atom.api.app import get_app
from linker_atom.api.base import UdfAPIRoute
from linker_atom.config import settings
from linker_atom.lib.exception import VqlError
from linker_atom.lib.log import logger


app = get_app()


@app.on_event("startup")
async def startup_event():
    app.state.detector = Engine()
    model_ids = settings.model_id_list
    for model_id in model_ids:
        try:
            app.state.detector.load_model(model_id)
            logger.info(f"load {model_id} model success.")
            break
        except:
            logger.warning(f"load {model_id} model failed.")
            break

router = UdfAPIRoute()

@router.post("/v1/process/batch_infer", response_model=BatchResponse, name="batch_infer")
async def detect_urls(request: Request, body: BatchQueryBody) -> BatchResponse:
    s_time = time.time()
    engine: Engine = request.app.state.detector
    try:
        res = engine.batch_predict(
            model_id=body.model_id,
            prompts=body.prompts,
            initial_prompt=body.initial_prompt,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
            top_p=body.top_p
        )
    except Exception as error:
        logger.error(traceback.format_exc())
        resp = Response(took=(time.time() - s_time) * 1000, code=500, error=error)
        return resp
    resp = BatchResponse(took=(time.time() - s_time) * 1000, code=200, answer=res)
    return resp

app.include_router(
    router=router,
    prefix=settings.atom_api_prefix
)

def run():
    auto_loop_setup(True)
    uvicorn.run(
        app='wsgi:app',
        host='0.0.0.0',
        port=settings.atom_port,
        workers=settings.atom_workers,
        access_log=False
    )

if __name__ == '__main__':
    run()