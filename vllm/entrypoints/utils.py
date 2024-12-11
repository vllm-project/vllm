import anyio
from starlette.requests import Request
from starlette.types import Message


async def is_disconnected_patch(request: Request) -> bool:
    """This is a patch for starlette's Request.is_disconnected(), which,
    if a BaseHTTPMiddleware is added to the server, returns False even if
    the client disconnects.
    See discussion here: https://github.com/encode/starlette/discussions/2094
    This workaround is based on the code in comment
    https://github.com/encode/starlette/discussions/2094#discussioncomment-9084737
    It can be removed if and when the starlette issue is resolved.
    """
    assert hasattr(request,
                   '_is_disconnected'), "private API in starlette changed"
    assert isinstance(request._is_disconnected,
                      bool), "private API in starlette changed"

    if request._is_disconnected:
        return True

    message: Message = {}

    # this timeout may need to be tweaked.
    # Ideally request.receive() is non-blocking, in which case the timeout
    # doesn't matter. But the ASGI spec seems to imply it should be a blocking
    # callable, as it doesn't specify what should be returned in case of
    # no new messages. In this situation you can return an empty dict,
    # but given that it's against the spec it seems inadvisable.
    with anyio.move_on_after(0.01):
        message = await request.receive()

    if message.get("type") == "http.disconnect":
        request._is_disconnected = True

    return request._is_disconnected
