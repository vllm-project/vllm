import asyncio
import itertools

import aiohttp
from aiohttp import web


class RoundRobinProxy:

    def __init__(self, target_ports):
        self.target_ports = target_ports
        self.port_cycle = itertools.cycle(self.target_ports)

    async def handle_request(self, request):
        target_port = next(self.port_cycle)
        target_url = f"http://localhost:{target_port}{request.path_qs}"

        async with aiohttp.ClientSession() as session:
            try:
                # Forward the request
                async with session.request(
                        method=request.method,
                        url=target_url,
                        headers=request.headers,
                        data=request.content,
                ) as response:
                    # Start sending the response
                    resp = web.StreamResponse(status=response.status,
                                              headers=response.headers)
                    await resp.prepare(request)

                    # Stream the response content
                    async for chunk in response.content.iter_any():
                        await resp.write(chunk)

                    await resp.write_eof()
                    return resp

            except Exception as e:
                return web.Response(text=f"Error: {str(e)}", status=500)


async def main():
    proxy = RoundRobinProxy([8100, 8200])
    app = web.Application()
    app.router.add_route('*', '/{path:.*}', proxy.handle_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8000)
    await site.start()

    print("Proxy server started on http://localhost:8000")

    # Keep the server running
    await asyncio.Event().wait()


if __name__ == '__main__':
    asyncio.run(main())
