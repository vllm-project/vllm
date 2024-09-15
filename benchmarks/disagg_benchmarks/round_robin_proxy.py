import asyncio
import aiohttp
from aiohttp import web
import itertools

class AsyncRoundRobinProxy:
    def __init__(self, backend_ports):
        self.backend_ports = itertools.cycle(backend_ports)
        self.session = None

    async def start(self):
        self.session = aiohttp.ClientSession()

    async def stop(self):
        if self.session:
            await self.session.close()

    async def handle_request(self, request):
        backend_port = next(self.backend_ports)
        print("forwarding to port", backend_port)
        backend_url = f"http://localhost:{backend_port}{request.path_qs}"

        try:
            async with self.session.request(
                method=request.method,
                url=backend_url,
                headers=request.headers,
                data=await request.read()
            ) as backend_response:
                response = web.StreamResponse(
                    status=backend_response.status,
                    headers=backend_response.headers
                )
                await response.prepare(request)

                async for chunk in backend_response.content.iter_any():
                    await response.write(chunk)

                await response.write_eof()
                return response

        except aiohttp.ClientError as e:
            return web.Response(text=f"Backend error: {str(e)}", status=502)

async def run_backend(port):
    async def handle(request):
        if request.path == '/stream':
            response = web.StreamResponse(
                status=200,
                headers={'Content-Type': 'text/plain'}
            )
            await response.prepare(request)
            for i in range(10):
                await response.write(f"Chunk {i}\n".encode())
                await asyncio.sleep(0.5)  # Simulate delay between chunks
            return response
        else:
            return web.Response(text=f"Response from backend on port {port}")

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', port)
    await site.start()
    print(f"Backend running on http://localhost:{port}")

async def main():
    proxy = AsyncRoundRobinProxy([8100, 8200])
    await proxy.start()

    app = web.Application()
    app.router.add_route('*', '/{tail:.*}', proxy.handle_request)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8000)

    await asyncio.gather(
        site.start(),
        run_backend(8100),
        run_backend(8200)
    )

    print("Proxy running on http://localhost:8000")

    try:
        await asyncio.Future()  # Run forever
    finally:
        await proxy.stop()
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())