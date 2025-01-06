import time
import zmq
import zmq.asyncio
import random
import asyncio


async def worker_routine(worker_url: str,
                   context: zmq.asyncio.Context = None, i: int = 0):
    """Worker routine"""
    # Socket to talk to dispatcher
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.IDENTITY, f"worker-{i}-{time.time()}".encode())
    socket.connect(worker_url)
    print(f"worker-{i} {worker_url} started")
    while True:
        identity, url, headers, string  = await socket.recv_multipart()
        print(f"worker-{i} Received request identity: [{identity} ]")
        print(f"worker-{i} Received request url: [{url} ]")
        print(f"worker-{i} Received request headers: [{headers} ]")
        print(f"worker-{i} Received request string: [{string} ]")
        streamreply = ['{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}',
'{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}',
'{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1694268190,"model":"gpt-4o-mini", "system_fingerprint": "fp_44709d6fcb", "choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}'
]
        for j in range(len(streamreply)):
            # Do some 'work'
            # time.sleep(random.randint(1, 5))
            await asyncio.sleep(random.randint(1, 5))
            # Send reply back to client
            reply = f"worker-{i} reply part-{j} {string} \n {streamreply[j]}"
            # reply = streamreply[j]
            print(f"worker-{i} Sent reply: [{identity} {reply} ]")
            await socket.send_multipart([identity, reply.encode()])

async def main():
    """Server routine"""

    url_worker = "inproc://workers"
    url_client = "tcp://localhost:5555"

    # Prepare our context and sockets
    context = zmq.asyncio.Context()

    # Socket to talk to clients
    clients = context.socket(zmq.ROUTER)
    clients.bind(url_client)
    print("Server ROUTER started at", url_client)
    # Socket to talk to workers
    workers = context.socket(zmq.DEALER)
    workers.bind(url_worker)
    print("Worker DEALER started at", url_worker)

    tasks = [asyncio.create_task(worker_routine(url_worker, context, i)) for i in range(5)]
    proxy_task =  asyncio.to_thread(zmq.proxy, clients, workers)
    
    try:
        await asyncio.gather(*tasks, proxy_task)
    except KeyboardInterrupt:
        print("Server interrupted")
    except zmq.ZMQError as e:
        print("ZMQError:", e)
    finally:
        # We never get here but clean up anyhow
        clients.close()
        workers.close()
        context.term()


if __name__ == "__main__":
    asyncio.run(main())