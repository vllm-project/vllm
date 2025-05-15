import ray
from uuid import uuid4
import zmq

@ray.remote
class Actor:
    def __init__(self, address, engine_index):
        self.address = address
        self.engine_index = engine_index

    def start(self):
        input_ctx = zmq.Context()
        self.socket = input_ctx.socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.engine_index.to_bytes(length=2, byteorder="little"))
        self.socket.connect(self.address)
    
    def loop(self):
        while True:
            print("waiting for message")
            message = self.socket.recv_multipart()
            print("received message", message)

def get_open_zmq_ipc_path() -> str:
    return f"ipc:///tmp/{uuid4()}"

if __name__ == "__main__":
    input_ctx = zmq.Context()
    address = get_open_zmq_ipc_path()
    socket = input_ctx.socket(zmq.ROUTER)
    socket.bind(address)
    
    actors = []
    for engine_index in range(2):
        actor = Actor.remote(address, engine_index)
        ray.get(actor.start.remote())
        actor.loop.remote()
        actors.append(actor)

    for i in range(10):
        if i % 2 == 0:
            engine_index = 0
        else:
            engine_index = 1
        
        identity = engine_index.to_bytes(length=2, byteorder="little")
        data = "hello world " + str(i)
        message = (identity, data.encode("utf-8"))
        print("sending message ", message)
        socket.send_multipart(message)
        print("sent message")

    import time
    time.sleep(1000)