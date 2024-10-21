import pika


class RabbitMQ:
    def __init__(self,
                 host: str,
                 port: int,
                 queue_name: str):
        self.parameters = pika.ConnectionParameters(host=host, port=port)
        self.queue_name = queue_name

        self._connect()

    def _connect(self):
        try:
            self.connection = pika.BlockingConnection(self.parameters)
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=True)
        except Exception as e:
            raise RuntimeError(f"Unable to connect RabbitMQ, {e}.")
    
    def push(self, message):
        if self.channel.is_closed:
            self._connect()

        try:
            self.channel.basic_publish(exchange='', routing_key=self.queue_name, body=message)
            return
        except Exception as e:
            raise RuntimeError(f"Unable to push msg to RabbitMQ, {e}.")

    def pull(self, auto_ack: bool = False):
        if self.channel.is_closed:
            self._connect()

        try:
            method_frame, _, body = self.channel.basic_get(self.queue_name, auto_ack=auto_ack)
            if method_frame:
                return method_frame, body
            else:
                return None, None
        except Exception as e:
            raise RuntimeError(f"Unable to pull msg from RabbitMQ, {e}.")

    def ack(self, method_frame):
        try:
            self.channel.basic_ack(method_frame.delivery_tag)
        except Exception as e:
            raise RuntimeError(f"Unable to ack msg to RabbitMQ, {e}.")

    def reject(self, method_frame):
        self.channel.basic_reject(delivery_tag=method_frame.delivery_tag)

    def close(self):
        self.connection.close()