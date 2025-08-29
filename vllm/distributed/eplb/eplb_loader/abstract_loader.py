from abc import abstractmethod


class BaseLoader:
    @abstractmethod
    def prepare_send(self):
        pass

    @abstractmethod
    def prepare_recv(self):
        pass

    @abstractmethod
    def send_recv(self):
        pass

    @abstractmethod
    def update_weight(self):
        pass
