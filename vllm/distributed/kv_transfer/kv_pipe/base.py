
from abc import ABC, abstractmethod


class KVPipeBase(ABC):
    
    @abstractmethod 
    def send_tensor(self, tensor):
        raise NotImplementedError
    
    @abstractmethod 
    def recv_tensor(self):
        raise NotImplementedError

    @abstractmethod
    def close(self):
        raise NotImplementedError
