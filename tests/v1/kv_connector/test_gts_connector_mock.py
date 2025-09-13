# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Mock GTS client for testing GTS connector without the actual GTS SDK
"""
from unittest.mock import MagicMock
from typing import Any, Dict, List, Optional, Union
import torch


class MockGTSFuture:
    """Mock future for async operations"""
    
    def __init__(self, ready: bool = True):
        self._ready = ready
        self._result = None
    
    def wait(self):
        """Block until operation completes"""
        self._ready = True
    
    def is_ready(self) -> bool:
        """Check if operation is complete"""
        return self._ready
    
    def result(self):
        """Get operation result"""
        return self._result


class MockTensorSelector:
    """Mock tensor selector for building queries"""
    
    def __init__(self, tensor_name: str):
        self.tensor_name = tensor_name
        self._conditions = {}
    
    def where(self, **kwargs):
        """Add dimension constraints"""
        self._conditions.update(kwargs)
        return self
    
    def build(self):
        """Build the selector"""
        return {"tensor": self.tensor_name, "conditions": self._conditions}


class MockGTSClient:
    """Mock GTS client that simulates the expected interface"""
    
    def __init__(self, server_address: str):
        self.server_address = server_address
        self.registered_tensors: Dict[str, Any] = {}
        self.tensor_metadata: Dict[str, Dict] = {}
        self.copy_operations: List[Dict] = []
        
    def create_and_register(self, 
                           name: str,
                           dims: List,
                           dtype: torch.dtype,
                           device: str,
                           data: Optional[torch.Tensor] = None) -> Any:
        """Mock tensor registration"""
        handle = MagicMock()
        handle.name = name
        handle.dims = dims
        handle.dtype = dtype
        handle.device = device
        
        # Store registration info
        self.registered_tensors[name] = handle
        self.tensor_metadata[name] = {
            'dims': dims,
            'dtype': dtype,
            'device': device,
            'shape': data.shape if data is not None else None
        }
        
        print(f"Mock GTS: Registered tensor '{name}' with dims {dims}")
        return handle
    
    def tensor(self, name: str) -> MockTensorSelector:
        """Create tensor selector"""
        return MockTensorSelector(name)
    
    def copy(self, src, dst, options: Optional[Dict] = None) -> MockGTSFuture:
        """Mock copy operation"""
        copy_op = {
            'src': src,
            'dst': dst,
            'options': options or {},
            'timestamp': torch.tensor(0).float().item()  # Mock timestamp
        }
        self.copy_operations.append(copy_op)
        
        print(f"Mock GTS: Copy from {src} to {dst} with options {options}")
        
        # Return future that completes based on async flag
        is_async = options and options.get('async', False)
        return MockGTSFuture(ready=not is_async)
    
    def get_registered_tensors(self) -> Dict[str, Any]:
        """Get all registered tensors (for testing)"""
        return self.registered_tensors
    
    def get_copy_operations(self) -> List[Dict]:
        """Get all copy operations (for testing)"""
        return self.copy_operations


# Mock the gts module
class MockGTS:
    """Mock GTS module"""
    Client = MockGTSClient
    
    # Constants that might be used
    ALL = "ALL" 
    
    @staticmethod
    def slice(start: int, end: int):
        return {"type": "slice", "start": start, "end": end}


# Patch the gts import for testing
import sys
sys.modules['gts'] = MockGTS()


if __name__ == "__main__":
    # Test the mock client
    client = MockGTSClient("localhost:6174")
    
    # Test tensor registration
    tensor = torch.randn(10, 2, 64, 8, 128, dtype=torch.bfloat16, device='cpu')
    handle = client.create_and_register(
        name="test/layer:0/kv",
        dims=[
            ("blocks", 10, (0, 10)),
            ("kv", 2, (0, 2)),
            ("tokens", 64, (0, 64)),
            ("heads", 8, (0, 4)),  # Shard
            ("head_dim", 128, (0, 128))
        ],
        dtype=tensor.dtype,
        device='cpu',
        data=tensor
    )
    
    # Test tensor selector
    src = client.tensor(tensor).where(blocks=[0, 1]).where(kv=MockGTS.ALL)
    dst = client.tensor("role:kv_consumer")
    
    # Test copy
    future = client.copy(src.build(), dst, options={"async": True})
    print(f"Copy ready: {future.is_ready()}")
    
    future.wait()
    print(f"Copy ready after wait: {future.is_ready()}")
    
    print(f"Registered tensors: {list(client.get_registered_tensors().keys())}")
    print(f"Copy operations: {len(client.get_copy_operations())}")