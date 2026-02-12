import logging
import os
from typing import List

import torch

logger = logging.getLogger(__name__)
HF3FS_AVAILABLE = True


class Hf3fsClient:
    """Mock HF3FS client using file backend for debugging and testing."""

    def __init__(self, path: str, size: int, bytes_per_page: int, entries: int):
        self._size = size
        self._bytes_per_page = bytes_per_page
        self._entries = entries
        self._file_path = path

        self._ensure_file_exists()
        logger.debug("Initialized mock HF3FS client: %s (%d bytes)", path, size)

    def _ensure_file_exists(self) -> None:
        """Create file if it doesn't exist."""
        if not os.path.exists(self._file_path):
            with open(self._file_path, "w+b") as f:
                f.truncate(self._size)

    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        """Read data from file at specified offsets into tensors."""
        results = []

        try:
            with open(self._file_path, "rb") as f:
                for offset, tensor in zip(offsets, tensors):
                    num_bytes = tensor.numel() * tensor.element_size()

                    if offset < 0 or offset + num_bytes > self._size:
                        results.append(-1)
                        continue

                    f.seek(offset)
                    buffer_data = f.read(num_bytes)

                    if len(buffer_data) == num_bytes == self._bytes_per_page:
                        tensor_data = self._convert_buffer_to_tensor(
                            buffer_data, tensor.dtype
                        )
                        tensor.copy_(
                            tensor_data.reshape(tensor.shape).to(tensor.device)
                        )
                        results.append(self._bytes_per_page)
                    else:
                        logger.error(
                            "Read size mismatch: got %d, expected %d",
                            len(buffer_data),
                            num_bytes,
                        )
                        results.append(-1)
        except Exception as e:
            logger.error("Batch read error: %s", e)
            results.extend([-1] * (len(offsets) - len(results)))

        return results

    def _convert_buffer_to_tensor(
        self, buffer_data: bytes, dtype: torch.dtype
    ) -> torch.Tensor:
        """Convert buffer data to tensor with proper dtype handling."""
        if dtype == torch.bfloat16:
            tensor_data = torch.frombuffer(buffer_data, dtype=torch.uint16)
            return tensor_data.view(dtype=torch.bfloat16)
        else:
            return torch.frombuffer(buffer_data, dtype=dtype)

    def batch_write(
        self, offsets: List[int], tensors: List[torch.Tensor], event: torch.cuda.Event
    ) -> List[int]:
        """Write data from tensors to file at specified offsets."""
        results = []

        try:
            torch.cuda.current_stream().wait_event(event)

            # Convert tensors to bytes
            data_bytes_list = [self._tensor_to_bytes(tensor) for tensor in tensors]

            # Write to file
            with open(self._file_path, "r+b") as f:
                for offset, data_bytes in zip(offsets, data_bytes_list):
                    if offset < 0 or offset + len(data_bytes) > self._size:
                        results.append(-1)
                        continue

                    f.seek(offset)
                    bytes_written = f.write(data_bytes)

                    if bytes_written == len(data_bytes) == self._bytes_per_page:
                        results.append(self._bytes_per_page)
                    else:
                        logger.error(
                            "Write size mismatch: wrote %d, expected %d",
                            bytes_written,
                            self._bytes_per_page,
                        )
                        results.append(-1)

        except Exception as e:
            logger.error("Batch write error: %s", e)
            results.extend([-1] * (len(offsets) - len(results)))

        return results

    def _tensor_to_bytes(self, tensor: torch.Tensor) -> bytes:
        """Convert tensor to bytes with proper dtype handling."""
        cpu_tensor = tensor.cpu()
        if cpu_tensor.dtype == torch.bfloat16:
            return cpu_tensor.view(dtype=torch.uint16).numpy().tobytes()
        else:
            return cpu_tensor.numpy().tobytes()

    def get_size(self) -> int:
        """Get the total size of the storage file."""
        return self._size

    def close(self) -> None:
        """Close the client (no-op for file backend)."""
        pass

    def flush(self) -> None:
        """Flush any pending writes (no-op for file backend)."""
        pass
