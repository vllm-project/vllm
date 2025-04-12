# SPDX-License-Identifier: Apache-2.0

import base64
import io
from pathlib import Path

import torch

from vllm.logger import init_logger
from vllm.utils import PlaceholderModule

from .base import MediaIO

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.csv as pac
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
except ImportError:
    pd = PlaceholderModule("pandas")  # type: ignore[assignment]
    pa = PlaceholderModule("pyarrow")  # type: ignore[assignment]
    pq = PlaceholderModule("pyarrow.parquet")  # type: ignore[assignment]
    ipc = PlaceholderModule("pyarrow.ipc")  # type: ignore[assignment]
    pac = PlaceholderModule("pyarrow.csv")  # type: ignore[assignment]

logger = init_logger(__name__)


class TimeSeriesMediaIO(MediaIO[torch.Tensor]):
    """Media I/O operations for time series data."""

    def load_bytes(self, data: bytes) -> torch.Tensor:
        """Load time series data from bytes by detecting format.
        
        Supports Parquet, Arrow File, and CSV formats.
        Format is detected by the first 8 bytes of the bytes.
        """
        buffer = io.BytesIO(data)

        # Read first 8 bytes to identify file format
        magic_bytes = buffer.read(8)
        buffer.seek(0)  # Reset position after reading

        # Check for Parquet (starts with 'PAR1')
        if magic_bytes.startswith(b'PAR1'):
            try:
                table = pq.read_table(buffer)
                return table.to_pandas().values.flatten()
            except Exception as e:
                raise ValueError(f"Invalid Parquet file: {str(e)}") from e

        # Check for Arrow IPC (starts with 'ARROW1')
        elif magic_bytes.startswith(b'ARROW1'):
            try:
                reader = pa.ipc.open_file(buffer)
                table = reader.read_all()
                return table.to_pandas().values.flatten()
            except pa.lib.ArrowInvalid as e:
                raise ValueError(f"Invalid Arrow file: {str(e)}") from e

        # Default to CSV (no reliable magic bytes)
        else:
            try:
                table = pac.read_csv(buffer)
                return table.to_pandas().values.flatten()
            except Exception as e:
                raise ValueError(
                    f"Invalid CSV file or unknown format: {str(e)}") from e

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        """Load time series data from base64 encoded string."""
        return self.load_bytes(base64.b64decode(data))

    def load_file(self, filepath: Path) -> torch.Tensor:
        """Load time series data from a file."""
        suffix = filepath.suffix.lower()

        if suffix == '.parquet':
            try:
                table = pq.read_table(filepath)
                return table.to_pandas().values.flatten()
            except Exception as e:
                raise ValueError(f"Invalid Parquet file: {str(e)}") from e
        elif suffix in ['.arrow', '.ipc']:
            try:
                reader = ipc.open_file(filepath)
                table = reader.read_all()
                return table.to_pandas().values.flatten()
            except Exception as e:
                raise ValueError(f"Invalid Arrow file: {str(e)}") from e
        elif suffix == '.csv':
            try:
                table = pac.read_csv(filepath)
                return table.to_pandas().values.flatten()
            except Exception as e:
                raise ValueError(f"Invalid CSV file: {str(e)}") from e
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

    def encode_base64(self,
                      media: list[list[float]],
                      format: str = "csv") -> str:
        """Encode time series data to base64 string."""
        with io.BytesIO() as buffer:
            if format == "csv":
                df = pd.DataFrame(media)
                df.to_csv(buffer, index=False)
            elif format == "parquet":
                df = pd.DataFrame(media)
                df.to_parquet(buffer)
            elif format == "arrow":
                df = pd.DataFrame(media)
                df.to_parquet(buffer, engine="pyarrow")
            else:
                raise ValueError(f"Unsupported format: {format}")
            data = buffer.getvalue()
        return base64.b64encode(data).decode('utf-8')
