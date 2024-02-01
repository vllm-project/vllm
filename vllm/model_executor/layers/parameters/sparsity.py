import torch

from magic_wand import SparseTensor, SparseBitmaskStorageFormat


class SparseParameter(SparseTensor):

    @staticmethod
    def __new__(
        cls,
        shape: torch.Size,
        dtype: torch.dtype,
    ):
        assert torch.__version__ > (1,
                                    10), "SparseTensor requires PyTorch 1.11+"
        self = torch.Tensor._make_wrapper_subclass(cls,
                                                   size=shape,
                                                   dtype=dtype,
                                                   requires_grad=False)
        self.storage_format_cls = SparseBitmaskStorageFormat
        self.compressed_data = None
        self.dense_data = None
        self._is_param = True

        return self

    def get_dense_data(self) -> torch.Tensor:
        if self.dense_data is not None:
            raise ValueError(
                "Called get_data_dense() but dense_data already exists.")
        self.dense_data = self._unpack()
        return self.dense_data

    def _unpack(self) -> torch.Tensor:
        if self.has_compressed_data():
            return self.compressed_data.decompress()
        else:
            return torch.empty(size=self.shape,
                               dtype=self.dtype,
                               device="cuda")

    def pack(self) -> None:
        if self.dense_data is None:
            raise ValueError("Called pack() but dense_data does not exist.")
        self.copy_(self.dense_data)
        self.dense_data = None
