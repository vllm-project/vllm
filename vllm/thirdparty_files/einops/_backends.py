"""
Backends in `einops` are organized to meet the following requirements
- backends are not imported unless those are actually needed, because
    - backends may not be installed
    - importing all available backends will drive to significant memory footprint
    - backends may be present but installed with errors (but never used),
      importing may drive to crashes
- backend should be either symbolic or imperative
    - this determines which methods (from_numpy/to_numpy or create_symbol/eval_symbol) should be defined
- if backend can't provide symbols for shape dimensions, UnknownSize objects are used
"""

import sys

__author__ = "Alex Rogozhnikov"

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False


def get_backend(tensor) -> "AbstractBackend":
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
    """
    _type = type(tensor)
    _result = _type2backend.get(_type, None)
    if _result is not None:
        return _result

    for framework_name, backend in list(_loaded_backends.items()):
        if backend.is_appropriate_type(tensor):
            _type2backend[_type] = backend
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print("Testing for subclass of ", BackendSubclass)
        if BackendSubclass.framework_name not in _loaded_backends:
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print("Imported backend for ", BackendSubclass.framework_name)
                backend = BackendSubclass()
                _loaded_backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    _type2backend[_type] = backend
                    return backend

    raise RuntimeError("Tensor type unknown to einops {}".format(type(tensor)))


class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes."""

    framework_name: str

    def is_appropriate_type(self, tensor):
        """helper method should recognize tensors it can handle"""
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, input_dict):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop):
        # supplementary method used only in testing, so should implement CPU version
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats - same lengths as x.shape"""
        raise NotImplementedError()

    def concat(self, tensors, axis: int):
        """concatenates tensors along axis.
        Assume identical across tensors: devices, dtypes and shapes except selected axis."""
        raise NotImplementedError()

    def is_float_type(self, x):
        # some backends (torch) can't compute average for non-floating types.
        # Decided to drop average for all backends if type is not floating
        raise NotImplementedError()

    def layers(self):
        raise NotImplementedError("backend does not provide layers")

    def __repr__(self):
        return "<einops backend for {}>".format(self.framework_name)

    def einsum(self, pattern, *x):
        raise NotImplementedError("backend does not support einsum")


class UnknownSize:
    """pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements"""

    def __floordiv__(self, other):
        return self

    def __eq__(self, other):
        return True  # we don't know actual size

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __hash__(self):
        return hash(None)


class NumpyBackend(AbstractBackend):
    framework_name = "numpy"

    def __init__(self):
        import numpy

        self.np = numpy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return x

    def to_numpy(self, x):
        return x

    def arange(self, start, stop):
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.np.stack(tensors)

    def tile(self, x, repeats):
        return self.np.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.np.concatenate(tensors, axis=axis)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def add_axis(self, x, new_position):
        return self.np.expand_dims(x, new_position)

    def einsum(self, pattern, *x):
        return self.np.einsum(pattern, *x)


class JaxBackend(NumpyBackend):
    framework_name = "jax"

    def __init__(self):
        super(JaxBackend, self).__init__()
        self.onp = self.np

        import jax.numpy

        self.np = jax.numpy

    def from_numpy(self, x):
        return self.np.asarray(x)

    def to_numpy(self, x):
        return self.onp.asarray(x)


class TorchBackend(AbstractBackend):
    framework_name = "torch"

    def __init__(self):
        import torch

        self.torch = torch
        # importing would register operations in torch._dynamo for torch.compile
        from . import _torch_specific  # noqa

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def from_numpy(self, x):
        variable = self.torch.from_numpy(x)
        if self.is_float_type(variable):
            # attach grad only to floating types
            variable.requires_grad = True
        return variable

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop):
        return self.torch.arange(start, stop, dtype=self.torch.int64)

    def reduce(self, x, operation, reduced_axes):
        if operation == "min":
            return x.amin(dim=reduced_axes)
        elif operation == "max":
            return x.amax(dim=reduced_axes)
        elif operation == "sum":
            return x.sum(dim=reduced_axes)
        elif operation == "mean":
            return x.mean(dim=reduced_axes)
        elif operation in ("any", "all", "prod"):
            # pytorch supports reducing only one operation at a time
            for i in list(sorted(reduced_axes))[::-1]:
                x = getattr(x, operation)(dim=i)
            return x
        else:
            raise NotImplementedError("Unknown reduction ", operation)

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.torch.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return self.torch.cat(tensors, dim=axis)

    def add_axis(self, x, new_position):
        return self.torch.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in [self.torch.float16, self.torch.float32, self.torch.float64, self.torch.bfloat16]

    def layers(self):
        from .layers import torch

        return torch

    def einsum(self, pattern, *x):
        return self.torch.einsum(pattern, *x)


class CupyBackend(AbstractBackend):
    framework_name = "cupy"

    def __init__(self):
        import cupy

        self.cupy = cupy

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.cupy.ndarray)

    def from_numpy(self, x):
        return self.cupy.asarray(x)

    def to_numpy(self, x):
        return self.cupy.asnumpy(x)

    def arange(self, start, stop):
        return self.cupy.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.cupy.stack(tensors)

    def tile(self, x, repeats):
        return self.cupy.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.cupy.concatenate(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return self.cupy.expand_dims(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def einsum(self, pattern, *x):
        return self.cupy.einsum(pattern, *x)


class ChainerBackend(AbstractBackend):
    framework_name = "chainer"

    def __init__(self):
        import chainer
        import numpy

        self.numpy = numpy
        self.chainer = chainer

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.chainer.Variable)

    def from_numpy(self, x):
        return self.chainer.Variable(x.astype("float32"))

    def to_numpy(self, x):
        if isinstance(x, self.chainer.Variable):
            x = x.data
        return x

    def arange(self, start, stop):
        return self.numpy.arange(start, stop)

    def reduce(self, x, operation, axes):
        return getattr(self.chainer.functions, operation)(x, axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.chainer.functions.stack(tensors)

    def tile(self, x, repeats):
        return self.chainer.functions.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.chainer.functions.concat(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return self.chainer.functions.expand_dims(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def layers(self):
        from .layers import chainer

        return chainer

    def einsum(self, pattern, *x):
        return self.chainer.functions.einsum(pattern, *x)


class HashableTuple:
    """Overcomes non-hashability of symbolic elements"""

    def __init__(self, elements: tuple):
        self.elements = elements

    def __iter__(self):
        for x in self.elements:
            yield x

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    # default equality and hash is used (True only with itself, hash taken of id)


class TensorflowBackend(AbstractBackend):
    framework_name = "tensorflow"

    def __init__(self):
        import tensorflow

        self.tf = tensorflow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.tf.Tensor, self.tf.Variable))

    def from_numpy(self, x):
        assert self.tf.executing_eagerly()
        return self.tf.convert_to_tensor(x)

    def to_numpy(self, x):
        assert self.tf.executing_eagerly()
        return x.numpy()

    def arange(self, start, stop):
        return self.tf.range(start, stop)

    def shape(self, x):
        if self.tf.executing_eagerly():
            return tuple(UnknownSize() if d is None else int(d) for d in x.shape)
        else:
            static_shape = x.shape.as_list()
            tf_shape = self.tf.shape(x)
            # use the static shape where known, otherwise use the TF shape components
            shape = tuple([s or tf_shape[dim] for dim, s in enumerate(static_shape)])
            try:
                hash(shape)
                return shape
            except:
                # unhashable symbols in shape. Wrap tuple to be hashable.
                return HashableTuple(shape)

    def reduce(self, x, operation, axes):
        return getattr(self.tf, "reduce_" + operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.tf.reshape(x, shape)

    def transpose(self, x, axes):
        return self.tf.transpose(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.tf.stack(tensors)

    def tile(self, x, repeats):
        return self.tf.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.tf.concat(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return self.tf.expand_dims(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def layers(self):
        from .layers import tensorflow

        return tensorflow

    def einsum(self, pattern, *x):
        return self.tf.einsum(pattern, *x)


class KerasBackend(AbstractBackend):
    framework_name = "tensorflow.keras"

    def __init__(self):
        import tensorflow as tf

        self.tf = tf
        self.keras = tf.keras
        self.K = tf.keras.backend

    def is_appropriate_type(self, tensor):
        return self.tf.is_tensor(tensor) and self.K.is_keras_tensor(tensor)

    def create_symbol(self, shape):
        return self.keras.Input(batch_shape=shape)

    def eval_symbol(self, symbol, input_dict):
        model = self.keras.models.Model([var for (var, _) in input_dict], symbol)
        return model.predict_on_batch([val for (_, val) in input_dict])

    def arange(self, start, stop):
        return self.K.arange(start, stop)

    def shape(self, x):
        shape = self.K.shape(x)  # tf tensor
        return HashableTuple(tuple(shape))

    def reduce(self, x, operation, axes):
        return getattr(self.K, operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.K.reshape(x, shape)

    def transpose(self, x, axes):
        return self.K.permute_dimensions(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.K.stack(tensors)

    def tile(self, x, repeats):
        return self.K.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.K.concatenate(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return self.K.expand_dims(x, new_position)

    def is_float_type(self, x):
        return "float" in self.K.dtype(x)

    def layers(self):
        from .layers import keras

        return keras


class OneFlowBackend(AbstractBackend):
    framework_name = "oneflow"

    def __init__(self):
        import oneflow as flow

        self.flow = flow

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.flow.Tensor)

    def from_numpy(self, x):
        variable = self.flow.from_numpy(x)
        if self.is_float_type(variable):
            # attach grad only to floating types
            variable.requires_grad = True
        return variable

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop):
        return self.flow.arange(start, stop, dtype=self.flow.int64)

    def reduce(self, x, operation, reduced_axes):
        for axis in sorted(reduced_axes, reverse=True):
            if operation == "min":
                x, _ = x.min(dim=axis)
            elif operation == "max":
                x, _ = x.max(dim=axis)
            elif operation in ["sum", "mean", "prod", "any", "all"]:
                x = getattr(x, operation)(dim=axis)
            else:
                raise NotImplementedError("Unknown reduction ", operation)
        return x

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.flow.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(*repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return self.flow.concat(tensors, dim=axis)

    def add_axis(self, x, new_position):
        return self.flow.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in [self.flow.float16, self.flow.float32, self.flow.float64]

    def layers(self):
        from .layers import oneflow

        return oneflow

    def einsum(self, pattern, *x):
        return self.flow.einsum(pattern, *x)


class PaddleBackend(AbstractBackend):
    framework_name = "paddle"

    def __init__(self):
        import paddle

        self.paddle = paddle

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.paddle.Tensor, self.paddle.static.Variable))

    def from_numpy(self, x):
        tensor = self.paddle.to_tensor(x)
        tensor.stop_gradient = False
        return tensor

    def to_numpy(self, x):
        return x.detach().numpy()

    def arange(self, start, stop):
        return self.paddle.arange(start, stop, dtype=self.paddle.int64)

    def reduce(self, x, operation, axes):
        if len(axes) == x.ndim:
            # currently paddle returns 1d tensor instead of 0d
            return super().reduce(x, operation, axes).squeeze(0)
        else:
            return super().reduce(x, operation, axes)

    def transpose(self, x, axes):
        return x.transpose(axes)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.paddle.stack(tensors)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def tile(self, x, repeats):
        return x.tile(repeats)

    def concat(self, tensors, axis: int):
        return self.paddle.concat(tensors, axis=axis)

    def add_axis(self, x, new_position):
        return x.unsqueeze(new_position)

    def is_float_type(self, x):
        return x.dtype in [self.paddle.float16, self.paddle.float32, self.paddle.float64]

    def layers(self):
        from .layers import paddle

        return paddle

    def einsum(self, pattern, *x):
        return self.paddle.einsum(pattern, *x)

    def shape(self, x):
        return tuple(x.shape)
