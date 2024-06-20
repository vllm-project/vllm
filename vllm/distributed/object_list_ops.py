"""
This file is necessary until new version of torch.distributed is released with
https://github.com/pytorch/pytorch/commit/b96b1e8cff029bb0a73283e6e7f6cc240313f1dc
"""
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import (_get_pg_default_device,
                                                _object_to_tensor,
                                                _tensor_to_object)


def send_object_list(object_list, dst, group=None, device=None):
    """
    Sends picklable objects in ``object_list`` synchronously.

    Similar to :func:`send`, but Python objects can be passed in.
    Note that all objects in ``object_list`` must be picklable in order to be
    sent.

    Args:
        object_list (List[Any]): List of input objects to sent.
            Each object must be picklable. Receiver must provide lists of
            equal sizes.
        dst (int): Destination rank to send ``object_list`` to.
            Destination rank is based on global process group
            (regardless of ``group`` argument)
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, the objects are
            serialized and converted to tensors which are moved to the
            ``device`` before sending. Default is ``None``.

    Returns:
        ``None``.
    """
    if dist.get_rank() == dst:
        raise ValueError(
            "Invalid destination rank: destination rank should not be the "
            "same as the rank of the current process.")

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # sent to this device.
    current_device = device or _get_pg_default_device(group)
    # Serialize object_list elements to tensors on src rank.
    tensor_list, size_list = zip(
        *
        [_object_to_tensor(obj, current_device, group) for obj in object_list])
    object_sizes_tensor = torch.cat(size_list)

    # Send object sizes
    dist.send(object_sizes_tensor, dst=dst, group=group)

    # Concatenate and send serialized object tensors
    # Note: torch.cat will do an extra memory copy to the current device,
    # if the tensor_list has only one element, we can skip the copy.
    if len(tensor_list) == 1:  # type: ignore[possibly-undefined]
        object_tensor = tensor_list[0]
    else:
        object_tensor = torch.cat(tensor_list)

    dist.send(object_tensor, dst=dst, group=group)


def recv_object_list(object_list, src=None, group=None, device=None):
    """
    Receives picklable objects in ``object_list`` synchronously.

    Similar to :func:`recv`, but can receive Python objects.

    Args:
        object_list (List[Any]): List of objects to receive into.
            Must provide a list of sizes equal to the size of the list
            being sent.
        src (int, optional): Source rank from which to recv ``object_list``.
            Source rank is based on global process group
            (regardless of ``group`` argument)
            Will receive from any rank if set to None. Default is ``None``.
        group: (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used. Default is ``None``.
        device (``torch.device``, optional): If not None, receives on
            this device. Default is ``None``.

    Returns:
        Sender rank. -1 if rank is not part of the group. If rank is part 
        of the group, ``object_list`` will contain the sent objects from
        ``src`` rank.
    """

    # Current device selection.
    # To preserve backwards compatibility, ``device`` is default to ``None``
    # in which case we run current logic of device selection, i.e.
    # ``current_device`` is CUDA if backend is NCCL otherwise CPU device. In the
    # case it is not ``None`` we move the size and object tensors to be
    # received to this device.
    current_device = device or _get_pg_default_device(group)
    object_sizes_tensor = torch.empty(len(object_list),
                                      dtype=torch.long,
                                      device=current_device)

    # Receive object sizes
    rank_sizes = dist.recv(object_sizes_tensor, src=src, group=group)

    # Tensor to receive serialized objects into.
    object_tensor = torch.empty(  # type: ignore[call-overload]
        torch.sum(object_sizes_tensor).item(),  # type: ignore[arg-type]
        dtype=torch.uint8,
        device=current_device)

    rank_objects = dist.recv(object_tensor, src=src, group=group)
    assert (rank_sizes == rank_objects
            ), "Mismatch in return ranks for object sizes and objects."
    # Deserialize objects using their stored sizes.
    offset = 0
    for i, obj_size in enumerate(object_sizes_tensor):
        obj_view = object_tensor[offset:offset + obj_size]
        obj_view = obj_view.type(torch.uint8)
        offset += obj_size
        object_list[i] = _tensor_to_object(obj_view, obj_size, group)
    return rank_objects
