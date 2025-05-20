# SPDX-License-Identifier: Apache-2.0
import contextlib
import math
import threading
import time
import uuid
from collections import defaultdict, OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import torch
import zmq

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group)
from vllm.logger import init_logger
from vllm.utils import make_zmq_path, make_zmq_socket, round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

logger = init_logger(__name__)

try:
    from nixl._api import nixl_agent as NixlWrapper
    logger.info("NIXL is available")
except ImportError:
    logger.warning("NIXL is not available")
    NixlWrapper = None

###################################################################
# Helper classes and functions
###################################################################

def init_nixl_agent(
    buffer_size: int,
    buffer_ptr: int,
    nixl_page_size: int = 4096,
) -> tuple[NixlWrapper, Any, Any]:
    """Initialize the NIXL agent.

    Args:
        buffer_size (int): The size of the buffer.
        buffer_ptr (int): The pointer to the buffer.
        nixl_page_size (int, optional): The page size of NIXL. Defaults to 4096.

    Returns:
        NixlWrapper: The NIXL agent.
        reg_dlist: the registered memory descriptor list.
        xfer_dlist: the local transfer descriptor list.
    """
    if NixlWrapper is None:
        raise RuntimeError("NIXL is not available")

    # Create a NIXL agent
    nixl_agent = NixlWrapper(str(uuid.uuid4()))

    # Register the memory
    memory_desc = [(buffer_ptr, buffer_size, 0, "")]
    reg_descs = nixl_agent.get_reg_descs(memory_desc, mem_type="DRAM")
    nixl_agent.register_memory(reg_descs)

    # Create xfer handlers
    xfer_desc = []
    for base_addr in range(buffer_ptr, 
                           buffer_ptr + buffer_size,
                           nixl_page_size):
        xfer_desc.append((base_addr, nixl_page_size, 0))

    descs = nixl_agent.get_xfer_descs(xfer_desc, mem_type="DRAM")
    local_xfer_dlist = nixl_agent.prep_xfer_dlist(
            "", descs, mem_type="DRAM")

    return nixl_agent, reg_descs, local_xfer_dlist

@dataclass
class DestinationSpec:
    """DestinationSpec is used to specify the destination of kv sending task.

    Attributes:
        rank (int): The rank of the destination.
        host (str): The path of the destination.
        base_port (int): The base port of the destination.
    """
    rank: int
    host: str
    base_port: int

    def __str__(self) -> str:
        return f"DestinationSpec(rank={self.rank}, host={self.host}, base_port={self.base_port})"

    def get_id(self) -> str:
        """Get the id of the destination spec.

        Returns:
            str: The id of the destination spec.
        """
        return f"{self.rank}_{self.host}_{self.base_port}"

class SourceSpec(msgspec.Struct):
    """SourceSpec is used to specify the source of kv sending task.
    """
    # The request id of the kv cache
    request_id: str

    # The layer id of the kv cache
    layer_id: int

    # The range of tokens to be offloaded
    start: int  # For token_range slice
    stop: int   # For token_range slice

    # The shape of the offloaded KV cache tensor as a tuple
    shape: tuple[int, ...]

    # The dtype of the offloaded KV cache tensor as a string
    dtype_str: str

    @property
    def token_range(self) -> slice:
        """Get the token range as a slice object."""
        return slice(self.start, self.stop)

    @property
    def tensor_shape(self) -> torch.Size:
        """Get the shape as a torch.Size object."""
        return torch.Size(self.shape)

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype as a torch.dtype object."""
        return getattr(torch, self.dtype_str)

    def get_size(self) -> int:
        """Get the size in bytes of the cooresponding kv cache."""
        return math.prod(self.shape) * self.dtype.itemsize

    def __str__(self) -> str:
        return (f"SourceSpec(request_id={self.request_id}, "
                f"layer_id={self.layer_id}, "
                f"token_range={self.token_range}, shape={self.tensor_shape})")

class RingBufferAllocator:
    """RingBufferAllocator is a simple ring buffer allocator for managing
    memory allocation and deallocation.
    """

    def __init__(self, size: int, align_to: int = 256) -> None:
        """Initialize the ring buffer allocator with the given size.

        Args:
            size (int): The size of the ring buffer (in bytes).
            align_to (int): The alignment size (in bytes). Default is 8.
        """
        self._size = size
        self._buffer = torch.empty(size, dtype=torch.uint8)
        self._high_watermark = 0
        self._low_watermark = 0
        self._align_to = align_to

        self._allocated = OrderedDict()  # Track allocated buffers

        # Register pin memory
        cudart = torch.cuda.cudart()
        cudart.cudaHostRegister(self._buffer.data_ptr(), size, 0)

    def _align_size(self, base: int) -> int:
        """Align the given size to the nearest multiple of the alignment size.

        Args:
            base (int): The size to be aligned.

        Returns:
            int: The aligned size.
        """
        return ((base - 1) // self._align_to + 1) * self._align_to

    def allocate(self, size: int) -> tuple[int, Optional[torch.Tensor]]:
        """Allocate a buffer of the given size.

        Args:
            size (int): The size of the buffer to be allocated.

        Returns:
            Optional[tuple[int, torch.Tensor]]: A tuple containing the virtual 
                address of the allocated buffer and the buffer itself. If 
                allocation fails, returns None.
        """
        # During allocation, we always make sure that high watermark and
        # low watermark are aligned to the alignment size
        aligned_size = self._align_size(size)   # Align the requested size
        turnaround_size = (self._high_watermark // self._size + 1) * self._size

        local_high = self._high_watermark % self._size
        local_low = self._low_watermark % self._size

        if local_high >= local_low:
            if local_high == local_low and \
                    self._high_watermark > self._low_watermark:
                # No space available
                return -1, None

            # If high watermark + requested size is okay, directly allocate
            if local_high + size < self._size:
                address = self._high_watermark
                self._allocated[address] = aligned_size
                start = local_high
                end = start + size
                self._high_watermark += aligned_size
                return address, self._buffer[start:end]
            else:
                # If high watermark + requested size is not okay, we need to
                # wrap around and allocate again
                self._high_watermark = turnaround_size
                return self.allocate(size)
        else:
            # High watermark is below low watermark, check if we can allocate
            if local_high + size < local_low:
                address = self._high_watermark
                self._allocated[address] = aligned_size
                start = local_high
                end = start + size
                self._high_watermark += aligned_size
                return address, self._buffer[start:end]
            else:
                # No space available
                return -1, None

    def free(self, address: int) -> None:
        """Free the buffer at the given address.

        Args:
            address (int): The virtual address of the buffer to be freed,
                which is returned by the allocate() method.
        """
        assert address in self._allocated, \
                "Address not found in allocated buffers"

        # Pop the address from the allocated dict, and update the 
        # low watermark
        self._allocated.pop(address)

        # If there is nothing allocated, set low_watermark to high watermark
        new_low_watermark = self._high_watermark

        # Else, set the low_watermark to the first address in the allocated
        # dict
        for addr in self._allocated.keys():
            new_low_watermark = addr
            break
        self._low_watermark = new_low_watermark

    @property
    def high_watermark(self) -> int:
        return self._high_watermark

    @property
    def low_watermark(self) -> int:
        return self._low_watermark

    def virtual_to_physical(self, vaddr: int) -> torch.Tensor:
        """Convert a virtual address to a physical address.

        Args:
            vaddr (int): The virtual address to be converted.

        Returns:
            torch.Tensor: The physical address of the buffer.
        """
        return vaddr + self._size

    def get_size(self) -> int:
        """Get the size of the ring buffer.

        Returns:
            int: The size of the ring buffer.
        """
        return self._size

    def get_buffer_ptr(self) -> int:
        """Get the pointer to the buffer.

        Returns:
            int: The pointer to the buffer.
        """
        return self._buffer.data_ptr()

###################################################################
# NIXL Related Classes
###################################################################

class NixlProtocolMsg(msgspec.Struct):
    msg_type: str
    req_uuid: str
    source_spec: Optional[SourceSpec] = None
    receiver_addr: Optional[int] = None



def make_send_req_msg(
    source_spec: SourceSpec,
    req_uuid: str
) -> bytes:
    """Make the send request message.

    Args:
        source_spec (SourceSpec): The source spec.

    Returns:
        bytes: The send request message.
    """
    # Create the request message
    msg_type = "REQMSG"
    receiver_addr = None
    send_req_msg = NixlProtocolMsg(
        msg_type=msg_type,
        req_uuid=req_uuid,
        source_spec=source_spec,
        receiver_addr=receiver_addr
    )
    # Encode the message
    send_req_msg_bytes = msgspec.msgpack.encode(send_req_msg)
    return send_req_msg_bytes

def make_receive_ready_msg(
        req_uuid: str,
        receiver_addr: int,
) -> bytes:
    """Make the receive ready message.

    Args:
        req_uuid (str): The request uuid.
        receiver_addr (int): The receiver address.

    Returns:
        bytes: The receive ready message.
    """
    # Create the request message
    msg_type = "READYMSG"
    source_spec = None
    receive_ready_msg = NixlProtocolMsg(
        msg_type=msg_type,
        req_uuid=req_uuid,
        source_spec=source_spec,
        receiver_addr=receiver_addr
    )
    # Encode the message
    receive_ready_msg_bytes = msgspec.msgpack.encode(receive_ready_msg)
    return receive_ready_msg_bytes

def make_send_finish_msg(
        req_uuid: str,
) -> bytes:
    """Make the send finish message.

    Args:
        req_uuid (str): The request uuid.

    Returns:
        bytes: The send finish message.
    """
    # Create the request message
    msg_type = "FINISHMSG"
    source_spec = None
    receiver_addr = None
    send_finish_msg = NixlProtocolMsg(
        msg_type=msg_type,
        req_uuid=req_uuid,
        source_spec=source_spec,
        receiver_addr=receiver_addr
    )
    # Encode the message
    send_finish_msg_bytes = msgspec.msgpack.encode(send_finish_msg)
    return send_finish_msg_bytes


class NixlCPUSender:
    def __init__(
        self,
        buffer_size: int,
        buffer_ptr: int,
        nixl_page_size: int = 4096,
    ) -> None:
        self._buffer_size = buffer_size
        self._buffer_ptr = buffer_ptr
        self._nixl_page_size = nixl_page_size

        # Destination spec id -> peer name
        self._remote_agents: dict[str, str] = {}

        self._nixl_wrapper, self._reg_dlist, self._local_xfer_dlist = \
            init_nixl_agent(buffer_size, buffer_ptr, nixl_page_size)

        # Add ZMQ context for handshakes
        self._zmq_ctx = zmq.Context()

        # Requests that are ready to send
        # uuid -> remote agent name
        self._ready_requests: dict[str, str] = {}

        # NOTE(ApostaC): we don't track the requests that are waiting for the 
        # receiver to be ready, and may want to add this in the future

        # Msg decoder
        self._msg_decoder = msgspec.msgpack.Decoder(NixlProtocolMsg)

    def send(
        self,
        src_addr: int,
        dst_addr: int,
        data_size: int
    ) -> None:
        """Send data from src_addr to dst_addr using NIXL.

        Args:
            src_addr (int): Source address.
            dst_addr (int): Destination address.
            data_size (int): Size of the data in bytes to be sent.
        """
        pass
    
    def prepare_send(
        self,
        source_spec: SourceSpec,
        destination_spec: DestinationSpec,
    ) -> str:
        """Prepare the send operation by allocation the receive buffer 
        on the destination side.

        Args:
            source_spec (SourceSpec): The source spec.
            destination_spec (DestinationSpec): The destination spec.

        Returns:
            str: The uuid of the prepared send
        """
        dest_id = destination_spec.get_id()
        if dest_id not in self._remote_agents:
            # Perform handshake with the destination
            self._nixl_handshake(destination_spec)

        remote_agent_name = self._remote_agents[dest_id]

        # Create the request message
        req_uuid = str(uuid.uuid4())
        msg = make_send_req_msg(source_spec, req_uuid)

        # Send it to the remote agent
        self._nixl_wrapper.send_notif(remote_agent_name, msg)

        return req_uuid

    def check_and_remove_prepared_send(
        self,
        send_uuid: str,
    ) -> Optional[str]:
        """Check if the prepared send is ready to be sent.
        If the send is ready, remove it from the ready requests.

        Args:
            send_uuid (str): The uuid of the prepared send.

        Returns:
            Optional[str]: The remote agent name if the send is ready,
                None otherwise.
        """
        # Update the ready requests
        notifs = self._nixl_wrapper.get_new_notifs()
        for remote_agent_name in notifs:
            for msg in notifs[remote_agent_name]:
                # Decode the message
                obj = self._msg_decoder.decode(msg)
                if obj.msg_type == "READYMSG":
                    # Add the request to the ready requests
                    self._ready_requests[obj.req_uuid] = remote_agent_name
                else:
                    logger.error("Unexpected message type: %s", obj.msg_type)
                    continue

        # Check if the send uuid is in the ready requests
        if send_uuid in self._ready_requests:
            # Remove the request from the ready requests
            remote_agent_name = self._ready_requests.pop(send_uuid)
            return remote_agent_name
        else:
            return None

    def _nixl_handshake(self, destination_spec: DestinationSpec) -> None:
        """Perform handshake with a remote NIXL CPU instance.
        
        Args:
            destination_spec (DestinationSpec): The destination spec.
        """
        assert get_tensor_model_parallel_rank() == destination_spec.rank, \
            "Got different rank in destination spec and current rank"

        port = destination_spec.base_port + destination_spec.rank
        path = make_zmq_path("tcp", destination_spec.host, port)

        local_meta = self._nixl_wrapper.get_agent_metadata()
        with zmq_ctx(zmq.REQ, path) as sock:
            # Send query for metadata
            logger.debug("Sending handshake request to %s", destination_spec)
            sock.send(local_meta)

            metadata_bytes = sock.recv()
            
            # Get remote agent name and register it
            remote_agent_name = self._nixl_wrapper.add_remote_agent(
                metadata_bytes)
            
            # Store remote agent info
            self._remote_agents[destination_spec.get_id()] = remote_agent_name
            
            logger.debug("Successfully completed handshake with %s", 
                         destination_spec)


class NixlCPUReceiver:
    def __init__(
        self,
        allocator: RingBufferAllocator = None,
        nixl_page_size: int = 4096,
    ) -> None:
        self._buffer_size = allocator.get_size()
        self._buffer_ptr = allocator.get_buffer_ptr()
        self._nixl_page_size = nixl_page_size
        self._allocator = allocator

        assert self._allocator is not None, "Allocator is required"

        # Requests that are pending for allocation
        # uuid -> tuple[SourceSpec, peer name]
        self._pending_allocation: dict[str, tuple[SourceSpec, str]] = {}

        # Already allocated requests
        # uuid -> SourceSpec and uuid -> virtual address
        self._inflight_requests: dict[str, SourceSpec] = {}
        self._inflight_request_vaddr: dict[str, int] = {}

        # Finished requests
        # uuid -> tuple[SourceSpec, virtual address]
        self._finished_requests: dict[str, tuple[SourceSpec, int]] = {}

        # source zmq id -> peer name
        self._remote_agents: dict[str, str] = {}

        self._nixl_wrapper, self._reg_dlist, self._local_xfer_dlist = \
            init_nixl_agent(self._buffer_size, self._buffer_ptr, 
                            nixl_page_size)

        # Add handshake listener thread
        self._handshake_listener_t: Optional[threading.Thread] = None
        self._stop_listener = threading.Event()

        # Msg decoder
        self._msg_decoder = msgspec.msgpack.Decoder(NixlProtocolMsg)

    def _process_msgs(self):
        """Process the received messages from the NIXL agent."""
        notifs = self._nixl_wrapper.get_new_notifs()
        for remote_agent_name in notifs:
            for msg in notifs[remote_agent_name]:
                # Decode the messag
                obj = self._msg_decoder.decode(msg)
                if obj.msg_type == "REQMSG":
                    # Add the request to the pending allocation
                    self._pending_allocation[obj.req_uuid] = (obj.source_spec,
                                                             remote_agent_name)
                elif obj.msg_type == "FINISHMSG":
                    # Add the request to the finished requests
                    if obj.req_uuid in self._inflight_requests:
                        source_spec = self._inflight_requests.pop(obj.req_uuid)
                        vaddr = self._inflight_request_vaddr.pop(obj.req_uuid)
                        self._finished_requests[obj.req_uuid] = (source_spec, vaddr)
                    else:
                        logger.error("Request %s not found in inflight requests", 
                                     obj.req_uuid)
                else:
                    logger.error("Unexpected message type: %s", obj.msg_type)
                    continue

    def _process_allocation_requests(self):
        """Process the allocation requests and allocate the buffers."""
        allocated_requests = []
        for req_uuid, (source_spec, peer_name) in \
                self._pending_allocation.items():
            # Try to allocate the buffer
            vaddr, buffer = self._allocator.allocate(source_spec.get_size())
            if vaddr == -1:
                # No space available, skip all the requests

                # NOTE: an alternative is to try allocation for other requests
                # and then come back to this one, but this may create 
                # starvation
                break

            # Add the request to the inflight requests
            self._inflight_requests[req_uuid] = source_spec
            self._inflight_request_vaddr[req_uuid] = vaddr

            # Send back the ready message
            paddr = self._allocator.virtual_to_physical(vaddr)
            ready_msg = make_receive_ready_msg(req_uuid, paddr)
            self._nixl_wrapper.send_notif(peer_name, ready_msg)

            # Add the request to the allocated requests 
            allocated_requests.append(req_uuid)

        # Remove the allocated requests from the pending allocation
        for req_uuid in allocated_requests:
            del self._pending_allocation[req_uuid]

    def progress(self) -> None:
        """Process the received requests and the data
        """
        self._process_msgs()
        self._process_allocation_requests()

    def get_finished(self) -> list[tuple[SourceSpec, int]]:
        """Get the requests that finishes receiving.

        Returns:
            list[tuple[SourceSpec, int]]: A list of tuples containing the source 
                spec and the address.
        """
        pass

    def start_handshake_listener(self, host: str, base_port: int) -> None:
        """Start the background thread that listens for handshake requests.
        
        Args:
            host (str): Host address to listen on
            base_port (int): Base port number to listen on
        """
        ready_event = threading.Event()
        self._handshake_listener_t = threading.Thread(
            target=self._nixl_handshake_listener,
            args=(host, base_port, ready_event),
            daemon=True,
            name="nixl_cpu_handshake_listener"
        )
        self._handshake_listener_t.start()
        ready_event.wait()

    def _nixl_handshake_listener(
        self, 
        host: str,
        base_port: int,
        ready_event: threading.Event
    ) -> None:
        """Background thread that listens for and responds to handshake requests.
        
        Args:
            host (str): Host address to listen on
            base_port (int): Base port number to listen on
            ready_event (threading.Event): Event to signal when listener is ready
        """
        # Prepare metadata
        local_meta = self._nixl_wrapper.get_agent_metadata()
        
        # Setup ZMQ socket
        port = base_port + get_tensor_model_parallel_rank()
        path = make_zmq_path("tcp", host, port)
        logger.debug("Starting handshake listener on path: %s", path)
        
        with zmq_ctx(zmq.ROUTER, path) as sock:
            ready_event.set()
            
            while not self._stop_listener.is_set():
                try:
                    identity, _, msg = sock.recv_multipart(flags=zmq.NOBLOCK)
                    remote_agent_name = self._nixl_wrapper.add_remote_agent(
                        msg)
                    self._remote_agents[identity] = remote_agent_name
                    logger.debug("Successfully received handshake from %s", 
                                 identity)
                    # Send back the local metadata to the sender
                    sock.send_multipart([identity, b"", local_meta])
                    logger.debug("Sent local metadata back to %s", identity)
                except zmq.error.Again:
                    # No message available
                    time.sleep(0.1)
                except Exception as e:
                    logger.error("Error in handshake listener: %s", e)
                    break
            logger.debug("Stopping handshake listener")

    def stop_handshake_listener(self) -> None:
        """Stop the handshake listener thread."""
        if self._handshake_listener_t is not None:
            self._stop_listener.set()
            self._handshake_listener_t.join()
            self._handshake_listener_t = None


@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: Optional[zmq.Context] = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(ctx=ctx,
                              path=addr,
                              socket_type=socket_type,
                              bind=socket_type == zmq.ROUTER)
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)
