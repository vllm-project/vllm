"""Isolation spike for the ONE unknown gating overlapped prefetch (#1):

Does torch's CUDA-graph capture support a side-stream fork/join
(`copy_stream.wait_stream(capture_stream)` ... `capture_stream.wait_stream(copy_stream)`)?

The original HiSparse host-backup hit "dependency created on uncaptured work in
another stream" and was forced inline. Hypothesis: that was an *incomplete* fork,
and the documented fork/join pattern captures cleanly. This tests exactly that,
using only torch (no custom op) so it runs in any env. The gather is a
device-side scatter (index_copy_) issued on the copy stream — representative of
the overlap's copy-stream work; the H2D-from-pinned detail inside the real
gather_plan kernel is orthogonal (standard cudaMemcpyAsync, graph-capturable, and
already works eagerly).
"""

import torch

DEV = "cuda"


def main():
    n, w = 8, 64
    src = torch.arange(n * w, dtype=torch.float32, device=DEV).view(n, w)
    dst_idx = torch.arange(n, dtype=torch.long, device=DEV)
    hot = torch.zeros(n, w, device=DEV)
    xin = torch.ones(256, 256, device=DEV)
    copy = torch.cuda.Stream()

    def gather():
        hot.index_copy_(0, dst_idx, src)

    # Eager reference.
    gather()
    torch.cuda.synchronize()
    ref = hot.clone()

    # Warmup (torch graph-capture convention) incl. the fork/join.
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        for _ in range(3):
            _ = xin @ xin
            copy.wait_stream(warm)
            with torch.cuda.stream(copy):
                gather()
            warm.wait_stream(copy)
    torch.cuda.current_stream().wait_stream(warm)
    torch.cuda.synchronize()

    hot.zero_()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g):
            main_s = torch.cuda.current_stream()
            _ = xin @ xin                 # compute on the capture stream
            copy.wait_stream(main_s)      # FORK the copy stream into the capture
            with torch.cuda.stream(copy):
                gather()                  # copy-stream work, captured as a branch
            main_s.wait_stream(copy)      # JOIN before the consumer reads hot
            _ = hot.sum()                 # consumer on the capture stream
    except Exception as e:  # noqa: BLE001
        print("CAPTURE_FAILED", repr(e))
        return

    hot.zero_()
    g.replay()
    torch.cuda.synchronize()
    if torch.allclose(hot, ref):
        print("CAPTURE_OK: side-stream fork/join captured + replayed correctly")
    else:
        print("CAPTURE_REPLAY_MISMATCH", (hot - ref).abs().max().item())


if __name__ == "__main__":
    main()
