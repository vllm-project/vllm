# Pipeline activation send buffer

Set `--additional-config '{"pp_send_buffer_size":2}'` to enable bounded activation snapshots with Model Runner V2 and a fixed pipeline partition. The default is zero.

Each non-last pipeline worker reserves independent output storage for each slot before KV memory budgeting. After forward it copies intermediate tensors into a slot and posts sends on a communication stream ordered after the copy. Later forwards can reuse model output storage. A slot remains alive until its sends complete. A full ring waits for its oldest send.

This feature retains the fixed layer partition. Chunked prefill and speculative decoding retain their existing model and platform requirements. Output length and graph execution follow the existing serving configuration. Their combinations require device end-to-end validation.

Slots are sized for `max_num_batched_tokens`. Additional slots reduce available KV memory and add device copies. Sleep mode and external launcher are rejected. Ascend sequence/context parallel configurations are rejected until their communication layout is adapted.

For an Ascend deployment, install both matching `codex/pp-send-buffer-only` branches of vLLM and vLLM Ascend. Compare buffer sizes zero and two using identical weights, request packing, graph and speculative settings. This isolates buffering from layer-boundary movement. No device performance result is claimed by this extraction.
