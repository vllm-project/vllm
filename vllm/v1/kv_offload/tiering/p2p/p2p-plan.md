# Generic p2p plan

## Sequence Diagram

```mermaid

sequenceDiagram
    participant Cons_TM as Consumer TieringManager
    participant Cons_P2P as Consumer P2P Tier
    participant Prod_P2P as Producer P2P Tier
    participant Prod_TM as Producer TieringManager

    Cons_P2P->>Cons_P2P: Open listener thread
    Prod_P2P->>Prod_P2P: Open listener thread

    Note over Cons_TM,Prod_TM: ── Producer has previously cached blocks for this prompt ──

    Cons_TM->>Cons_P2P: submit_load(job_metadata, kv_transfer_params)
    Note right of Cons_P2P: kv_transfer_params:<br/>kv_request_id, do_p2p_fetch=true,<br/>remote_host, remote_port

    Cons_P2P->>Prod_P2P: 𝗖𝗧𝗥𝗟: lookup(kv_request_id, block_hashes)

    Note right of Prod_P2P: Match block_hashes against<br/>local CPU cache

    Prod_P2P--)Cons_P2P: 𝗖𝗧𝗥𝗟: lookup_resp(kv_request_id, hit_indexes)

    Cons_P2P->>Prod_P2P: 𝗖𝗧𝗥𝗟: fetch(kv_request_id, block_hashes, src_block_ids)

    Prod_P2P-)Cons_P2P: 𝗗𝗔𝗧𝗔: NIXL.Transfer(WRITE, src_descs, dst_descs) <br/> for the subset of hashes that hit
    Prod_P2P-->>Cons_P2P:  𝗖𝗧𝗥𝗟: TransferDone(kv_request_id, served_indexes)

    Cons_TM->>Cons_P2P: get_finished()
    Note right of Cons_P2P: Hits → loaded into GPU as a normal cache hit<br/>Misses → recomputed by the engine
```
