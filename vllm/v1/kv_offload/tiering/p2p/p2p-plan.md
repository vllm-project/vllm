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

    Note over Cons_TM,Cons_P2P: Step N — aggregate per-block lookups

    Cons_TM->>Cons_P2P: lookup(key, req_context) [per block]
    Note right of Cons_P2P: kv_transfer_params:<br/>kv_request_id, do_p2p_fetch=true,<br/>remote_host, remote_port<br/>Returns None; entry registered<br/>at kv_request_id + block_hash

    Cons_TM->>Cons_P2P: on_schedule_end()
    Cons_P2P->>Prod_P2P: 𝗖𝗧𝗥𝗟: LookupMsg(kv_request_id, [block_hashes…])
    Note right of Cons_P2P: One LookupMsg per peer + kv_request_id<br/>per step that has unsent entries

    Note right of Prod_P2P: Match block_hashes against<br/>local CPU cache (hit/miss per hash)

    Prod_P2P--)Cons_P2P: 𝗖𝗧𝗥𝗟: LookupRespMsg(kv_request_id,<br/>block_hashes=[…], hits=[True/False…])
    Note right of Prod_P2P: Pairs are self-describing —<br/>producer may split or coalesce<br/>responses across LookupMsgs for<br/>the same kv_request_id

    Note over Cons_TM,Cons_P2P: Step N+k — resolve

    Cons_TM->>Cons_P2P: lookup(key, req_context) [retry]
    Note right of Cons_P2P: Hit → True (entry deleted)<br/>Miss → False (entry deleted)<br/>Still in-flight → None (try again)

    Cons_TM->>Cons_P2P: submit_load(job_metadata) [only hit blocks]
    Note right of Cons_P2P: block_ids are pre-allocated CPU slots for hits only

    Cons_P2P->>Prod_P2P: 𝗖𝗧𝗥𝗟: fetch(kv_request_id, block_hashes, dst_block_indexes)

    Prod_P2P-)Cons_P2P: 𝗗𝗔𝗧𝗔: NIXL.Transfer(WRITE, src_descs, dst_descs) <br/> for the confirmed hit blocks
    Prod_P2P-->>Cons_P2P:  𝗖𝗧𝗥𝗟: TransferDone(kv_request_id, success)

    Cons_TM->>Cons_P2P: get_finished()
    Note right of Cons_P2P: Hits → loaded into GPU as a normal cache hit<br/>Misses → recomputed by the engine
```
