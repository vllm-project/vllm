// Composition model: the shm_broadcast MessageQueues that carry one TP=2
// inference step loop in the multiprocess executor.
//
//   EX  (EngineCore / MultiprocExecutor)  writes RPC, reads RESP1 (+ RESP2)
//   W1  (worker, TP rank 0, output rank)  reads RPC, writes TP and RESP1
//   W2  (worker, TP rank 1)               reads RPC and TP, writes RESP2
//
//   RPC   rpc_broadcast_mq           EX -> {W1, W2}
//   RESP1 worker_response_mq (W1)    W1 -> EX
//   RESP2 worker_response_mq (W2)    W2 -> EX   (ALL_REPLY only)
//   TP    TP GroupCoordinator.mq_broadcaster   W1 -> W2 (TP_BCAST only)
//
// Every queue is an M-slot ring with written/read flags. A write sets the
// slot and sends one notification per reader. The notification travels
// (inflight), lands in the reader's CONFLATE slot, and a parked reader
// consumes it on wake-up. A reader may always park when its slot is not
// ready, which covers every spin-grace policy. NOTIFY_DROP abstracts the
// channel: "none" is the fixed channel, "first" loses a reader's first
// notification (subscription not yet known to the PUB), "any" may lose any
// notification (SNDHWM=1 with zmq's send throttle). message_queue.qnt models
// when each happens.
//
// The executor keeps up to INFLIGHT RPCs outstanding (async scheduling) and
// gathers responses in order: it reads the first pending response queue, or
// waits on it when it is not ready, as get_response()'s dequeue() loop and
// FutureWrapper's wait_for_response() do.
module inference {
  const M: int
  const STEPS: int
  const INFLIGHT: int
  const ALL_REPLY: bool
  const TP_BCAST: bool
  const NOTIFY_DROP: str
  // Model the recheck/slice timeouts as transitions. Off for deadlock checks.
  const TIMEOUTS: bool

  pure val EX = 0
  pure val W1 = 1
  pure val W2 = 2
  pure val PROCS = Set(EX, W1, W2)
  pure val RPC = 0
  pure val RESP1 = 1
  pure val RESP2 = 2
  pure val TP = 3
  pure val QUEUES = Set(RPC, RESP1, RESP2, TP)
  pure val SLOTS = 0.to(M - 1)

  pure def readersOf(q: int): Set[int] =
    if (q == RPC) Set(W1, W2) else if (q == TP) Set(W2) else Set(EX)
  pure def respQ(w: int): int = if (w == W1) RESP1 else RESP2
  pure def responds(w: int): bool = ALL_REPLY or w == W1
  pure val RESP_QUEUES: Set[int] = if (ALL_REPLY) Set(RESP1, RESP2) else Set(RESP1)
  pure def minOf(s: Set[int]): int = s.fold(99, (a, b) => if (b < a) b else a)

  // rings
  var written: int -> (int -> bool)
  var seqOf: int -> (int -> int)
  var readFlag: int -> (int -> (int -> bool))
  var wSeq: int -> int
  var rSeq: int -> (int -> int)
  var ok: bool
  // notification channels
  var inflight: int -> (int -> int)
  var conflated: int -> (int -> bool)
  var drops: int
  // executor
  var exPc: str
  var exSent: int
  var exGathered: int
  var pending: Set[int]
  // workers
  var wPc: int -> str

  def ready(q: int, r: int): bool = {
    val s = rSeq.get(q).get(r) % M
    written.get(q).get(s) and not(readFlag.get(q).get(s).get(r))
  }

  def slotFree(q: int): bool = {
    val s = wSeq.get(q) % M
    not(written.get(q).get(s)) or readersOf(q).forall(r => readFlag.get(q).get(s).get(r))
  }

  // One expression rather than an `if` over sets of sets, which Apalache
  // encodes as a disabled transition.
  def dropChoices(q: int): Set[Set[int]] = {
    val lossy = NOTIFY_DROP == "any" or (NOTIFY_DROP == "first" and wSeq.get(q) == 0)
    readersOf(q).powerset().filter(lost => lossy or lost.size() == 0)
  }

  // The queue a parked process waits on.
  def parkQ(p: int): int =
    if (p == EX) minOf(pending) else if (wPc.get(p) == "tpPark") TP else RPC

  def parked(p: int): bool =
    if (p == EX) exPc == "park" else wPc.get(p) == "rpcPark" or wPc.get(p) == "tpPark"

  action ringSame = all {
    written' = written, seqOf' = seqOf, readFlag' = readFlag, wSeq' = wSeq,
    rSeq' = rSeq, ok' = ok,
  }
  action chanSame = all { inflight' = inflight, conflated' = conflated, drops' = drops }
  action exSame = all {
    exPc' = exPc, exSent' = exSent, exGathered' = exGathered, pending' = pending,
  }

  // Write the next slot of q and notify its readers, minus the lost ones.
  action doWrite(q: int, lost: Set[int]): bool = {
    val s = wSeq.get(q) % M
    all {
      written' = written.setBy(q, m => m.set(s, true)),
      seqOf' = seqOf.setBy(q, m => m.set(s, wSeq.get(q))),
      readFlag' = readFlag.setBy(q, m => m.set(s, PROCS.mapBy(_ => false))),
      wSeq' = wSeq.set(q, wSeq.get(q) + 1),
      rSeq' = rSeq,
      ok' = ok,
      inflight' = inflight.setBy(q, m => PROCS.mapBy(r =>
        if (readersOf(q).contains(r) and not(lost.contains(r))) m.get(r) + 1
        else m.get(r))),
      conflated' = conflated,
      drops' = drops + lost.size(),
    }
  }

  // Read the current slot of q as reader r. Does not consume notifications.
  action doRead(q: int, r: int): bool = {
    val s = rSeq.get(q).get(r) % M
    all {
      ok' = (ok and seqOf.get(q).get(s) == rSeq.get(q).get(r)),
      readFlag' = readFlag.setBy(q, m => m.setBy(s, f => f.set(r, true))),
      rSeq' = rSeq.setBy(q, m => m.set(r, m.get(r) + 1)),
      written' = written, seqOf' = seqOf, wSeq' = wSeq,
      chanSame,
    }
  }

  action init = all {
    written' = QUEUES.mapBy(_ => SLOTS.mapBy(_ => false)),
    seqOf' = QUEUES.mapBy(_ => SLOTS.mapBy(_ => -1)),
    readFlag' = QUEUES.mapBy(_ => SLOTS.mapBy(_ => PROCS.mapBy(_ => false))),
    wSeq' = QUEUES.mapBy(_ => 0),
    rSeq' = QUEUES.mapBy(_ => PROCS.mapBy(_ => 0)),
    ok' = true,
    inflight' = QUEUES.mapBy(_ => PROCS.mapBy(_ => 0)),
    conflated' = QUEUES.mapBy(_ => PROCS.mapBy(_ => false)),
    drops' = 0,
    exPc' = "ready",
    exSent' = 0,
    exGathered' = 0,
    pending' = Set(),
    wPc' = Set(W1, W2).mapBy(_ => "rpc"),
  }

  // ---- guards, shared by the actions and the progress invariant ----
  // The executor commits to an enqueue and stays inside acquire_write until
  // a slot frees; it cannot gather responses meanwhile.
  def canExBegin: bool = exPc == "ready" and exSent < STEPS
    and exSent - exGathered < INFLIGHT
  def canExSend: bool = exPc == "sending" and slotFree(RPC)
  def canExStart: bool = exPc == "ready" and exGathered < exSent
  // get_response() dequeues the response queues in order, so the executor
  // reads, or waits on, only the first one still pending.
  def canExRead(q: int): bool = exPc == "gather" and pending != Set()
    and q == minOf(pending) and ready(q, EX)
  def canExFinish: bool = exPc == "gather" and pending == Set()
  def canExPark: bool = exPc == "gather" and pending != Set()
    and not(ready(minOf(pending), EX))
  def canPoll(p: int): bool = parked(p) and conflated.get(parkQ(p)).get(p)
  def canRpcRead(w: int): bool = wPc.get(w) == "rpc" and ready(RPC, w)
  def canRpcPark(w: int): bool = wPc.get(w) == "rpc" and not(ready(RPC, w))
    and rSeq.get(RPC).get(w) < STEPS
  def canTpWrite: bool = wPc.get(W1) == "tp" and slotFree(TP)
  def canTpRead: bool = wPc.get(W2) == "tp" and ready(TP, W2)
  def canTpPark: bool = wPc.get(W2) == "tp" and not(ready(TP, W2))
  def canResp(w: int): bool = wPc.get(w) == "resp" and slotFree(respQ(w))
  def canDeliver(q: int, r: int): bool = inflight.get(q).get(r) > 0

  // ---- executor ----
  action exBegin: bool = all {
    canExBegin,
    exPc' = "sending",
    exSent' = exSent, exGathered' = exGathered, pending' = pending,
    ringSame, chanSame, wPc' = wPc,
  }

  // collective_rpc: enqueue on rpc_broadcast_mq.
  action exSend: bool = all {
    canExSend,
    nondet lost = dropChoices(RPC).oneOf()
    doWrite(RPC, lost),
    exPc' = "ready", exSent' = exSent + 1, exGathered' = exGathered,
    pending' = pending,
    wPc' = wPc,
  }

  // FutureWrapper.result() / _gather_responses for the oldest RPC.
  action exStart: bool = all {
    canExStart,
    exPc' = "gather", pending' = RESP_QUEUES,
    exSent' = exSent, exGathered' = exGathered,
    ringSame, chanSame, wPc' = wPc,
  }

  action exRead(q: int): bool = all {
    canExRead(q),
    doRead(q, EX),
    pending' = pending.exclude(Set(q)),
    exPc' = exPc, exSent' = exSent, exGathered' = exGathered,
    wPc' = wPc,
  }

  action exFinish: bool = all {
    canExFinish,
    exPc' = "ready", exGathered' = exGathered + 1,
    exSent' = exSent, pending' = pending,
    ringSame, chanSame, wPc' = wPc,
  }

  // dequeue() / spin_until_readable() waits on the first pending queue.
  action exPark: bool = all {
    canExPark,
    exPc' = "park",
    exSent' = exSent, exGathered' = exGathered, pending' = pending,
    ringSame, chanSame, wPc' = wPc,
  }

  // ---- workers ----
  action rpcRead(w: int): bool = all {
    canRpcRead(w),
    doRead(RPC, w),
    wPc' = wPc.set(w,
      if (TP_BCAST) "tp" else if (responds(w)) "resp" else "rpc"),
    exSame,
  }

  action rpcPark(w: int): bool = all {
    canRpcPark(w),
    wPc' = wPc.set(w, "rpcPark"),
    ringSame, chanSame, exSame,
  }

  // TP rank 0 broadcast_object() inside the step.
  action tpWrite: bool = all {
    canTpWrite,
    nondet lost = dropChoices(TP).oneOf()
    doWrite(TP, lost),
    wPc' = wPc.set(W1, "resp"),
    exSame,
  }

  action tpRead: bool = all {
    canTpRead,
    doRead(TP, W2),
    wPc' = wPc.set(W2, if (responds(W2)) "resp" else "rpc"),
    exSame,
  }

  action tpPark: bool = all {
    canTpPark,
    wPc' = wPc.set(W2, "tpPark"),
    ringSame, chanSame, exSame,
  }

  // enqueue on the worker's response queue.
  action respWrite(w: int): bool = all {
    canResp(w),
    nondet lost = dropChoices(respQ(w)).oneOf()
    doWrite(respQ(w), lost),
    wPc' = wPc.set(w, "rpc"),
    exSame,
  }

  // ---- channel and wake-ups ----
  action deliver(q: int, r: int): bool = all {
    canDeliver(q, r),
    inflight' = inflight.setBy(q, m => m.set(r, m.get(r) - 1)),
    conflated' = conflated.setBy(q, m => m.set(r, true)),
    drops' = drops,
    ringSame, exSame, wPc' = wPc,
  }

  // poll() returns on the notification and recv() consumes it.
  action poll(p: int): bool = {
    val q = parkQ(p)
    all {
      canPoll(p),
      conflated' = conflated.setBy(q, m => m.set(p, false)),
      inflight' = inflight, drops' = drops,
      ringSame,
      if (p == EX) all {
        exPc' = "gather", exSent' = exSent, exGathered' = exGathered,
        pending' = pending, wPc' = wPc,
      } else all {
        wPc' = wPc.set(p, if (wPc.get(p) == "tpPark") "tp" else "rpc"),
        exSame,
      },
    }
  }

  // SHM_READER_RECHECK_INTERVAL_MS (5 s), or the executor's spin deadline.
  action timeout(p: int): bool = all {
    TIMEOUTS,
    parked(p),
    ringSame, chanSame,
    if (p == EX) all {
      exPc' = "gather", exSent' = exSent, exGathered' = exGathered,
      pending' = pending, wPc' = wPc,
    } else all {
      wPc' = wPc.set(p, if (wPc.get(p) == "tpPark") "tp" else "rpc"),
      exSame,
    },
  }

  def allDone: bool = exGathered == STEPS

  // Stutter when finished or when nothing can move, so a stuck run is
  // reported by the `progress` invariant, not as a checker deadlock that
  // would mask every other invariant's result.
  action done = all {
    allDone or not(progress),
    ringSame, chanSame, exSame, wPc' = wPc,
  }

  action step = any {
    exBegin, exSend, exStart, exFinish, exPark, tpWrite, tpRead, tpPark, done,
    { nondet q = RESP_QUEUES.oneOf() exRead(q) },
    { nondet w = Set(W1, W2).oneOf() any { rpcRead(w), rpcPark(w), respWrite(w) } },
    { nondet p = PROCS.oneOf() any { poll(p), timeout(p) } },
    { nondet q = QUEUES.oneOf() nondet r = PROCS.oneOf() deliver(q, r) },
  }

  // Every parked process whose queue is ready has a wake-up on the way.
  val noLostWakeup = PROCS.forall(p =>
    (parked(p) and ready(parkQ(p), p))
      implies (conflated.get(parkQ(p)).get(p) or inflight.get(parkQ(p)).get(p) > 0))

  // Every queue delivers in order and never overwrites an unread slot.
  val ringSafe = ok

  // Without timeouts, the step loop never stops before the last response:
  // some process or I/O thread can always move.
  val progress = allDone
    or canExBegin or canExSend or canExStart or canExFinish or canExPark
    or RESP_QUEUES.exists(q => canExRead(q))
    or PROCS.exists(p => canPoll(p))
    or Set(W1, W2).exists(w => canRpcRead(w) or canRpcPark(w) or canResp(w))
    or canTpWrite or canTpRead or canTpPark
    or QUEUES.exists(q => PROCS.exists(r => canDeliver(q, r)))
}

// The fixed channel, decode path (only the output rank replies).
module fixedDecode {
  import inference(M = 2, STEPS = 3, INFLIGHT = 2, ALL_REPLY = false,
    TP_BCAST = true, NOTIFY_DROP = "none", TIMEOUTS = false).*
}

// The fixed channel, collective_rpc with every rank replying.
module fixedAllReply {
  import inference(M = 2, STEPS = 3, INFLIGHT = 2, ALL_REPLY = true,
    TP_BCAST = true, NOTIFY_DROP = "none", TIMEOUTS = false).*
}

// SNDHWM=1: any notification may be lost (PR #58917).
module hwm1Decode {
  import inference(M = 2, STEPS = 3, INFLIGHT = 2, ALL_REPLY = false,
    TP_BCAST = true, NOTIFY_DROP = "any", TIMEOUTS = false).*
}

// Notify subscription not covered by wait_until_ready.
module raceDecode {
  import inference(M = 2, STEPS = 3, INFLIGHT = 2, ALL_REPLY = false,
    TP_BCAST = true, NOTIFY_DROP = "first", TIMEOUTS = false).*
}

// Ring backpressure: one-slot rings, outstanding RPCs at the bound.
module backpressureAtBound {
  import inference(M = 1, STEPS = 3, INFLIGHT = 3, ALL_REPLY = true,
    TP_BCAST = true, NOTIFY_DROP = "none", TIMEOUTS = false).*
}

// Ring backpressure: one more outstanding RPC than the rings can hold.
module backpressureOver {
  import inference(M = 1, STEPS = 4, INFLIGHT = 4, ALL_REPLY = true,
    TP_BCAST = true, NOTIFY_DROP = "none", TIMEOUTS = false).*
}
