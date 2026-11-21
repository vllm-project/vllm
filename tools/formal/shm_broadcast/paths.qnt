// Model of the MessageQueue paths outside the ring wake-up protocol:
// overflow payloads on local_socket, remote readers on remote_socket, and
// cancellation through MessageQueue.shutdown().
//
// One writer, one local reader and one remote reader. The ring has M slots.
// Each message may overflow: the writer marks its slot, sends the payload on
// the local data socket, publishes the slot, then pings (enqueue sends the
// payload inside acquire_write, so a send timeout leaves the slot
// unpublished). Every message then goes to the remote
// reader on the remote data socket. The local notification channel is
// lossless here (message_queue.qnt checks it).
//
// Data sockets are zmq XPUBs with a high water mark. DATA_MODE "drop" is the
// default XPUB: a send to a full pipe is silently dropped. "block" is
// XPUB_NODROP: a blocking send waits until the pipe has room. The model uses
// the true pipe occupancy; zmq's stale view only drops earlier.
//
// Shutdown stops the writer process and sends each reader its cancel ping.
// The local reader's ring wait polls the cancel socket. CANCELLABLE_RECV
// makes the overflow-payload and remote-reader socket waits poll it too.
module paths {
  const M: int
  const MAXMSG: int
  const HWM: int
  const DATA_MODE: str
  const CANCELLABLE_RECV: bool

  pure val SLOTS = 0.to(M - 1)

  // ring (one local reader)
  var written: int -> bool
  var seqOf: int -> int
  var ovf: int -> bool
  var readFlag: int -> bool
  // writer
  var wSeq: int
  var wPc: str
  var stopped: bool
  var drops: int
  // channels
  var dataQ: List[int]
  var remoteQ: List[int]
  var notifyInflight: int
  var conflated: bool
  // local reader
  var lPc: str
  var lSeq: int
  // remote reader
  var rmPc: str
  var rmSeq: int
  // shutdown
  var shutting: bool
  var cancelL: bool
  var cancelR: bool
  var ok: bool

  def ready: bool = {
    val s = lSeq % M
    written.get(s) and not(readFlag.get(s))
  }
  def slotFree: bool = {
    val s = wSeq % M
    not(written.get(s)) or readFlag.get(s)
  }
  pure def hasRoom(q: List[int]): bool = HWM == 0 or q.length() < HWM
  def wFinished: bool = wSeq == MAXMSG and wPc == "acquire"

  action ringSame = all {
    written' = written, seqOf' = seqOf, ovf' = ovf, readFlag' = readFlag,
  }
  action writerSame = all {
    wSeq' = wSeq, wPc' = wPc, stopped' = stopped, drops' = drops,
  }
  action chanSame = all {
    dataQ' = dataQ, remoteQ' = remoteQ, notifyInflight' = notifyInflight,
    conflated' = conflated,
  }
  action localSame = all { lPc' = lPc, lSeq' = lSeq }
  action remoteSame = all { rmPc' = rmPc, rmSeq' = rmSeq }
  action shutSame = all { shutting' = shutting, cancelL' = cancelL, cancelR' = cancelR }

  action init = all {
    written' = SLOTS.mapBy(_ => false),
    seqOf' = SLOTS.mapBy(_ => -1),
    ovf' = SLOTS.mapBy(_ => false),
    readFlag' = SLOTS.mapBy(_ => false),
    wSeq' = 0, wPc' = "acquire", stopped' = false, drops' = 0,
    dataQ' = List(), remoteQ' = List(), notifyInflight' = 0, conflated' = false,
    lPc' = "check", lSeq' = 0,
    rmPc' = "recv", rmSeq' = 0,
    shutting' = false, cancelL' = false, cancelR' = false,
    ok' = true,
  }

  // ---- guards ----
  def canAcquire: bool = not(stopped) and wPc == "acquire" and wSeq < MAXMSG and slotFree
  def canData: bool = not(stopped) and wPc == "data"
    and (DATA_MODE == "drop" or hasRoom(dataQ))
  def canNotify: bool = not(stopped) and wPc == "notify"
  def canRemote: bool = not(stopped) and wPc == "remote"
    and (DATA_MODE == "drop" or hasRoom(remoteQ))
  def canDeliver: bool = notifyInflight > 0
  def canLRead: bool = lPc == "check" and ready
  def canLPark: bool = lPc == "check" and not(ready)
  def canLExit: bool = lPc == "check" and shutting
  def canLPoll: bool = lPc == "park" and (conflated or cancelL)
  def canLPayload: bool = lPc == "payload" and dataQ.length() > 0
  def canLPayloadCancel: bool = CANCELLABLE_RECV and lPc == "payload" and cancelL
  def canRmRecv: bool = rmPc == "recv" and remoteQ.length() > 0
  def canRmCancel: bool = CANCELLABLE_RECV and rmPc == "recv" and cancelR

  // ---- writer: MessageQueue.enqueue ----
  // acquire_write: an inline message is written and published; an overflow
  // message is marked and stays unpublished until its payload is sent.
  action acquire(o: bool): bool = {
    val s = wSeq % M
    all {
      canAcquire,
      written' = written.set(s, not(o)),
      seqOf' = seqOf.set(s, wSeq),
      ovf' = ovf.set(s, o),
      readFlag' = readFlag.set(s, false),
      wSeq' = wSeq + 1,
      wPc' = if (o) "data" else "notify",
      stopped' = stopped, drops' = drops,
      chanSame, localSame, remoteSame, shutSame, ok' = ok,
    }
  }

  // local_socket.send_multipart(payload) inside acquire_write, then the
  // overflow slot is published.
  action dataSend: bool = {
    val s = (wSeq - 1) % M
    all {
      canData,
      if (hasRoom(dataQ)) all { dataQ' = dataQ.append(wSeq - 1), drops' = drops }
      else all { dataQ' = dataQ, drops' = drops + 1 },
      written' = written.set(s, true),
      seqOf' = seqOf, ovf' = ovf, readFlag' = readFlag,
      wPc' = "notify",
      wSeq' = wSeq, stopped' = stopped,
      remoteQ' = remoteQ, notifyInflight' = notifyInflight, conflated' = conflated,
      localSame, remoteSame, shutSame, ok' = ok,
    }
  }

  action notify: bool = all {
    canNotify,
    notifyInflight' = notifyInflight + 1,
    wPc' = "remote",
    wSeq' = wSeq, stopped' = stopped, drops' = drops,
    dataQ' = dataQ, remoteQ' = remoteQ, conflated' = conflated,
    ringSame, localSame, remoteSame, shutSame, ok' = ok,
  }

  // remote_socket.send_multipart(message).
  action remoteSend: bool = all {
    canRemote,
    if (hasRoom(remoteQ)) all { remoteQ' = remoteQ.append(wSeq - 1), drops' = drops }
    else all { remoteQ' = remoteQ, drops' = drops + 1 },
    wPc' = "acquire",
    wSeq' = wSeq, stopped' = stopped,
    dataQ' = dataQ, notifyInflight' = notifyInflight, conflated' = conflated,
    ringSame, localSame, remoteSame, shutSame, ok' = ok,
  }

  action deliver: bool = all {
    canDeliver,
    notifyInflight' = notifyInflight - 1,
    conflated' = true,
    dataQ' = dataQ, remoteQ' = remoteQ,
    ringSame, writerSame, localSame, remoteSame, shutSame, ok' = ok,
  }

  // ---- local reader: MessageQueue.dequeue ----
  // acquire_read releases the slot before an overflow payload is received.
  action lRead: bool = {
    val s = lSeq % M
    all {
      canLRead,
      readFlag' = readFlag.set(s, true),
      lPc' = if (ovf.get(s)) "payload" else "check",
      lSeq' = if (ovf.get(s)) lSeq else lSeq + 1,
      written' = written, seqOf' = seqOf, ovf' = ovf,
      writerSame, chanSame, remoteSame, shutSame,
      ok' = (ok and seqOf.get(s) == lSeq),
    }
  }

  action lPark: bool = all {
    canLPark,
    lPc' = "park", lSeq' = lSeq,
    ringSame, writerSame, chanSame, remoteSame, shutSame, ok' = ok,
  }

  // The read loop checks shutting_down after every wait, spinning or not.
  action lExit: bool = all {
    canLExit,
    lPc' = "exited", lSeq' = lSeq,
    ringSame, writerSame, chanSame, remoteSame, shutSame, ok' = ok,
  }

  // poll() returns on a notification or the (never consumed) cancel ping.
  action lPoll: bool = all {
    canLPoll,
    conflated' = false,
    lPc' = "check", lSeq' = lSeq,
    dataQ' = dataQ, remoteQ' = remoteQ, notifyInflight' = notifyInflight,
    ringSame, writerSame, remoteSame, shutSame, ok' = ok,
  }

  // MessageQueue._recv(local_socket) for the overflow payload.
  action lPayload: bool = all {
    canLPayload,
    ok' = (ok and dataQ.head() == lSeq),
    dataQ' = dataQ.tail(),
    lPc' = "check", lSeq' = lSeq + 1,
    remoteQ' = remoteQ, notifyInflight' = notifyInflight, conflated' = conflated,
    ringSame, writerSame, remoteSame, shutSame,
  }

  action lPayloadCancel: bool = all {
    canLPayloadCancel,
    lPc' = "exited", lSeq' = lSeq,
    ringSame, writerSame, chanSame, remoteSame, shutSame, ok' = ok,
  }

  // ---- remote reader: MessageQueue._recv(remote_socket) ----
  action rmRecv: bool = all {
    canRmRecv,
    ok' = (ok and remoteQ.head() == rmSeq),
    remoteQ' = remoteQ.tail(),
    rmSeq' = rmSeq + 1, rmPc' = rmPc,
    dataQ' = dataQ, notifyInflight' = notifyInflight, conflated' = conflated,
    ringSame, writerSame, localSame, shutSame,
  }

  action rmCancel: bool = all {
    canRmCancel,
    rmPc' = "exited", rmSeq' = rmSeq,
    ringSame, writerSame, chanSame, localSame, shutSame, ok' = ok,
  }

  // ---- shutdown: the writer stops; each reader gets its cancel ping ----
  action shutdown: bool = all {
    not(shutting),
    shutting' = true, cancelL' = true, cancelR' = true,
    stopped' = true,
    wSeq' = wSeq, wPc' = wPc, drops' = drops,
    ringSame, chanSame, localSame, remoteSame, ok' = ok,
  }

  def enabledAny: bool = canAcquire or canData or canNotify or canRemote
    or canDeliver or canLRead or canLPark or canLExit or canLPoll
    or canLPayload or canLPayloadCancel or canRmRecv or canRmCancel

  def finished: bool =
    if (shutting) lPc == "exited" and rmPc == "exited"
    else wFinished and lSeq == MAXMSG and rmSeq == MAXMSG

  action done = all {
    finished or not(enabledAny),
    ringSame, writerSame, chanSame, localSame, remoteSame, shutSame, ok' = ok,
  }

  action step = any {
    acquire(true), acquire(false), dataSend, notify, remoteSend, deliver,
    lRead, lPark, lExit, lPoll, lPayload, lPayloadCancel,
    rmRecv, rmCancel, shutdown, done,
  }

  // Every reader sees every message, in order.
  val inOrder = ok

  // No message is ever dropped.
  val noLoss = drops == 0

  // Without timeouts, the system never stops short: before shutdown every
  // message reaches both readers, and after shutdown both readers exit.
  val progress = finished or enabledAny
}

// Current code: default XPUB data sockets, uncancellable socket waits.
module pathsCurrent {
  import paths(M = 2, MAXMSG = 4, HWM = 2, DATA_MODE = "drop",
    CANCELLABLE_RECV = false).*
}

// XPUB_NODROP only.
module pathsNoDrop {
  import paths(M = 2, MAXMSG = 4, HWM = 2, DATA_MODE = "block",
    CANCELLABLE_RECV = false).*
}

// Cancellable socket waits only.
module pathsCancel {
  import paths(M = 2, MAXMSG = 4, HWM = 2, DATA_MODE = "drop",
    CANCELLABLE_RECV = true).*
}

// Both: XPUB_NODROP data sockets and cancellable socket waits.
module pathsFixed {
  import paths(M = 2, MAXMSG = 4, HWM = 2, DATA_MODE = "block",
    CANCELLABLE_RECV = true).*
}
