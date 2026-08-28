# Migrating from Legacy JetStream API to `jetstream` Package

This guide helps you migrate from the legacy JetStream API in the `nats` package
(`nats.JetStreamContext`) to the new `jetstream` package
(`github.com/nats-io/nats.go/jetstream`).

- [Why Migrate?](#why-migrate)
- [Getting Started](#getting-started)
- [Stream Management](#stream-management)
- [Consumer Management](#consumer-management)
- [Publishing](#publishing)
- [Consuming Messages](#consuming-messages)
  - [Replacing js.Subscribe()](#replacing-jssubscribe)
  - [Replacing js.PullSubscribe()](#replacing-jspullsubscribe)
  - [Ordered Consumers](#ordered-consumers)
  - [Push Consumers](#push-consumers)
  - [Subscription Options Mapping](#subscription-options-mapping)
  - [Error Handling in Consume/Messages](#error-handling-in-consumemessages)
- [Message Acknowledgement](#message-acknowledgement)
- [KeyValue Store](#keyvalue-store)
- [Object Store](#object-store)

## Why Migrate?

The legacy JetStream API (`nats.JetStreamContext`) is deprecated. The `jetstream`
package provides a cleaner, more predictable API with several key improvements:

- **Explicit resource management.** Streams and consumers are created and managed
  explicitly. The legacy `js.Subscribe()` implicitly created consumers behind
  the scenes, leading to surprising behavior.

- **Pull consumers as the default.** Pull consumers with `Consume()` and
  `Messages()` provide the same continuous message delivery as the legacy push-based
  `Subscribe()`, but with better flow control and no slow consumer issues.

- **`context.Context` throughout.** All API calls accept `context.Context` for
  timeout and cancellation, replacing the mix of `MaxWait`, `AckWait`, and
  `Context()` options.

- **Clear interface separation.** Instead of one large `JetStreamContext` interface,
  functionality is split across focused interfaces: `JetStream`, `Stream` and
  `Consumer`.

## Getting Started

The core NATS connection remains unchanged. Only the JetStream initialization
differs:

```go
import (
    "github.com/nats-io/nats.go"
    "github.com/nats-io/nats.go/jetstream"
)

nc, _ := nats.Connect(nats.DefaultURL)
```

**Legacy:**

```go
js, _ := nc.JetStream()

// With domain
js, _ := nc.JetStream(nats.Domain("hub"))

// With custom API prefix
js, _ := nc.JetStream(nats.APIPrefix("myprefix"))
```

**New:**

```go
js, _ := jetstream.New(nc)

// With domain
js, _ := jetstream.NewWithDomain(nc, "hub")

// With custom API prefix
js, _ := jetstream.NewWithAPIPrefix(nc, "myprefix")
```

### Initialization Options

| Legacy                            | New                                        |
|-----------------------------------|--------------------------------------------|
| `nats.Domain(domain)`             | `jetstream.NewWithDomain(nc, domain)`      |
| `nats.APIPrefix(prefix)`          | `jetstream.NewWithAPIPrefix(nc, prefix)`   |
| `nats.PublishAsyncMaxPending(n)`  | `jetstream.WithPublishAsyncMaxPending(n)`  |
| `nats.PublishAsyncErrHandler(cb)` | `jetstream.WithPublishAsyncErrHandler(cb)` |

## Stream Management

`StreamConfig` is essentially the same struct — it just lives in the `jetstream`
package now. The new API takes `StreamConfig` by value (not pointer) and
management methods return a `Stream` handle instead of `*StreamInfo`.

| Legacy                          | New                                 | Notes                                                             |
|---------------------------------|-------------------------------------|-------------------------------------------------------------------|
| `js.AddStream(cfg)`             | `js.CreateStream(ctx, cfg)`         | Also: `CreateOrUpdateStream()`                                    |
| `js.UpdateStream(cfg)`          | `js.UpdateStream(ctx, cfg)`         |                                                                   |
| `js.DeleteStream(name)`         | `js.DeleteStream(ctx, name)`        |                                                                   |
| `js.StreamInfo(name)`           | `s.Info(ctx)` / `s.CachedInfo()`    | Get stream handle first via `js.Stream(ctx, name)`                |
| `js.PurgeStream(name, opts...)` | `s.Purge(ctx, opts...)`             | Options: `WithPurgeSubject`, `WithPurgeSequence`, `WithPurgeKeep` |
| `js.GetMsg(name, seq)`          | `s.GetMsg(ctx, seq)`                |                                                                   |
| `js.GetLastMsg(name, subj)`     | `s.GetLastMsgForSubject(ctx, subj)` |                                                                   |
| `js.DeleteMsg(name, seq)`       | `s.DeleteMsg(ctx, seq)`             | Also: `s.SecureDeleteMsg()`                                       |
| `js.Streams()`                  | `js.ListStreams(ctx)`               | Returns lister with `.Info()` channel and `.Err()`                |
| `js.StreamNames()`              | `js.StreamNames(ctx)`               | Returns lister with `.Name()` channel and `.Err()`                |

The key architectural difference is that stream-specific operations (purge, get/delete
messages) now live on the `Stream` interface instead of the top-level context. Get
a stream handle first, then operate on it:

```go
s, _ := js.Stream(ctx, "ORDERS")
s.Purge(ctx)
msg, _ := s.GetMsg(ctx, 100)
```

## Consumer Management

The biggest conceptual change: in the legacy API, `js.Subscribe()` would
implicitly create consumers. In the new API, consumer creation is always explicit
and separate from message consumption.

| Legacy                                   | New                                    | Notes                                                          |
|------------------------------------------|----------------------------------------|----------------------------------------------------------------|
| `js.AddConsumer(stream, cfg)`            | `js.CreateConsumer(ctx, stream, cfg)`  | Also: `CreateOrUpdateConsumer()`, `UpdateConsumer()`           |
| `js.Subscribe(subj, handler)` (implicit) | No equivalent                          | Must create consumer explicitly first                          |
| `js.ConsumerInfo(stream, name)`          | `cons.Info(ctx)` / `cons.CachedInfo()` | Get consumer handle first via `js.Consumer(ctx, stream, name)` |
| `js.DeleteConsumer(stream, name)`        | `js.DeleteConsumer(ctx, stream, name)` |                                                                |
| `js.Consumers(stream)`                   | `s.ListConsumers(ctx)`                 | Returns lister with `.Info()` channel and `.Err()`             |
| `js.ConsumerNames(stream)`               | `s.ConsumerNames(ctx)`                 | Returns lister with `.Name()` channel and `.Err()`             |

Consumer management is available at two levels:

- On `JetStream` — requires stream name as parameter (e.g. `js.CreateConsumer(ctx, "ORDERS", cfg)`), bypassing the need to fetch a stream
- On `Stream` — no stream name needed (e.g. `s.CreateConsumer(ctx, cfg)`)

The new API provides three creation methods:

- `CreateConsumer` — fails if the consumer already exists with different config
- `UpdateConsumer` - fails if the consumer does not exist
- `CreateOrUpdateConsumer` — creates or updates as needed

**Additional notes on consumer behavior:**

- The default ack policy changed between the APIs. In the legacy API,
  `AddConsumer()` defaulted to `AckNone`. In the new API, the default is
  `AckExplicit`.

- In the legacy API, `sub.Unsubscribe()` on an implicitly created
  consumer would automatically delete that consumer on the server. The new API
  does not perform any automatic cleanup - consumers must be deleted explicitly
  via `DeleteConsumer()`, or via `InactiveThreshold` on the consumer
  config to let the server remove it automatically after a period of inactivity.

Push consumers use separate methods: `CreatePushConsumer`, `CreateOrUpdatePushConsumer`,
`UpdatePushConsumer`, and `PushConsumer` (for getting a handle).

```go
s, _ := js.Stream(ctx, "ORDERS")
cons, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:   "processor",
})
```

## Publishing

Publishing is largely the same, with the addition of `context.Context` for
synchronous operations.

### Synchronous Publish

**Legacy:**

```go
ack, _ := js.Publish("ORDERS.new", []byte("hello"))
ack, _ = js.PublishMsg(&nats.Msg{
    Subject: "ORDERS.new",
    Data:    []byte("hello"),
})
```

**New:**

```go
ack, _ := js.Publish(ctx, "ORDERS.new", []byte("hello"))
ack, _ = js.PublishMsg(ctx, &nats.Msg{
    Subject: "ORDERS.new",
    Data:    []byte("hello"),
})
```

### Async Publish

**Legacy:**

```go
ackF, _ := js.PublishAsync("ORDERS.new", []byte("hello"))

select {
case ack := <-ackF.Ok():
    fmt.Println(ack.Sequence)
case err := <-ackF.Err():
    fmt.Println(err)
}

// Wait for all pending acks
<-js.PublishAsyncComplete()
```

**New:**

```go
// Async publish does not take context (returns immediately)
ackF, _ := js.PublishAsync("ORDERS.new", []byte("hello"))

select {
case ack := <-ackF.Ok():
    fmt.Println(ack.Sequence)
case err := <-ackF.Err():
    fmt.Println(err)
}

<-js.PublishAsyncComplete()
```

### Publish Options

| Legacy                                   | New                                               |
|------------------------------------------|---------------------------------------------------|
| `nats.MsgId(id)`                         | `jetstream.WithMsgID(id)`                         |
| `nats.ExpectStream(name)`                | `jetstream.WithExpectStream(name)`                |
| `nats.ExpectLastSequence(seq)`           | `jetstream.WithExpectLastSequence(seq)`           |
| `nats.ExpectLastSequencePerSubject(seq)` | `jetstream.WithExpectLastSequencePerSubject(seq)` |
| `nats.ExpectLastMsgId(id)`               | `jetstream.WithExpectLastMsgID(id)`               |
| `nats.RetryWait(dur)`                    | `jetstream.WithRetryWait(dur)`                    |
| `nats.RetryAttempts(n)`                  | `jetstream.WithRetryAttempts(n)`                  |
| `nats.StallWait(dur)`                    | `jetstream.WithStallWait(dur)`                    |

## Consuming Messages

This is the most significant area of change. The legacy API offered many
subscription flavors (`Subscribe`, `SubscribeSync`, `QueueSubscribe`,
`ChanSubscribe`, `PullSubscribe`) that blurred the line between consumer
creation, stream lookup and message consumption. The new API separates these
concerns: first create a consumer, then choose how to receive messages.

With the exception of PullSubscribe, all legacy subscription flavors utilized push consumers under the hood. The new API recommends pull consumers for all use cases, as they provide better flow control and no risk of slow consumer issues. Pull-based consumption is available via `Consume()` and `Messages()`, which maintain persistent pull subscriptions with pre-buffering for efficient continuous delivery. Push consumers are still supported for users who prefer that model, but pull consumers are the recommended default.

### Replacing `js.Subscribe()`

The legacy `js.Subscribe()` created a push consumer behind the scenes (unless
explicitly specified otherwise via `nats.Bind()` or `nats.Durable()`) and
delivered messages either via a callback. In the new API, the recommended
replacement is a **pull consumer** with `Consume()` or `Messages()`. These
provide the same continuous delivery with better flow control.

#### Legacy: callback subscription

```go
sub, _ := js.Subscribe("ORDERS.*", func(msg *nats.Msg) {
    fmt.Printf("Received: %s\n", string(msg.Data))
    msg.Ack()
}, nats.Durable("processor"), nats.ManualAck)
defer sub.Unsubscribe()
```

#### New: callback with `Consume()`

`Consume()` is the closest equivalent to `js.Subscribe()` — it delivers messages
to a callback function continuously.

```go
s, _ := js.Stream(ctx, "ORDERS")
cons, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:       "processor",
    FilterSubject: "ORDERS.*",
})

cc, _ := cons.Consume(func(msg jetstream.Msg) {
    fmt.Printf("Received: %s\n", string(msg.Data()))
    msg.Ack()
})
defer cc.Stop()
```

> Note: `ManualAck()` is not needed — messages are never auto-acknowledged in
> the new API.

#### New: iterator with `Messages()`

`Messages()` provides an iterator-based approach, useful when you want explicit
control over when the next message is fetched.

```go
cons, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:       "processor",
    FilterSubject: "ORDERS.*",
})

iter, _ := cons.Messages()
for {
    msg, err := iter.Next()
    if err != nil {
        // handle error
    }
    fmt.Printf("Received: %s\n", string(msg.Data()))
    msg.Ack()
}
// Call iter.Stop() when done
```

Both `Consume()` and `Messages()` maintain overlapping pull requests to the
server, providing efficient continuous delivery without gaps.

#### Legacy: synchronous subscription

```go
sub, _ := js.SubscribeSync("ORDERS.*", nats.Durable("processor"))
msg, _ := sub.NextMsg(time.Second)
```

**New:** Use `Messages()` and call `Next()`:

```go
cons, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:       "processor",
    FilterSubject: "ORDERS.*",
})

iter, _ := cons.Messages()
msg, _ := iter.Next()
```

#### Legacy: queue subscription

```go
// Multiple instances share work via a queue group
sub, _ := js.QueueSubscribe("ORDERS.*", "workers", handler,
    nats.Durable("processor"))
```

**New with pull consumers:** With pull consumers, there is no need for an
explicit queue group. Multiple application instances (or goroutines) calling
`Consume()` or `Messages()` on the same durable consumer will naturally
distribute messages among themselves — the server tracks pending acknowledgements
and avoids delivering the same message to multiple consumers:

```go
cons, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:   "processor",
})

cc, _ := cons.Consume(handler)
defer cc.Stop()
```

**New with push consumers:** If you need push-based queue semantics, set
`DeliverGroup` on a push consumer — this is the direct equivalent of the legacy
queue group:

```go
cons, _ := s.CreateOrUpdatePushConsumer(ctx, jetstream.ConsumerConfig{
    Durable:        "processor",
    DeliverSubject: "deliver.orders",
    DeliverGroup:   "workers",
})

cc, _ := cons.Consume(handler)
defer cc.Stop()
```

> **Note:** Push consumers with `DeliverGroup` cannot be flow controlled. If you
> experience slow consumer issues, consider using pull-based consumers instead —
> multiple instances on the same durable consumer achieve the same work
> distribution without the slow consumer risk.

#### Legacy: channel subscription

```go
ch := make(chan *nats.Msg, 64)
sub, _ := js.ChanSubscribe("ORDERS.*", ch, nats.Durable("processor"))

for msg := range ch {
    msg.Ack()
}
```

**New:** There is no direct channel-based equivalent. Use `Consume()` or
`Messages()` instead.

### Replacing `js.PullSubscribe()`

The legacy pull subscription required creating a subscription and then calling
`Fetch()` in a loop.

#### Legacy: pull subscribe + fetch loop

```go
sub, _ := js.PullSubscribe("ORDERS.*", "processor")

for {
    msgs, _ := sub.Fetch(10, nats.MaxWait(5*time.Second))
    for _, msg := range msgs {
        fmt.Printf("Received: %s\n", string(msg.Data))
        msg.Ack()
    }
}
```

**New with `Fetch()`/`FetchNoWait()` (one-off batch):**

If you specifically need one-off batch fetching, `Fetch()` is available directly
on the consumer — no separate subscription step:

```go
cons, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:       "processor",
    FilterSubject: "ORDERS.*",
})

// non-blocking, returns a `FetchResult` that provides messages and error
msgs, _ := cons.Fetch(10, jetstream.FetchMaxWait(5*time.Second))
for msg := range msgs.Messages() {
    fmt.Printf("Received: %s\n", string(msg.Data()))
    msg.Ack()
}
if msgs.Error() != nil {
    // handle error
}
```

> **Warning:** `Fetch()`, `FetchNoWait()`, and `FetchBytes()` are one-off,
> single pull requests. They do not perform pre-buffering optimizations. For
> continuous message processing, always prefer `Consume()` or `Messages()`.
> When using `FetchBytes()`, the requested byte size must stay under the
> client's max pending bytes limit (64MB by default), otherwise it will trigger
> slow consumer errors on the underlying subscription.

### Ordered Consumers

Ordered consumers provide strictly ordered, gap-free message delivery. The library
automatically recreates the underlying consumer on sequence gaps or heartbeat
failures.

**Legacy:**

```go
sub, _ := js.Subscribe("ORDERS.*", handler, nats.OrderedConsumer())
```

**New:**

```go
cons, _ := js.OrderedConsumer(ctx, "ORDERS", jetstream.OrderedConsumerConfig{
    FilterSubjects: []string{"ORDERS.*"},
})

// Use the same consumption methods as regular consumers
cc, _ := cons.Consume(func(msg jetstream.Msg) {
    fmt.Printf("Received: %s\n", string(msg.Data()))
})
defer cc.Stop()
```

### Push Consumers

Pull consumers are recommended for most use cases, but push consumers are also
supported. Push consumers require `DeliverSubject` in their config and only
support `Consume()` (not `Fetch()` or `Messages()`).

**Legacy:**

```go
sub, _ := js.Subscribe("ORDERS.*", handler,
    nats.Durable("processor"),
    nats.DeliverSubject("deliver.orders"),
    nats.IdleHeartbeat(30*time.Second),
)
```

**New:**

```go
cons, _ := s.CreateOrUpdatePushConsumer(ctx, jetstream.ConsumerConfig{
    Durable:        "processor",
    FilterSubject:  "ORDERS.*",
    DeliverSubject: "deliver.orders",
    IdleHeartbeat:  30 * time.Second,
})

cc, _ := cons.Consume(func(msg jetstream.Msg) {
    fmt.Printf("Received: %s\n", string(msg.Data()))
    msg.Ack()
})
defer cc.Stop()
```

### Subscription Options Mapping

Most legacy `SubOpt` options map directly to `ConsumerConfig` fields. Since
consumer creation is explicit, these are set at creation time rather than passed
as subscription options.

| Legacy SubOpt                       | New ConsumerConfig field                                                  |
|-------------------------------------|---------------------------------------------------------------------------|
| `nats.Durable("name")`              | `Durable: "name"`                                                         |
| `nats.ConsumerName("name")`         | `Name: "name"`                                                            |
| `nats.Description("desc")`          | `Description: "desc"`                                                     |
| `nats.DeliverAll()`                 | `DeliverPolicy: jetstream.DeliverAllPolicy`                               |
| `nats.DeliverLast()`                | `DeliverPolicy: jetstream.DeliverLastPolicy`                              |
| `nats.DeliverLastPerSubject()`      | `DeliverPolicy: jetstream.DeliverLastPerSubjectPolicy`                    |
| `nats.DeliverNew()`                 | `DeliverPolicy: jetstream.DeliverNewPolicy`                               |
| `nats.StartSequence(seq)`           | `DeliverPolicy: jetstream.DeliverByStartSequencePolicy, OptStartSeq: seq` |
| `nats.StartTime(t)`                 | `DeliverPolicy: jetstream.DeliverByStartTimePolicy, OptStartTime: &t`     |
| `nats.AckExplicit()`                | `AckPolicy: jetstream.AckExplicitPolicy`                                  |
| `nats.AckAll()`                     | `AckPolicy: jetstream.AckAllPolicy`                                       |
| `nats.AckNone()`                    | `AckPolicy: jetstream.AckNonePolicy`                                      |
| `nats.ManualAck()`                  | Not needed (messages are never auto-acked)                                |
| `nats.MaxDeliver(n)`                | `MaxDeliver: n`                                                           |
| `nats.MaxAckPending(n)`             | `MaxAckPending: n`                                                        |
| `nats.BackOff(durations)`           | `BackOff: durations`                                                      |
| `nats.ReplayOriginal()`             | `ReplayPolicy: jetstream.ReplayOriginalPolicy`                            |
| `nats.ReplayInstant()`              | `ReplayPolicy: jetstream.ReplayInstantPolicy`                             |
| `nats.RateLimit(bps)`               | `RateLimit: bps`                                                          |
| `nats.HeadersOnly()`                | `HeadersOnly: true`                                                       |
| `nats.InactiveThreshold(dur)`       | `InactiveThreshold: dur`                                                  |
| `nats.ConsumerFilterSubjects(s...)` | `FilterSubjects: s`                                                       |
| `nats.ConsumerReplicas(n)`          | `Replicas: n`                                                             |
| `nats.ConsumerMemoryStorage()`      | `MemoryStorage: true`                                                     |

The following options have no direct equivalent — use the consumer handle
directly instead:

| Legacy SubOpt                 | New equivalent                                                      |
|-------------------------------|---------------------------------------------------------------------|
| `nats.Bind(stream, consumer)` | `js.Consumer(ctx, stream, consumer)` or `s.Consumer(ctx, consumer)` |
| `nats.BindStream(stream)`     | Use `js.Stream(ctx, stream)` to get a stream handle                 |
| `nats.OrderedConsumer()`      | `js.OrderedConsumer(ctx, stream, cfg)`                              |

### Consume/Messages Options

`Consume()` and `Messages()` accept options that control pull request behavior:

| Option                     | Description                                                  |
|----------------------------|--------------------------------------------------------------|
| `PullMaxMessages(n)`       | Max messages buffered (default: 500)                         |
| `PullMaxBytes(n)`          | Max bytes buffered (mutually exclusive with PullMaxMessages) |
| `PullExpiry(dur)`          | Pull request timeout (default: 30s)                          |
| `PullHeartbeat(dur)`       | Idle heartbeat interval                                      |
| `PullThresholdMessages(n)` | Refill threshold (default: 50% of max)                       |
| `PullThresholdBytes(n)`    | Byte-based refill threshold                                  |
| `StopAfter(n)`             | Auto-stop after N messages                                   |
| `ConsumeErrHandler(fn)`    | Custom error handler                                         |

### Error Handling in Consume/Messages

Both `Consume()` and `Messages()` handle server-sent status messages internally.
Some errors are terminal (stop consumption), while others are recoverable
(consumption continues).

**Terminal errors** — consumption stops automatically:

- `ErrConsumerDeleted` — the consumer was deleted on the server
- `ErrBadRequest` — invalid request (e.g. misconfigured consumer)
- Connection closed — for `Consume()` this surfaces as `ErrConnectionClosed`;
  for `Messages()`, `Next()` returns `ErrMsgIteratorClosed`

**Recoverable errors** — reported via error handler, consumption continues:

- `ErrNoHeartbeat` — missed idle heartbeats from server; a new pull request
  is issued automatically
- `ErrConsumerLeadershipChanged` — consumer moved to a different server in the
  cluster; pending counts are reset
- `nats.ErrNoResponders` — no JetStream service available (temporary)

#### Error handling with `Consume()`

Use `ConsumeErrHandler` to be notified about both terminal and recoverable errors:

```go
cc, _ := cons.Consume(func(msg jetstream.Msg) {
    msg.Ack()
}, jetstream.ConsumeErrHandler(func(cc jetstream.ConsumeContext, err error) {
    if errors.Is(err, jetstream.ErrConsumerDeleted) ||
        errors.Is(err, jetstream.ErrBadRequest) {
        log.Fatalf("terminal consumer error: %v", err)
    }
    log.Printf("recoverable consumer error: %v", err)
}))
defer cc.Stop()
```

#### Error handling with `Messages()`

With `Messages()`, terminal errors are returned directly by `Next()`. By default,
`ErrNoHeartbeat` is also returned by `Next()` (controlled by
`WithMessagesErrOnMissingHeartbeat`), but it is not terminal — you can continue
calling `Next()`:

```go
iter, _ := cons.Messages()
for {
    msg, err := iter.Next()
    if err != nil {
        if errors.Is(err, jetstream.ErrMsgIteratorClosed) {
            // iterator was stopped (either explicitly or due to connection close)
            break
        }
        if errors.Is(err, jetstream.ErrNoHeartbeat) {
            // recoverable — new pull request is issued, keep going
            log.Println("missed heartbeat, re-pulling")
            continue
        }
        // ErrConsumerDeleted, ErrBadRequest are terminal
        log.Fatalf("terminal error: %v", err)
    }
    msg.Ack()
}
```

## Message Acknowledgement

Ack methods are similar, with minor naming changes. The main difference is that
message fields are accessed via methods instead of struct fields.

| Legacy                  | New                          |
|-------------------------|------------------------------|
| `msg.Ack()`             | Unchanged                    |
| `msg.AckSync()`         | `msg.DoubleAck(ctx)`         |
| `msg.Nak()`             | Unchanged                    |
| `msg.NakWithDelay(dur)` | Unchanged                    |
| `msg.InProgress()`      | Unchanged                    |
| `msg.Term()`            | Unchanged                    |
| N/A                     | `msg.TermWithReason(reason)` |
| `msg.Metadata()`        | Unchanged                    |

### Accessing Message Data

**Legacy:** Direct struct fields on `*nats.Msg`:

```go
fmt.Println(string(msg.Data))
fmt.Println(msg.Subject)
fmt.Println(msg.Header.Get("key"))
```

**New:** Methods on `jetstream.Msg` interface:

```go
fmt.Println(string(msg.Data()))
fmt.Println(msg.Subject())
fmt.Println(msg.Headers().Get("key"))
```

## KeyValue Store

The KV API is nearly identical. The main changes are:

1. All methods take `context.Context` as the first parameter
2. New `CreateOrUpdateKeyValue()` and `UpdateKeyValue()` methods
3. Types live in the `jetstream` package

**Legacy:**

```go
js, _ := nc.JetStream()
kv, _ := js.CreateKeyValue(&nats.KeyValueConfig{
    Bucket: "profiles",
})

kv.Put("sue.color", []byte("blue"))
entry, _ := kv.Get("sue.color")
fmt.Println(string(entry.Value()))

watcher, _ := kv.Watch("sue.*")
defer watcher.Stop()
```

**New:**

```go
js, _ := jetstream.New(nc)
kv, _ := js.CreateKeyValue(ctx, jetstream.KeyValueConfig{
    Bucket: "profiles",
})

kv.Put(ctx, "sue.color", []byte("blue"))
entry, _ := kv.Get(ctx, "sue.color")
fmt.Println(string(entry.Value()))

watcher, _ := kv.Watch(ctx, "sue.*")
defer watcher.Stop()
```

### KV Management Methods

| Legacy                      | New                                   |
|-----------------------------|---------------------------------------|
| `js.KeyValue(bucket)`       | `js.KeyValue(ctx, bucket)`            |
| `js.CreateKeyValue(cfg)`    | `js.CreateKeyValue(ctx, cfg)`         |
| N/A                         | `js.UpdateKeyValue(ctx, cfg)`         |
| N/A                         | `js.CreateOrUpdateKeyValue(ctx, cfg)` |
| `js.DeleteKeyValue(bucket)` | `js.DeleteKeyValue(ctx, bucket)`      |
| `js.KeyValueStoreNames()`   | `js.KeyValueStoreNames(ctx)`          |
| `js.KeyValueStores()`       | `js.KeyValueStores(ctx)`              |

## Object Store

Same pattern as KV — all methods gain `context.Context`, types move to `jetstream`
package.

**Legacy:**

```go
js, _ := nc.JetStream()
os, _ := js.CreateObjectStore(&nats.ObjectStoreConfig{
    Bucket: "configs",
})

os.PutString("config-1", "data")
result, _ := os.Get("config-1")
data, _ := io.ReadAll(result)
```

**New:**

```go
js, _ := jetstream.New(nc)
os, _ := js.CreateObjectStore(ctx, jetstream.ObjectStoreConfig{
    Bucket: "configs",
})

os.PutString(ctx, "config-1", "data")
result, _ := os.Get(ctx, "config-1")
data, _ := io.ReadAll(result)
```

### Object Store Management Methods

| Legacy                         | New                                      |
|--------------------------------|------------------------------------------|
| `js.ObjectStore(bucket)`       | `js.ObjectStore(ctx, bucket)`            |
| `js.CreateObjectStore(cfg)`    | `js.CreateObjectStore(ctx, cfg)`         |
| N/A                            | `js.UpdateObjectStore(ctx, cfg)`         |
| N/A                            | `js.CreateOrUpdateObjectStore(ctx, cfg)` |
| `js.DeleteObjectStore(bucket)` | `js.DeleteObjectStore(ctx, bucket)`      |
