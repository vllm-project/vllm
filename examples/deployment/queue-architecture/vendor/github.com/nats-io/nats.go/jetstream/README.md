
# JetStream Simplified Client [![JetStream API Reference](https://pkg.go.dev/badge/github.com/nats-io/nats.go/jetstream.svg)](https://pkg.go.dev/github.com/nats-io/nats.go/jetstream)

This doc covers the basic usage of the `jetstream` package in `nats.go` client.

- [Overview](#overview)
- [Basic usage](#basic-usage)
- [Streams](#streams)
  - [Stream management (CRUD)](#stream-management-crud)
  - [Listing streams and stream names](#listing-streams-and-stream-names)
  - [Stream-specific operations](#stream-specific-operations)
- [Consumers](#consumers)
  - [Consumers management](#consumers-management)
  - [Listing consumers and consumer names](#listing-consumers-and-consumer-names)
  - [Ordered consumers](#ordered-consumers)
  - [Receiving messages from pull consumers](#receiving-messages-from-pull-consumers)
    - [Single fetch](#single-fetch)
    - [Continuous polling](#continuous-polling)
      - [Using `Consume()` receive messages in a callback](#using-consume-receive-messages-in-a-callback)
      - [Using `Messages()` to iterate over incoming messages](#using-messages-to-iterate-over-incoming-messages)
    - [Receiving messages from push consumers](#receiving-messages-from-push-consumers)
- [Publishing on stream](#publishing-on-stream)
  - [Synchronous publish](#synchronous-publish)
  - [Async publish](#async-publish)
- [KeyValue Store](#keyvalue-store)
  - [Basic usage of KV bucket](#basic-usage-of-kv-bucket)
  - [Watching for changes on a bucket](#watching-for-changes-on-a-bucket)
  - [Additional operations on a bucket](#additional-operations-on-a-bucket)
- [Object Store](#object-store)
  - [Basic usage of Object Store](#basic-usage-of-object-store)
  - [Watching for changes on a store](#watching-for-changes-on-a-store)
  - [Additional operations on a store](#additional-operations-on-a-store)
- [Examples](#examples)

## Overview

`jetstream` package is a new client API to interact with NATS JetStream, aiming
to replace the JetStream client implementation from `nats` package. The main
goal of this package is to provide a simple and clear way to interact with
JetStream API. Key differences between `jetstream` and `nats` packages include:

- Using smaller, simpler interfaces to manage streams and consumers
- Using more granular and predictable approach to consuming messages from a
  stream, instead of relying on often complicated and unpredictable
  `Subscribe()` method (and all of its flavors)
- Allowing the usage of pull consumers to continuously receive incoming messages
  (including ordered consumer functionality)
- Separating JetStream context from core NATS

`jetstream` package provides several ways of interacting with the API:

- `JetStream` - top-level interface, used to create and manage streams,
  consumers and publishing messages
- `Stream` - used to manage consumers for a specific stream, as well as
  performing stream-specific operations (purging, fetching and deleting messages
  by sequence number, fetching stream info)
- `Consumer` - used to get information about a consumer as well as consuming
  messages
- `Msg` - used for message-specific operations - reading data, headers and
  metadata, as well as performing various types of acknowledgements

Additionally, `jetstream` exposes [KeyValue Store](#keyvalue-store) and
[ObjectStore](#object-store) capabilities. KV and Object stores are abstraction
layers on top of JetStream Streams, simplifying key value and large data
storage on Streams.

> __NOTE__: `jetstream` requires nats-server >= 2.9.0 to work correctly.

## Basic usage

```go
package main

import (
    "context"
    "fmt"
    "strconv"
    "time"

    "github.com/nats-io/nats.go"
    "github.com/nats-io/nats.go/jetstream"
)

func main() {
    // In the `jetstream` package, almost all API calls rely on `context.Context` for timeout/cancellation handling
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    nc, _ := nats.Connect(nats.DefaultURL)

    // Create a JetStream management interface
    js, _ := jetstream.New(nc)

    // Create a stream
    s, _ := js.CreateStream(ctx, jetstream.StreamConfig{
        Name:     "ORDERS",
        Subjects: []string{"ORDERS.*"},
    })

    // Publish some messages
    for i := 0; i < 100; i++ {
        js.Publish(ctx, "ORDERS.new", []byte("hello message "+strconv.Itoa(i)))
        fmt.Printf("Published hello message %d\n", i)
    }

    // Create durable consumer
    c, _ := s.CreateOrUpdateConsumer(ctx, jetstream.ConsumerConfig{
        Durable:   "CONS",
        AckPolicy: jetstream.AckExplicitPolicy,
    })

    // Get 10 messages from the consumer
    messageCounter := 0
    msgs, err := c.Fetch(10)
    if err != nil {
        // handle error
    }

    for msg := range msgs.Messages() {
        msg.Ack()
        fmt.Printf("Received a JetStream message via fetch: %s\n", string(msg.Data()))
        messageCounter++
    }

    fmt.Printf("Received %d messages\n", messageCounter)

    if msgs.Error() != nil {
        fmt.Println("Error during Fetch(): ", msgs.Error())
    }

    // Receive messages continuously in a callback
    cons, _ := c.Consume(func(msg jetstream.Msg) {
        msg.Ack()
        fmt.Printf("Received a JetStream message via callback: %s\n", string(msg.Data()))
        messageCounter++
    })
    defer cons.Stop()

    // Iterate over messages continuously
    it, _ := c.Messages()
    for i := 0; i < 10; i++ {
        msg, _ := it.Next()
        msg.Ack()
        fmt.Printf("Received a JetStream message via iterator: %s\n", string(msg.Data()))
        messageCounter++
    }
    it.Stop()

    // block until all 100 published messages have been processed
    for messageCounter < 100 {
        time.Sleep(10 * time.Millisecond)
    }
}
```

## Streams

`jetstream` provides methods to manage and list streams, as well as perform
stream-specific operations (purging, fetching/deleting messages by sequence id)

### Stream management (CRUD)

```go
js, _ := jetstream.New(nc)

// create a stream (this is an idempotent operation)
s, _ := js.CreateStream(ctx, jetstream.StreamConfig{
    Name:     "ORDERS",
    Subjects: []string{"ORDERS.*"},
})

// update a stream
s, _ = js.UpdateStream(ctx, jetstream.StreamConfig{
    Name:        "ORDERS",
    Subjects:    []string{"ORDERS.*"},
    Description: "updated stream",
})

// get stream handle
s, _ = js.Stream(ctx, "ORDERS")

// delete a stream
js.DeleteStream(ctx, "ORDERS")
```

### Listing streams and stream names

```go
// list streams
streams := js.ListStreams(ctx)
for s := range streams.Info() {
    fmt.Println(s.Config.Name)
}
if streams.Err() != nil {
    fmt.Println("Unexpected error occurred")
}

// list stream names
names := js.StreamNames(ctx)
for name := range names.Name() {
    fmt.Println(name)
}
if names.Err() != nil {
    fmt.Println("Unexpected error occurred")
}
```

### Stream-specific operations

Using `Stream` interface, it is also possible to:

- Purge a stream

```go
// remove all messages from a stream
_ = s.Purge(ctx)

// remove all messages from a stream that are stored on a specific subject
_ = s.Purge(ctx, jetstream.WithPurgeSubject("ORDERS.new"))

// remove all messages up to specified sequence number
_ = s.Purge(ctx, jetstream.WithPurgeSequence(100))

// remove messages, but keep 10 newest
_ = s.Purge(ctx, jetstream.WithPurgeKeep(10))
```

- Get and delete messages from a stream

```go
// get message from stream with sequence number == 100
msg, _ := s.GetMsg(ctx, 100)

// get last message from "ORDERS.new" subject
msg, _ = s.GetLastMsgForSubject(ctx, "ORDERS.new")

// delete a message with sequence number == 100
_ = s.DeleteMsg(ctx, 100)
```

- Get information about a stream

```go
// Fetches the latest stream info from server
info, _ := s.Info(ctx)
fmt.Println(info.Config.Name)

// Returns the most recently fetched StreamInfo, without making an API call to the server
cachedInfo := s.CachedInfo()
fmt.Println(cachedInfo.Config.Name)
```

## Consumers

Both pull and push consumers are supported in `jetstream` package. For most use
cases, we recommend using pull consumers as they allow for more fine-grained
control over the message processing and can often prevent issues such as slow
consumers. However, unlike the JetStream API in `nats` package, pull consumers
allow for continuous message retrieval (similarly to how `nats.Subscribe()`
works). Because of that, push consumers can be easily replaced by pull
consumers for most of the use cases. Push consumers are supported mainly for
the purpose of ease of migration from `nats` package. The interfaces for
consuming messages via push and pull consumers are similar, with the main
difference being that push consumers do not support fetching individual batches
of messages.

### Consumers management

Both pull and push consumers can be managed using `jetstream` package. The
following example demonstrates how to create, update, fetch and delete a pull
consumer. Push consumers can be managed in a similar way, with method names
containing `Push` (e.g. `CreatePushConsumer`, `UpdatePushConsumer`,
`DeletePushConsumer`).

> __NOTE__: It is important to use `CreateConsumer` and `CreatePushConsumer`
methods to create the respective consumer types as they return the correct
interface (different for push and pull consumers). `DeliverSubject` is mandatory
when creating a push consumer and cannot be provided when creating a pull
consumer. Similarly, an attempt to get a push consumer using `Consumer` method
will result in an error (and vice versa).

CRUD operations on pull consumers can be achieved on 2 levels:

- on `JetStream` interface

```go
js, _ := jetstream.New(nc)

// create a consumer (this is an idempotent operation)
// an error will be returned if consumer already exists and has different configuration.
cons, _ := js.CreateConsumer(ctx, "ORDERS", jetstream.ConsumerConfig{
    Durable: "foo",
    AckPolicy: jetstream.AckExplicitPolicy,
})

// create an ephemeral pull consumer by not providing `Durable`
ephemeral, _ := js.CreateConsumer(ctx, "ORDERS", jetstream.ConsumerConfig{
    AckPolicy: jetstream.AckExplicitPolicy,
})


// consumer can also be created using CreateOrUpdateConsumer
// this method will either create a consumer if it does not exist
// or update existing consumer (if possible)
cons2 := js.CreateOrUpdateConsumer(ctx, "ORDERS", jetstream.ConsumerConfig{
    Name: "bar",
})

// consumers can be updated
// an error will be returned if consumer with given name does not exist
// or an illegal property is to be updated (e.g. AckPolicy)
updated, _ := js.UpdateConsumer(ctx, "ORDERS", jetstream.ConsumerConfig{
    AckPolicy: jetstream.AckExplicitPolicy,
    Description: "updated consumer",
})

// get consumer handle
cons, _ = js.Consumer(ctx, "ORDERS", "foo")

// delete a consumer
js.DeleteConsumer(ctx, "ORDERS", "foo")
```

- on `Stream` interface

```go
// Create a JetStream management interface
js, _ := jetstream.New(nc)

// get stream handle
stream, _ := js.Stream(ctx, "ORDERS")

// create consumer
cons, _ := stream.CreateConsumer(ctx, jetstream.ConsumerConfig{
    Durable:   "foo",
    AckPolicy: jetstream.AckExplicitPolicy,
})

// get consumer handle
cons, _ = stream.Consumer(ctx, "foo")

// delete a consumer
stream.DeleteConsumer(ctx, "foo")
```

`Consumer` interface, returned when creating/fetching consumers, allows fetching
`ConsumerInfo`:

```go
// Fetches latest consumer info from server
info, _ := cons.Info(ctx)
fmt.Println(info.Config.Durable)

// Returns the most recently fetched ConsumerInfo, without making an API call to the server
cachedInfo := cons.CachedInfo()
fmt.Println(cachedInfo.Config.Durable)
```

### Listing consumers and consumer names

```go
// list consumers
consumers := s.ListConsumers(ctx)
for cons := range consumers.Info() {
    fmt.Println(cons.Name)
}
if consumers.Err() != nil {
    fmt.Println("Unexpected error occurred")
}

// list consumer names
names := s.ConsumerNames(ctx)
for name := range names.Name() {
    fmt.Println(name)
}
if names.Err() != nil {
    fmt.Println("Unexpected error occurred")
}
```

### Ordered consumers

`jetstream`, in addition to basic named/ephemeral consumers, supports ordered
consumer functionality. Ordered is strictly processing messages in the order
that they were stored on the stream, providing a consistent and deterministic
message ordering. It is also resilient to consumer deletion.

Ordered consumers present the same set of message consumption methods as
standard pull consumers.

> __NOTE__: Ordered consumers are not supported for push consumers.

```go
js, _ := jetstream.New(nc)

// create a consumer (this is an idempotent operation)
cons, _ := js.OrderedConsumer(ctx, "ORDERS", jetstream.OrderedConsumerConfig{
    // Filter results from "ORDERS" stream by specific subject
    FilterSubjects: []string{"ORDERS.A"},
})
```

### Receiving messages from pull consumers

The `Consumer` interface allows fetching messages on demand, with a
pre-defined batch size or byte limit, or continuous push-like receiving of
messages.

#### __Single fetch__

This pattern allows fetching a defined number of messages in a single RPC.

- Using `Fetch` or `FetchBytes`, consumer will return up to the provided number
of messages/bytes. By default, `Fetch()` will wait 30 seconds before timing out
(this behavior can be configured using `FetchMaxWait()` option):

```go
// receive up to 10 messages from the stream
msgs, err := c.Fetch(10)
if err != nil {
    // handle error
}

for msg := range msgs.Messages() {
    fmt.Printf("Received a JetStream message: %s\n", string(msg.Data()))
}

if msgs.Error() != nil {
    // handle error
}

// receive up to 1024 B of data
msgs, err := c.FetchBytes(1024)
if err != nil {
// handle error
}

for msg := range msgs.Messages() {
    fmt.Printf("Received a JetStream message: %s\n", string(msg.Data()))
}

if msgs.Error() != nil {
    // handle error
}
```

Similarly, `FetchNoWait()` can be used in order to only return messages from the
stream available at the time of sending request:

```go
// FetchNoWait will not wait for new messages if the whole batch is not available at the time of sending request.
msgs, err := c.FetchNoWait(10)
if err != nil {
// handle error
}

for msg := range msgs.Messages() {
    fmt.Printf("Received a JetStream message: %s\n", string(msg.Data()))
}

if msgs.Error() != nil {
    // handle error
}
```

> __Warning__: Both `Fetch()` and `FetchNoWait()` have worse performance when
> used to continuously retrieve messages in comparison to `Messages()` or
`Consume()` methods, as they do not perform any optimizations (pre-buffering)
and new subscription is created for each execution.

#### Continuous polling

There are 2 ways to achieve push-like behavior using pull consumers in
`jetstream` package. Both `Messages()` and `Consume()` methods perform similar optimizations
and for most cases can be used interchangeably.

There is an advantage of using `Messages()` instead of `Consume()` for work-queue scenarios,
where messages should be fetched one by one, as it allows for finer control over fetching
single messages on demand.

Subject filtering is achieved by configuring a consumer with a `FilterSubject`
value.

##### Using `Consume()` to receive messages in a callback

```go
cons, _ := js.CreateOrUpdateConsumer(ctx, "ORDERS", jetstream.ConsumerConfig{
    AckPolicy: jetstream.AckExplicitPolicy,
    // receive messages from ORDERS.A subject only
    FilterSubject: "ORDERS.A"
})

consContext, _ := c.Consume(func(msg jetstream.Msg) {
    fmt.Printf("Received a JetStream message: %s\n", string(msg.Data()))
    // messages are not acknowledged automatically
    msg.Ack()
})
defer consContext.Stop()
```

Similar to `Messages()`, `Consume()` can be supplied with options to modify
the behavior of a single pull request:

- `PullMaxMessages(int)` - up to provided number of messages will be buffered
- `PullMaxBytes(int)` - up to provided number of bytes will be buffered. This
  setting and `PullMaxMessages` are mutually exclusive.
  The value should be set to a high enough value to accommodate the largest
  message expected from the server. Note that it may not be sufficient to set
  this value to the maximum message size, as this setting controls the client
  buffer size, not the max bytes requested from the server within a single pull
  request. If the value is set too low, the consumer will stall and not be able
  to consume messages.
- `PullExpiry(time.Duration)` - timeout on a single pull request to the server
- `PullThresholdMessages(int)` - amount of messages which triggers refilling the
  buffer
- `PullThresholdBytes(int)` - amount of bytes which triggers refilling the
  buffer
- `PullHeartbeat(time.Duration)` - idle heartbeat duration for a single pull
request. An error will be triggered if at least 2 heartbeats are missed
- `ConsumeErrHandler(func (ConsumeContext, error))` - when used, sets a
  custom error handler on `Consume()`, allowing e.g. tracking missing
  heartbeats.
- `PullMaxMessagesWithBytesLimit(int, int)` - up to the provided number of messages
  will be buffered and a single fetch size will be limited to the provided value.
  This is an advanced option and should be used with caution. Most of the time,
  `PullMaxMessages` or `PullMaxBytes` should be used instead. Note that the byte
  limit should never be set to a value lower than the maximum message size that
  can be expected from the server. If the byte limit is lower than the maximum
  message size, the consumer will stall and not be able to consume messages.

> __NOTE__: `Stop()` should always be called on `ConsumeContext` to avoid
> leaking goroutines.

##### Using `Messages()` to iterate over incoming messages

```go
iter, _ := cons.Messages()
for {
    msg, err := iter.Next()
    // Next can return error, e.g. when iterator is closed or no heartbeats were received
    if err != nil {
        //handle error
    }
    fmt.Printf("Received a JetStream message: %s\n", string(msg.Data()))
    msg.Ack()
}
iter.Stop()
```

It can also be configured to only store up to defined number of messages/bytes
in the buffer.

```go
// a maximum of 10 messages or 1024 bytes will be stored in memory (whichever is encountered first)
iter, _ := cons.Messages(jetstream.PullMaxMessages(10), jetstream.PullMaxBytes(1024))
```

`Messages()` exposes the following options:

- `PullMaxMessages(int)` - up to provided number of messages will be buffered
- `PullMaxBytes(int)` - up to provided number of bytes will be buffered. This
  setting and `PullMaxMessages` are mutually exclusive.
  The value should be set to a high enough value to accommodate the largest
  message expected from the server. Note that it may not be sufficient to set
  this value to the maximum message size, as this setting controls the client
  buffer size, not the max bytes requested from the server within a single pull
  request. If the value is set too low, the consumer will stall and not be able
  to consume messages.
- `PullExpiry(time.Duration)` - timeout on a single pull request to the server
- `PullThresholdMessages(int)` - amount of messages which triggers refilling the
  buffer
- `PullThresholdBytes(int)` - amount of bytes which triggers refilling the
  buffer
- `PullHeartbeat(time.Duration)` - idle heartbeat duration for a single pull
request. An error will be triggered if at least 2 heartbeats are missed (unless
`WithMessagesErrOnMissingHeartbeat(false)` is used)
- `PullMaxMessagesWithBytesLimit(int, int)` - up to the provided number of messages
  will be buffered and a single fetch size will be limited to the provided value.
  This is an advanced option and should be used with caution. Most of the time,
  `PullMaxMessages` or `PullMaxBytes` should be used instead. Note that the byte
  limit should never be set to a value lower than the maximum message size that
  can be expected from the server. If the byte limit is lower than the maximum
  message size, the consumer will stall and not be able to consume messages.

##### Using `Messages()` to fetch single messages one by one

When implementing work queue, it is possible to use `Messages()` in order to
fetch messages from the server one-by-one, without optimizations and
pre-buffering (to avoid redeliveries when processing messages at slow rate).

```go
// PullMaxMessages determines how many messages will be sent to the client in a single pull request
iter, _ := cons.Messages(jetstream.PullMaxMessages(1))
numWorkers := 5
sem := make(chan struct{}, numWorkers)
for {
    sem <- struct{}{}
    go func() {
        defer func() {
            <-sem
        }()
        msg, err := iter.Next()
        if err != nil {
            // handle err
        }
        fmt.Printf("Processing msg: %s\n", string(msg.Data()))
        doWork()
        msg.Ack()
    }()
}
```

#### Receiving messages from push consumers

The `PushConsumer` interface currently only allows message processing in a
callback using `Consume()`.

As heartbeat for push consumers is not managed when using `Consume()`, it is
important to set `IdleHeartbeat` on the consumer level. Similarly, `FlowControl`
can be set to prevent the consumer from receiving more messages than it can
handle.

```go
cons, _ := js.CreateOrUpdatePushConsumer(ctx, "ORDERS", jetstream.ConsumerConfig{
    DeliverSubject: nats.NewInbox()
    AckPolicy: jetstream.AckExplicitPolicy,
    // receive messages from ORDERS.A subject only
    FilterSubject: "ORDERS.A",
    // unlike pull consumers, idle heartbeat is configured on the consumer level
    IdleHeartbeat: 30 * time.Second
})

consContext, _ := c.Consume(func(msg jetstream.Msg) {
    fmt.Printf("Received a JetStream message: %s\n", string(msg.Data()))
    // messages are not acknowledged automatically
    msg.Ack()
})
defer consContext.Stop()
```

`Consume()` on `PushConsumer` can be supplied with `ConsumeErrHandler` option
to set a custom error handler allowing e.g. tracking missing heartbeats.

> __NOTE__: `Stop()` should always be called on `ConsumeContext` to avoid
> leaking goroutines.

## Publishing on stream

`JetStream` interface allows publishing messages on stream in 2 ways:

### __Synchronous publish__

```go
js, _ := jetstream.New(nc)

// Publish message on subject ORDERS.new
// Given subject has to belong to a stream
ack, err := js.PublishMsg(ctx, &nats.Msg{
    Data:    []byte("hello"),
    Subject: "ORDERS.new",
})
fmt.Printf("Published msg with sequence number %d on stream %q", ack.Sequence, ack.Stream)

// A helper method accepting subject and data as parameters
ack, err = js.Publish(ctx, "ORDERS.new", []byte("hello"))
```

Both `Publish()` and `PublishMsg()` can be supplied with options allowing
setting various headers. Additionally, for `PublishMsg()` headers can be set
directly on `nats.Msg`.

```go
// All 3 implementations work identically 
ack, err := js.PublishMsg(ctx, &nats.Msg{
    Data:    []byte("hello"),
    Subject: "ORDERS.new",
    Header: nats.Header{
        "Nats-Msg-Id": []string{"id"},
    },
})

ack, err = js.PublishMsg(ctx, &nats.Msg{
    Data:    []byte("hello"),
    Subject: "ORDERS.new",
}, jetstream.WithMsgID("id"))

ack, err = js.Publish(ctx, "ORDERS.new", []byte("hello"), jetstream.WithMsgID("id"))
```

### __Async publish__

```go
js, _ := jetstream.New(nc)

// publish message and do not wait for ack
ackF, err := js.PublishMsgAsync(ctx, &nats.Msg{
    Data:    []byte("hello"),
    Subject: "ORDERS.new",
})

// block and wait for ack
select {
case ack := <-ackF.Ok():
    fmt.Printf("Published msg with sequence number %d on stream %q", ack.Sequence, ack.Stream)
case err := <-ackF.Err():
    fmt.Println(err)
}

// similarly to synchronous publish, there is a helper method accepting subject and data
ackF, err = js.PublishAsync("ORDERS.new", []byte("hello"))
```

Just as for synchronous publish, `PublishAsync()` and `PublishMsgAsync()` accept
options for setting headers.

## KeyValue Store

JetStream KeyValue Stores offer a straightforward method for storing key-value
pairs within JetStream. These stores are supported by a specially configured
stream, designed to efficiently and compactly store these pairs. This structure
ensures rapid and convenient access to the data.

The KV Store, also known as a bucket, enables the execution of various operations:

- create/update a value for a given key
- get a value for a given key
- delete a value for a given key
- purge all values from a bucket
- list all keys in a bucket
- watch for changes on given key set or the whole bucket
- retrieve history of changes for a given key

### Basic usage of KV bucket

The most basic usage of KV bucket is to create or retrieve a bucket and perform
basic CRUD operations on keys.

```go
js, _ := jetstream.New(nc)
ctx := context.Background()

// Create a new bucket. Bucket name is required and has to be unique within a JetStream account.
kv, _ := js.CreateKeyValue(ctx, jetstream.KeyValueConfig{Bucket: "profiles"})

// Set a value for a given key
// Put will either create or update a value for a given key
kv.Put(ctx, "sue.color", []byte("blue"))

// Get an entry for a given key
// Entry contains key/value, but also metadata (revision, timestamp, etc.)) 
entry, _ := kv.Get(ctx, "sue.color")

// Prints `sue.color @ 1 -> "blue"`
fmt.Printf("%s @ %d -> %q\n", entry.Key(), entry.Revision(), string(entry.Value()))

// Update a value for a given key
// Update will fail if the key does not exist or the revision has changed
kv.Update(ctx, "sue.color", []byte("red"), 1)

// Create will fail if the key already exists
_, err := kv.Create(ctx, "sue.color", []byte("purple"))
fmt.Println(err) // prints `nats: key exists`

// Delete a value for a given key.
// Delete is not destructive, it will add a delete marker for a given key
// and all previous revisions will still be available
kv.Delete(ctx, "sue.color")

// getting a deleted key will return an error
_, err = kv.Get(ctx, "sue.color")
fmt.Println(err) // prints `nats: key not found`

// A bucket can be deleted once it is no longer needed
js.DeleteKeyValue(ctx, "profiles")
```

### Watching for changes on a bucket

KV buckets support Watchers, which can be used to watch for changes on a given
key or the whole bucket. Watcher will receive a notification on a channel when a
change occurs. By default, watcher will return initial values for all matching
keys. After sending all initial values, watcher will send nil on the channel to
signal that all initial values have been sent and it will start sending updates when
changes occur.

Watcher supports several configuration options:

- `IncludeHistory` will have the key watcher send all historical values
for each key (up to KeyValueMaxHistory).
- `IgnoreDeletes` will have the key watcher not pass any keys with
delete markers.
- `UpdatesOnly` will have the key watcher only pass updates on values
(without values already present when starting).
- `MetaOnly` will have the key watcher retrieve only the entry metadata, not the entry value.
- `ResumeFromRevision` instructs the key watcher to resume from a
specific revision number.

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
kv, _ := js.CreateKeyValue(ctx, jetstream.KeyValueConfig{Bucket: "profiles"})

kv.Put(ctx, "sue.color", []byte("blue"))

// A watcher can be created to watch for changes on a given key or the whole bucket
// By default, watcher will return most recent values for all matching keys.
// Watcher can be configured to only return updates by using jetstream.UpdatesOnly() option.
watcher, _ := kv.Watch(ctx, "sue.*")
defer watcher.Stop()

kv.Put(ctx, "sue.age", []byte("43"))
kv.Put(ctx, "sue.color", []byte("red"))

// First, the watcher sends most recent values for all matching keys.
// In this case, it will send a single entry for `sue.color`.
entry := <-watcher.Updates()
// Prints `sue.color @ 1 -> "blue"`
fmt.Printf("%s @ %d -> %q\n", entry.Key(), entry.Revision(), string(entry.Value()))

// After all current values have been sent, watcher will send nil on the channel.
entry = <-watcher.Updates()
if entry != nil {
    fmt.Println("Unexpected entry received")
}

// After that, watcher will send updates when changes occur
// In this case, it will send an entry for `sue.color` and `sue.age`.

entry = <-watcher.Updates()
// Prints `sue.age @ 2 -> "43"`
fmt.Printf("%s @ %d -> %q\n", entry.Key(), entry.Revision(), string(entry.Value()))

entry = <-watcher.Updates()
// Prints `sue.color @ 3 -> "red"`
fmt.Printf("%s @ %d -> %q\n", entry.Key(), entry.Revision(), string(entry.Value()))
```

### Additional operations on a bucket

In addition to basic CRUD operations and watching for changes, KV buckets
support several additional operations:

- `ListKeys` will return all keys in a bucket

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
kv, _ := js.CreateKeyValue(ctx, jetstream.KeyValueConfig{Bucket: "profiles"})

kv.Put(ctx, "sue.color", []byte("blue"))
kv.Put(ctx, "sue.age", []byte("43"))
kv.Put(ctx, "bucket", []byte("profiles"))

keys, _ := kv.ListKeys(ctx)

// Prints all 3 keys
for key := range keys.Keys() {
    fmt.Println(key)
}
```

- `Purge` and `PurgeDeletes` for removing all keys from a bucket

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
kv, _ := js.CreateKeyValue(ctx, jetstream.KeyValueConfig{Bucket: "profiles"})

kv.Put(ctx, "sue.color", []byte("blue"))
kv.Put(ctx, "sue.age", []byte("43"))
kv.Put(ctx, "bucket", []byte("profiles"))

// Purge will remove all keys from a bucket.
// The latest revision of each key will be kept
// with a delete marker, all previous revisions will be removed
// permanently.
kv.Purge(ctx)

// PurgeDeletes will remove all keys from a bucket
// with a delete marker.
kv.PurgeDeletes(ctx)
```

- `Status` will return the current status of a bucket

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
kv, _ := js.CreateKeyValue(ctx, jetstream.KeyValueConfig{Bucket: "profiles"})

kv.Put(ctx, "sue.color", []byte("blue"))
kv.Put(ctx, "sue.age", []byte("43"))
kv.Put(ctx, "bucket", []byte("profiles"))

status, _ := kv.Status(ctx)

fmt.Println(status.Bucket()) // prints `profiles`
fmt.Println(status.Values()) // prints `3`
fmt.Println(status.Bytes()) // prints the size of all values in bytes
```

## Object Store

JetStream Object Stores offer a straightforward method for storing large objects
within JetStream. These stores are backed by a specially configured streams,
designed to efficiently and compactly store these objects.

The Object Store, also known as a bucket, enables the execution of various
operations:

- create/update an object
- get an object
- delete an object
- list all objects in a bucket
- watch for changes on objects in a bucket
- create links to other objects or other buckets

### Basic usage of Object Store

The most basic usage of Object bucket is to create or retrieve a bucket and
perform basic CRUD operations on objects.

```go
js, _ := jetstream.New(nc)
ctx := context.Background()

// Create a new bucket. Bucket name is required and has to be unique within a JetStream account.
os, _ := js.CreateObjectStore(ctx, jetstream.ObjectStoreConfig{Bucket: "configs"})

config1 := bytes.NewBufferString("first config")
// Put an object in a bucket. Put expects an object metadata and a reader
// to read the object data from.
os.Put(ctx, jetstream.ObjectMeta{Name: "config-1"}, config1)

// Objects can also be created using various helper methods

// 1. As raw strings
os.PutString(ctx, "config-2", "second config")

// 2. As raw bytes
os.PutBytes(ctx, "config-3", []byte("third config"))

// 3. As a file
os.PutFile(ctx, "config-4.txt")

// Get an object
// Get returns a reader and object info
// Similar to Put, Get can also be used with helper methods
// to retrieve object data as a string, bytes or to save it to a file
object, _ := os.Get(ctx, "config-1")
data, _ := io.ReadAll(object)
info, _ := object.Info()

// Prints `configs.config-1 -> "first config"`
fmt.Printf("%s.%s -> %q\n", info.Bucket, info.Name, string(data))

// Delete an object.
// Delete will remove object data from stream, but object metadata will be kept
// with a delete marker.
os.Delete(ctx, "config-1")

// getting a deleted object will return an error
_, err := os.Get(ctx, "config-1")
fmt.Println(err) // prints `nats: object not found`

// A bucket can be deleted once it is no longer needed
js.DeleteObjectStore(ctx, "configs")
```

### Watching for changes on a store

Object Stores support Watchers, which can be used to watch for changes on
objects in a given bucket. Watcher will receive a notification on a channel when
a change occurs. By default, watcher will return the latest information for all
objects in a bucket. After sending all initial values, watcher will send nil on
the channel to signal that all initial values have been sent and it will start
sending updates when changes occur.

>__NOTE:__ Watchers do not retrieve values for objects, only metadata (containing
>information such as object name, bucket name, object size etc.). If object data
>is required, the `Get` method should be used.

Watcher supports several configuration options:

- `IncludeHistory` will have the watcher send historical updates for each
  object.
- `IgnoreDeletes` will have the watcher not pass any objects with delete
  markers.
- `UpdatesOnly` will have the watcher only pass updates on objects (without
  objects already present when starting).

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
os, _ := js.CreateObjectStore(ctx, jetstream.ObjectStoreConfig{Bucket: "configs"})

os.PutString(ctx, "config-1", "first config")

// By default, watcher will return most recent values for all objects in a bucket.
// Watcher can be configured to only return updates by using jetstream.UpdatesOnly() option.
watcher, _ := os.Watch(ctx)
defer watcher.Stop()

// create a second object
os.PutString(ctx, "config-2", "second config")

// update metadata of the first object
os.UpdateMeta(ctx, "config-1", jetstream.ObjectMeta{Name: "config-1", Description: "updated config"})

// First, the watcher sends most recent values for all matching objects.
// In this case, it will send a single entry for `config-1`.
object := <-watcher.Updates()
// Prints `configs.config-1 -> ""`
fmt.Printf("%s.%s -> %q\n", object.Bucket, object.Name, object.Description)

// After all current values have been sent, watcher will send nil on the channel.
object = <-watcher.Updates()
if object != nil {
    fmt.Println("Unexpected object received")
}

// After that, watcher will send updates when changes occur
// In this case, it will send an entry for `config-2` and `config-1`.
object = <-watcher.Updates()
// Prints `configs.config-2 -> ""`
fmt.Printf("%s.%s -> %q\n", object.Bucket, object.Name, object.Description)

object = <-watcher.Updates()
// Prints `configs.config-1 -> "updated config"`
fmt.Printf("%s.%s -> %q\n", object.Bucket, object.Name, object.Description)
```

### Additional operations on a store

In addition to basic CRUD operations and watching for changes, Object Stores
support several additional operations:

- `UpdateMeta` for updating object metadata, such as name, description, etc.

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
os, _ := js.CreateObjectStore(ctx, jetstream.ObjectStoreConfig{Bucket: "configs"})

os.PutString(ctx, "config", "data")

// update metadata of the object to e.g. add a description
os.UpdateMeta(ctx, "config", jetstream.ObjectMeta{Name: "config", Description: "this is a config"})

// object can be moved under a new name (unless it already exists)
os.UpdateMeta(ctx, "config", jetstream.ObjectMeta{Name: "config-1", Description: "updated config"})
```

- `List` for listing information about all objects in a bucket:

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
os, _ := js.CreateObjectStore(ctx, jetstream.ObjectStoreConfig{Bucket: "configs"})

os.PutString(ctx, "config-1", "cfg1")
os.PutString(ctx, "config-2", "cfg1")
os.PutString(ctx, "config-3", "cfg1")

// List will return information about all objects in a bucket
objects, _ := os.List(ctx)

// Prints all 3 objects
for _, object := range objects {
    fmt.Println(object.Name)
}
```

- `Status` will return the current status of a bucket

```go
js, _ := jetstream.New(nc)
ctx := context.Background()
os, _ := js.CreateObjectStore(ctx, jetstream.ObjectStoreConfig{Bucket: "configs"})

os.PutString(ctx, "config-1", "cfg1")
os.PutString(ctx, "config-2", "cfg1")
os.PutString(ctx, "config-3", "cfg1")

status, _ := os.Status(ctx)

fmt.Println(status.Bucket()) // prints `configs`
fmt.Println(status.Size()) // prints the size of the bucket in bytes
```

## Examples

You can find more examples of `jetstream` usage [here](https://github.com/nats-io/nats.go/tree/main/examples/jetstream).
