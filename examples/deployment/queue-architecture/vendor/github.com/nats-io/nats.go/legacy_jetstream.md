# Legacy JetStream API

This is a documentation for the legacy JetStream API. A README for the current
API can be found [here](jetstream/README.md)

## JetStream Basic Usage

```go
import "github.com/nats-io/nats.go"

// Connect to NATS
nc, _ := nats.Connect(nats.DefaultURL)

// Create JetStream Context
js, _ := nc.JetStream(nats.PublishAsyncMaxPending(256))

// Simple Stream Publisher
js.Publish("ORDERS.scratch", []byte("hello"))

// Simple Async Stream Publisher
for i := 0; i < 500; i++ {
    js.PublishAsync("ORDERS.scratch", []byte("hello"))
}
select {
case <-js.PublishAsyncComplete():
case <-time.After(5 * time.Second):
    fmt.Println("Did not resolve in time")
}

// Simple Async Ephemeral Consumer
js.Subscribe("ORDERS.*", func(m *nats.Msg) {
    fmt.Printf("Received a JetStream message: %s\n", string(m.Data))
})

// Simple Sync Durable Consumer (optional SubOpts at the end)
sub, err := js.SubscribeSync("ORDERS.*", nats.Durable("MONITOR"), nats.MaxDeliver(3))
m, err := sub.NextMsg(timeout)

// Simple Pull Consumer
sub, err := js.PullSubscribe("ORDERS.*", "MONITOR")
msgs, err := sub.Fetch(10)

// Unsubscribe
sub.Unsubscribe()

// Drain
sub.Drain()
```

## JetStream Basic Management

```go
import "github.com/nats-io/nats.go"

// Connect to NATS
nc, _ := nats.Connect(nats.DefaultURL)

// Create JetStream Context
js, _ := nc.JetStream()

// Create a Stream
js.AddStream(&nats.StreamConfig{
    Name:     "ORDERS",
    Subjects: []string{"ORDERS.*"},
})

// Update a Stream
js.UpdateStream(&nats.StreamConfig{
    Name:     "ORDERS",
    MaxBytes: 8,
})

// Create a Consumer
js.AddConsumer("ORDERS", &nats.ConsumerConfig{
    Durable: "MONITOR",
})

// Delete Consumer
js.DeleteConsumer("ORDERS", "MONITOR")

// Delete Stream
js.DeleteStream("ORDERS")
```
