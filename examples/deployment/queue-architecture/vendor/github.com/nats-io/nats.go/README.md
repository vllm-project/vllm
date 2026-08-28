# NATS - Go Client
A [Go](http://golang.org) client for the [NATS messaging system](https://nats.io).

[![License Apache 2][License-Image]][License-Url] [![Go Report Card][ReportCard-Image]][ReportCard-Url] [![Build Status][Build-Status-Image]][Build-Status-Url] [![GoDoc][GoDoc-Image]][GoDoc-Url] [![Coverage Status][Coverage-image]][Coverage-Url]

[License-Url]: https://www.apache.org/licenses/LICENSE-2.0
[License-Image]: https://img.shields.io/badge/License-Apache2-blue.svg
[ReportCard-Url]: https://goreportcard.com/report/github.com/nats-io/nats.go
[ReportCard-Image]: https://goreportcard.com/badge/github.com/nats-io/nats.go
[Build-Status-Url]: https://github.com/nats-io/nats.go/actions/workflows/ci.yaml?query=event%3Arelease
[Build-Status-Image]: https://github.com/nats-io/nats.go/actions/workflows/ci.yaml/badge.svg?event=release
[GoDoc-Url]: https://pkg.go.dev/github.com/nats-io/nats.go
[GoDoc-Image]: https://img.shields.io/badge/GoDoc-reference-007d9c
[Coverage-Url]: https://coveralls.io/r/nats-io/nats.go?branch=main
[Coverage-image]: https://coveralls.io/repos/github/nats-io/nats.go/badge.svg?branch=main

**Check out [NATS by example](https://natsbyexample.com) - An evolving collection of runnable, cross-client reference examples for NATS.**

## Installation

```bash
# To get the latest released Go client:
go get github.com/nats-io/nats.go@latest

# To get a specific version:
go get github.com/nats-io/nats.go@v1.53.1

# Note that the latest major version for NATS Server is v2:
go get github.com/nats-io/nats-server/v2@latest
```

## Basic Usage

```go
import "github.com/nats-io/nats.go"

// Connect to a server
nc, _ := nats.Connect(nats.DefaultURL)

// Simple Publisher
nc.Publish("foo", []byte("Hello World"))

// Simple Async Subscriber
nc.Subscribe("foo", func(m *nats.Msg) {
    fmt.Printf("Received a message: %s\n", string(m.Data))
})

// Responding to a request message
nc.Subscribe("request", func(m *nats.Msg) {
    m.Respond([]byte("answer is 42"))
})

// Simple Sync Subscriber
sub, err := nc.SubscribeSync("foo")
m, err := sub.NextMsg(timeout)

// Channel Subscriber
ch := make(chan *nats.Msg, 64)
sub, err := nc.ChanSubscribe("foo", ch)
msg := <- ch

// Unsubscribe
sub.Unsubscribe()

// Drain
sub.Drain()

// Requests
msg, err := nc.Request("help", []byte("help me"), 10*time.Millisecond)

// Replies
nc.Subscribe("help", func(m *nats.Msg) {
    nc.Publish(m.Reply, []byte("I can help!"))
})

// Drain connection (Preferred for responders)
// Close() not needed if this is called.
nc.Drain()

// Close connection
nc.Close()
```

## JetStream
[![JetStream API Reference](https://pkg.go.dev/badge/github.com/nats-io/nats.go/jetstream.svg)](https://pkg.go.dev/github.com/nats-io/nats.go/jetstream)

JetStream is the built-in NATS persistence system. `nats.go` provides a built-in
API enabling both managing JetStream assets as well as publishing/consuming
persistent messages.


### Basic usage

```go
// connect to nats server
nc, _ := nats.Connect(nats.DefaultURL)

// create jetstream context from nats connection
js, _ := jetstream.New(nc)

ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
defer cancel()

// get existing stream handle
stream, _ := js.Stream(ctx, "foo")

// retrieve consumer handle from a stream
cons, _ := stream.Consumer(ctx, "cons")

// consume messages from the consumer in callback
cc, _ := cons.Consume(func(msg jetstream.Msg) {
    fmt.Println("Received jetstream message: ", string(msg.Data()))
    msg.Ack()
})
defer cc.Stop()
```

To find more information on `nats.go` JetStream API, visit
[`jetstream/README.md`](jetstream/README.md)

> The current JetStream API replaces the [legacy JetStream API](legacy_jetstream.md)

## Service API

The service API (`micro`) allows you to [easily build NATS services](micro/README.md) The
services API is currently in beta release.

## New Authentication (Nkeys and User Credentials)
This requires server with version >= 2.0.0

NATS servers have a new security and authentication mechanism to authenticate with user credentials and Nkeys.
The simplest form is to use the helper method UserCredentials(credsFilepath).
```go
nc, err := nats.Connect(url, nats.UserCredentials("user.creds"))
```

The helper method creates two callback handlers to present the user JWT and sign the nonce challenge from the server.
The core client library never has direct access to your private key and simply performs the callback for signing the server challenge.
The helper will load and wipe and erase memory it uses for each connect or reconnect.

The helper also can take two entries, one for the JWT and one for the NKey seed file.
```go
nc, err := nats.Connect(url, nats.UserCredentials("user.jwt", "user.nk"))
```

You can also set the callback handlers directly and manage challenge signing directly.
```go
nc, err := nats.Connect(url, nats.UserJWT(jwtCB, sigCB))
```

Bare Nkeys are also supported. The nkey seed should be in a read only file, e.g. seed.txt
```bash
> cat seed.txt
# This is my seed nkey!
SUAGMJH5XLGZKQQWAWKRZJIGMOU4HPFUYLXJMXOO5NLFEO2OOQJ5LPRDPM
```

This is a helper function which will load and decode and do the proper signing for the server nonce.
It will clear memory in between invocations.
You can choose to use the low level option and provide the public key and a signature callback on your own.

```go
opt, err := nats.NkeyOptionFromSeed("seed.txt")
nc, err := nats.Connect(serverUrl, opt)

// Direct
nc, err := nats.Connect(serverUrl, nats.Nkey(pubNkey, sigCB))
```

## TLS

```go
// tls as a scheme will enable secure connections by default. This will also verify the server name.
nc, err := nats.Connect("tls://nats.demo.io:4443")

// If you are using a self-signed certificate, you need to have a tls.Config with RootCAs setup.
// We provide a helper method to make this case easier.
nc, err = nats.Connect("tls://localhost:4443", nats.RootCAs("./configs/certs/ca.pem"))

// If the server requires client certificate, there is a helper function for that too:
cert := nats.ClientCert("./configs/certs/client-cert.pem", "./configs/certs/client-key.pem")
nc, err = nats.Connect("tls://localhost:4443", cert)

// You can also supply a complete tls.Config

certFile := "./configs/certs/client-cert.pem"
keyFile := "./configs/certs/client-key.pem"
cert, err := tls.LoadX509KeyPair(certFile, keyFile)
if err != nil {
    t.Fatalf("error parsing X509 certificate/key pair: %v", err)
}

config := &tls.Config{
    ServerName: 	opts.Host,
    Certificates: 	[]tls.Certificate{cert},
    RootCAs:    	pool,
    MinVersion: 	tls.VersionTLS12,
}

nc, err = nats.Connect("nats://localhost:4443", nats.Secure(config))
if err != nil {
	t.Fatalf("Got an error on Connect with Secure Options: %+v\n", err)
}

```

## Wildcard Subscriptions

```go

// "*" matches any token, at any level of the subject.
nc.Subscribe("foo.*.baz", func(m *Msg) {
    fmt.Printf("Msg received on [%s] : %s\n", m.Subject, string(m.Data))
})

nc.Subscribe("foo.bar.*", func(m *Msg) {
    fmt.Printf("Msg received on [%s] : %s\n", m.Subject, string(m.Data))
})

// ">" matches any length of the tail of a subject, and can only be the last token
// E.g. 'foo.>' will match 'foo.bar', 'foo.bar.baz', 'foo.foo.bar.bax.22'
nc.Subscribe("foo.>", func(m *Msg) {
    fmt.Printf("Msg received on [%s] : %s\n", m.Subject, string(m.Data))
})

// Matches all of the above
nc.Publish("foo.bar.baz", []byte("Hello World"))

```

## Queue Groups

```go
// All subscriptions with the same queue name will form a queue group.
// Each message will be delivered to only one subscriber per queue group,
// using queuing semantics. You can have as many queue groups as you wish.
// Normal subscribers will continue to work as expected.

nc.QueueSubscribe("foo", "job_workers", func(_ *Msg) {
  received += 1
})
```

## Advanced Usage

```go

// Normally, the library will return an error when trying to connect and
// there is no server running. The RetryOnFailedConnect option will set
// the connection in reconnecting state if it failed to connect right away.
nc, err := nats.Connect(nats.DefaultURL,
    nats.RetryOnFailedConnect(true),
    nats.MaxReconnects(10),
    nats.ReconnectWait(time.Second),
    nats.ReconnectHandler(func(_ *nats.Conn) {
        // Note that this will be invoked for the first asynchronous connect.
    }))
if err != nil {
    // Should not return an error even if it can't connect, but you still
    // need to check in case there are some configuration errors.
}

// Flush connection to server, returns when all messages have been processed.
nc.Flush()
fmt.Println("All clear!")

// FlushTimeout specifies a timeout value as well.
err := nc.FlushTimeout(1*time.Second)
if err != nil {
    fmt.Println("Flushed timed out!")
} else {
    fmt.Println("All clear!")
}

// Auto-unsubscribe after MAX_WANTED messages received
const MAX_WANTED = 10
sub, err := nc.Subscribe("foo")
sub.AutoUnsubscribe(MAX_WANTED)

// Multiple connections
nc1 := nats.Connect("nats://host1:4222")
nc2 := nats.Connect("nats://host2:4222")

nc1.Subscribe("foo", func(m *Msg) {
    fmt.Printf("Received a message: %s\n", string(m.Data))
})

nc2.Publish("foo", []byte("Hello World!"))

```

## Clustered Usage

```go

var servers = "nats://localhost:1222, nats://localhost:1223, nats://localhost:1224"

nc, err := nats.Connect(servers)

// Optionally set ReconnectWait and MaxReconnect attempts.
// This example means 10 seconds total per backend.
nc, err = nats.Connect(servers, nats.MaxReconnects(5), nats.ReconnectWait(2 * time.Second))

// You can also add some jitter for the reconnection.
// This call will add up to 500 milliseconds for non TLS connections and 2 seconds for TLS connections.
// If not specified, the library defaults to 100 milliseconds and 1 second, respectively.
nc, err = nats.Connect(servers, nats.ReconnectJitter(500*time.Millisecond, 2*time.Second))

// You can also specify a custom reconnect delay handler. If set, the library will invoke it when it has tried
// all URLs in its list. The value returned will be used as the total sleep time, so add your own jitter.
// The library will pass the number of times it went through the whole list.
nc, err = nats.Connect(servers, nats.CustomReconnectDelay(func(attempts int) time.Duration {
    return someBackoffFunction(attempts)
}))

// Optionally disable randomization of the server pool
nc, err = nats.Connect(servers, nats.DontRandomize())

// Setup callbacks to be notified on disconnects, reconnects and connection closed.
nc, err = nats.Connect(servers,
	nats.DisconnectErrHandler(func(nc *nats.Conn, err error) {
		fmt.Printf("Got disconnected! Reason: %q\n", err)
	}),
	nats.ReconnectHandler(func(nc *nats.Conn) {
		fmt.Printf("Got reconnected to %v!\n", nc.ConnectedUrl())
	}),
	nats.ClosedHandler(func(nc *nats.Conn) {
		fmt.Printf("Connection closed. Reason: %q\n", nc.LastError())
	})
)

// When connecting to a mesh of servers with auto-discovery capabilities,
// you may need to provide a username/password or token in order to connect
// to any server in that mesh when authentication is required.
// Instead of providing the credentials in the initial URL, you will use
// new option setters:
nc, err = nats.Connect("nats://localhost:4222", nats.UserInfo("foo", "bar"))

// For token based authentication:
nc, err = nats.Connect("nats://localhost:4222", nats.Token("S3cretT0ken"))

// You can even pass the two at the same time in case one of the servers
// in the mesh requires token instead of user name and password.
nc, err = nats.Connect("nats://localhost:4222",
    nats.UserInfo("foo", "bar"),
    nats.Token("S3cretT0ken"))

// Note that if credentials are specified in the initial URLs, they take
// precedence on the credentials specified through the options.
// For instance, in the connect call below, the client library will use
// the user "my" and password "pwd" to connect to localhost:4222, however,
// it will use username "foo" and password "bar" when (re)connecting to
// a different server URL that it got as part of the auto-discovery.
nc, err = nats.Connect("nats://my:pwd@localhost:4222", nats.UserInfo("foo", "bar"))

```

## Context support (+Go 1.7)

```go
ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
defer cancel()

nc, err := nats.Connect(nats.DefaultURL)

// Request with context
msg, err := nc.RequestWithContext(ctx, "foo", []byte("bar"))

// Synchronous subscriber with context
sub, err := nc.SubscribeSync("foo")
msg, err := sub.NextMsgWithContext(ctx)

```

## Backward compatibility

In the development of nats.go, we are committed to maintaining backward compatibility and ensuring a stable and reliable  experience for all users. In general, we follow the standard go compatibility guidelines.
However, it's important to clarify our stance on certain types of changes:

- **Expanding structures:**
Adding new fields to structs is not considered a breaking change.

- **Adding methods to exported interfaces:**
Extending public interfaces with new methods is also not viewed as a breaking change within the context of this project. It is important to note that no unexported methods will be added to interfaces allowing users to implement them.

Additionally, this library always supports at least 2 latest minor Go versions. For example, if the latest Go version is 1.22, the library will support Go 1.21 and 1.22.

## Testing

See [TESTING.md](TESTING.md) for how to run the test suite locally.

## License

Unless otherwise noted, the NATS source files are distributed
under the Apache Version 2.0 license found in the LICENSE file.

[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fnats-io%2Fgo-nats.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fnats-io%2Fgo-nats?ref=badge_large)
