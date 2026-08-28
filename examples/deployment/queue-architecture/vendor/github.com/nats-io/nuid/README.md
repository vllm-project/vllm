# NUID

[![License Apache 2](https://img.shields.io/badge/License-Apache2-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![ReportCard](http://goreportcard.com/badge/nats-io/nuid)](http://goreportcard.com/report/nats-io/nuid)
[![Build Status](https://travis-ci.org/nats-io/nuid.svg?branch=master)](http://travis-ci.org/nats-io/nuid)
[![Release](https://img.shields.io/badge/release-v1.0.1-1eb0fc.svg)](https://github.com/nats-io/nuid/releases/tag/v1.0.1)
[![GoDoc](http://godoc.org/github.com/nats-io/nuid?status.png)](http://godoc.org/github.com/nats-io/nuid)
[![Coverage Status](https://coveralls.io/repos/github/nats-io/nuid/badge.svg?branch=master)](https://coveralls.io/github/nats-io/nuid?branch=master)

A highly performant unique identifier generator.

## Installation

Use the `go` command:

	$ go get github.com/nats-io/nuid

## Basic Usage
```go

// Utilize the global locked instance
nuid := nuid.Next()

// Create an instance, these are not locked.
n := nuid.New()
nuid = n.Next()

// Generate a new crypto/rand seeded prefix.
// Generally not needed, happens automatically.
n.RandomizePrefix()
```

## Performance
NUID needs to be very fast to generate and be truly unique, all while being entropy pool friendly.
NUID uses 12 bytes of crypto generated data (entropy draining), and 10 bytes of pseudo-random
sequential data that increments with a pseudo-random increment.

Total length of a NUID string is 22 bytes of base 62 ascii text, so 62^22 or
2707803647802660400290261537185326956544 possibilities.

NUID can generate identifiers as fast as 60ns, or ~16 million per second. There is an associated
benchmark you can use to test performance on your own hardware.

## License

Unless otherwise noted, the NATS source files are distributed
under the Apache Version 2.0 license found in the LICENSE file.
