# NKEYS

[![License Apache 2](https://img.shields.io/badge/License-Apache2-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Go Report Card](https://goreportcard.com/badge/github.com/nats-io/nkeys)](https://goreportcard.com/report/github.com/nats-io/nkeys)
[![Build Status](https://github.com/nats-io/nkeys/actions/workflows/release.yaml/badge.svg)](https://github.com/nats-io/nkeys/actions/workflows/release.yaml/badge.svg)
[![GoDoc](https://godoc.org/github.com/nats-io/nkeys?status.svg)](https://godoc.org/github.com/nats-io/nkeys)
[![Coverage Status](https://coveralls.io/repos/github/nats-io/nkeys/badge.svg?branch=main&service=github)](https://coveralls.io/github/nats-io/nkeys?branch=main)

A public-key signature system based on [Ed25519](https://ed25519.cr.yp.to/) for the NATS ecosystem.

## About

The NATS ecosystem will be moving to [Ed25519](https://ed25519.cr.yp.to/) keys for identity, authentication and authorization for entities such as Accounts, Users, Servers and Clusters.

Ed25519 is fast and resistant to side channel attacks. Generation of a seed key is all that is needed to be stored and kept safe, as the seed can generate both the public and private keys.

The NATS system will utilize Ed25519 keys, meaning that NATS systems will never store or even have access to any private keys. Authentication will utilize a random challenge response mechanism.

Dealing with 32 byte and 64 byte raw keys can be challenging. NKEYS is designed to formulate keys in a much friendlier fashion and references work done in cryptocurrencies, specifically [Stellar](https://www.stellar.org/).	Bitcoin and others used a form of Base58 (or Base58Check) to encode raw keys. Stellar utilized a more traditional Base32 with a CRC16 and a version or prefix byte. NKEYS utilizes a similar format where the prefix will be 1 byte for public and private keys and will be 2 bytes for seeds. The base32 encoding of these prefixes will yield friendly human readable prefixes, e.g. '**N**' = server, '**C**' = cluster, '**O**' = operator, '**A**' = account, and '**U**' = user. '**P**' is used for private keys. For seeds, the first encoded prefix is '**S**', and the second character will be the type for the public key, e.g. "**SU**" is a seed for a user key pair, "**SA**" is a seed for an account key pair.

## Installation

Use the `go` command:

	$ go get github.com/nats-io/nkeys

## nk - Command Line Utility

Located under the nk [directory](https://github.com/nats-io/nkeys/tree/master/nk).

## Basic API Usage
```go

// Create a new User KeyPair
user, _ := nkeys.CreateUser()

// Sign some data with a full key pair user.
data := []byte("Hello World")
sig, _ := user.Sign(data)

// Verify the signature.
err = user.Verify(data, sig)

// Access the seed, the only thing that needs to be stored and kept safe.
// seed = "SUAKYRHVIOREXV7EUZTBHUHL7NUMHPMAS7QMDU3GTIUWEI5LDNOXD43IZY"
seed, _ := user.Seed()

// Access the public key which can be shared.
// publicKey = "UD466L6EBCM3YY5HEGHJANNTN4LSKTSUXTH7RILHCKEQMQHTBNLHJJXT"
publicKey, _ := user.PublicKey()

// Create a full User who can sign and verify from a private seed.
user, _ = nkeys.FromSeed(seed)

// Create a User who can only verify signatures via a public key.
user, _ = nkeys.FromPublicKey(publicKey)

// Create a User KeyPair with our own random data.
var rawSeed [32]byte
_, err := io.ReadFull(rand.Reader, rawSeed[:])  // Or some other random source.
user2, _ := nkeys.FromRawSeed(PrefixByteUser, rawSeed)

```

## License

Unless otherwise noted, the NATS source files are distributed
under the Apache Version 2.0 license found in the LICENSE file.
