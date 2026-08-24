# Universally Unique Lexicographically Sortable Identifier

[![Project status](https://img.shields.io/github/release/oklog/ulid.svg?style=flat-square)](https://github.com/oklog/ulid/releases/latest)
![Build Status](https://github.com/oklog/ulid/actions/workflows/test.yml/badge.svg)
[![Go Report Card](https://goreportcard.com/badge/oklog/ulid?cache=0)](https://goreportcard.com/report/oklog/ulid)
[![Coverage Status](https://coveralls.io/repos/github/oklog/ulid/badge.svg?branch=master&cache=0)](https://coveralls.io/github/oklog/ulid?branch=master)
[![go.dev reference](https://img.shields.io/badge/go.dev-reference-007d9c?logo=go&logoColor=white&style=flat-square)](https://pkg.go.dev/github.com/oklog/ulid/v2)
[![Apache 2 licensed](https://img.shields.io/badge/license-Apache2-blue.svg)](https://raw.githubusercontent.com/oklog/ulid/master/LICENSE)

A Go port of [ulid/javascript](https://github.com/ulid/javascript) with binary format implemented.

## Background

A GUID/UUID can be suboptimal for many use-cases because:

- It isn't the most character efficient way of encoding 128 bits
- UUID v1/v2 is impractical in many environments, as it requires access to a unique, stable MAC address
- UUID v3/v5 requires a unique seed and produces randomly distributed IDs, which can cause fragmentation in many data structures
- UUID v4 provides no other information than randomness which can cause fragmentation in many data structures

A ULID however:

- Is compatible with UUID/GUID's
- 1.21e+24 unique ULIDs per millisecond (1,208,925,819,614,629,174,706,176 to be exact)
- Lexicographically sortable
- Canonically encoded as a 26 character string, as opposed to the 36 character UUID
- Uses Crockford's base32 for better efficiency and readability (5 bits per character)
- Case insensitive
- No special characters (URL safe)
- Monotonic sort order (correctly detects and handles the same millisecond)

## Install

This package requires Go modules.

```shell
go get github.com/oklog/ulid/v2
```

## Usage

ULIDs are constructed from two things: a timestamp with millisecond precision,
and some random data.

Timestamps are modeled as uint64 values representing a Unix time in milliseconds.
They can be produced by passing a [time.Time](https://pkg.go.dev/time#Time) to
[ulid.Timestamp](https://pkg.go.dev/github.com/oklog/ulid/v2#Timestamp),
or by calling  [time.Time.UnixMilli](https://pkg.go.dev/time#Time.UnixMilli)
and converting the returned value to `uint64`.

Random data is taken from a provided [io.Reader](https://pkg.go.dev/io#Reader).
This design allows for greater flexibility when choosing trade-offs, but can be
a bit confusing to newcomers.

If you just want to generate a ULID and don't (yet) care about details like
performance, cryptographic security, etc., use the
[ulid.Make](https://pkg.go.dev/github.com/oklog/ulid/v2#Make) helper function.
This function calls [time.Now](https://pkg.go.dev/time#Now) to get a timestamp,
and uses a source of entropy which is process-global,
[pseudo-random](https://pkg.go.dev/math/rand), and
[monotonic](https://pkg.go.dev/github.com/oklog/ulid/v2#LockedMonotonicReader).

```go
fmt.Println(ulid.Make())
// 01G65Z755AFWAKHE12NY0CQ9FH
```

More advanced use cases should utilize
[ulid.New](https://pkg.go.dev/github.com/oklog/ulid/v2#New).

```go
entropy := rand.New(rand.NewSource(time.Now().UnixNano()))
ms := ulid.Timestamp(time.Now())
fmt.Println(ulid.New(ms, entropy))
// 01G65Z755AFWAKHE12NY0CQ9FH
```

Care should be taken when providing a source of entropy.

The above example utilizes [math/rand.Rand](https://pkg.go.dev/math/rand#Rand),
which is not safe for concurrent use by multiple goroutines. Consider
alternatives such as
[x/exp/rand](https://pkg.go.dev/golang.org/x/exp/rand#LockedSource).
Security-sensitive use cases should always use cryptographically secure entropy
provided by [crypto/rand](https://pkg.go.dev/crypto/rand).

Performance-sensitive use cases should avoid synchronization when generating
IDs. One option is to use a unique source of entropy for each concurrent
goroutine, which results in no lock contention, but cannot provide strong
guarantees about the random data, and does not provide monotonicity within a
given millisecond. One common performance optimization is to pool sources of
entropy using a [sync.Pool](https://pkg.go.dev/sync#Pool).

Monotonicity is a property that says each ULID is "bigger than" the previous
one. ULIDs are automatically monotonic, but only to millisecond precision. ULIDs
generated within the same millisecond are ordered by their random component,
which means they are by default un-ordered. You can use
[ulid.MonotonicEntropy](https://pkg.go.dev/github.com/oklog/ulid/v2#MonotonicEntropy) or
[ulid.LockedMonotonicEntropy](https://pkg.go.dev/github.com/oklog/ulid/v2#LockedMonotonicEntropy)
to create ULIDs that are monotonic within a given millisecond, with caveats. See
the documentation for details.

If you don't care about time-based ordering of generated IDs, then there's no
reason to use ULIDs! There are many other kinds of IDs that are easier, faster,
smaller, etc. Consider UUIDs.

## Commandline tool

This repo also provides a tool to generate and parse ULIDs at the command line.

```shell
go install github.com/oklog/ulid/v2/cmd/ulid@latest
```

Usage:

```shell
Usage: ulid [-hlqz] [-f <format>] [parameters ...]
 -f, --format=<format>  when parsing, show times in this format: default, rfc3339, unix, ms
 -h, --help             print this help text
 -l, --local            when parsing, show local time instead of UTC
 -q, --quick            when generating, use non-crypto-grade entropy
 -z, --zero             when generating, fix entropy to all-zeroes
```

Examples:

```shell
$ ulid
01D78XYFJ1PRM1WPBCBT3VHMNV
$ ulid -z
01D78XZ44G0000000000000000
$ ulid 01D78XZ44G0000000000000000
Sun Mar 31 03:51:23.536 UTC 2019
$ ulid --format=rfc3339 --local 01D78XZ44G0000000000000000
2019-03-31T05:51:23.536+02:00
```

## Specification

Below is the current specification of ULID as implemented in this repository.

### Components

**Timestamp**
- 48 bits
- UNIX-time in milliseconds
- Won't run out of space till the year 10889 AD

**Entropy**
- 80 bits
- User defined entropy source.
- Monotonicity within the same millisecond with [`ulid.Monotonic`](https://godoc.org/github.com/oklog/ulid#Monotonic)

### Encoding

[Crockford's Base32](http://www.crockford.com/wrmg/base32.html) is used as shown.
This alphabet excludes the letters I, L, O, and U to avoid confusion and abuse.

```
0123456789ABCDEFGHJKMNPQRSTVWXYZ
```

### Binary Layout and Byte Order

The components are encoded as 16 octets. Each component is encoded with the Most Significant Byte first (network byte order).

```
0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                      32_bit_uint_time_high                    |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|     16_bit_uint_time_low      |       16_bit_uint_random      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       32_bit_uint_random                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                       32_bit_uint_random                      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### String Representation

```
 01AN4Z07BY      79KA1307SR9X4MV3
|----------|    |----------------|
 Timestamp           Entropy
  10 chars           16 chars
   48bits             80bits
   base32             base32
```

## Test

```shell
go test ./...
```

## Benchmarks

On a Intel Core i7 Ivy Bridge 2.7 GHz, MacOS 10.12.1 and Go 1.8.0beta1

```
BenchmarkNew/WithCryptoEntropy-8      2000000        771 ns/op      20.73 MB/s   16 B/op   1 allocs/op
BenchmarkNew/WithEntropy-8            20000000      65.8 ns/op     243.01 MB/s   16 B/op   1 allocs/op
BenchmarkNew/WithoutEntropy-8         50000000      30.0 ns/op     534.06 MB/s   16 B/op   1 allocs/op
BenchmarkMustNew/WithCryptoEntropy-8  2000000        781 ns/op      20.48 MB/s   16 B/op   1 allocs/op
BenchmarkMustNew/WithEntropy-8        20000000      70.0 ns/op     228.51 MB/s   16 B/op   1 allocs/op
BenchmarkMustNew/WithoutEntropy-8     50000000      34.6 ns/op     462.98 MB/s   16 B/op   1 allocs/op
BenchmarkParse-8                      50000000      30.0 ns/op     866.16 MB/s    0 B/op   0 allocs/op
BenchmarkMustParse-8                  50000000      35.2 ns/op     738.94 MB/s    0 B/op   0 allocs/op
BenchmarkString-8                     20000000      64.9 ns/op     246.40 MB/s   32 B/op   1 allocs/op
BenchmarkMarshal/Text-8               20000000      55.8 ns/op     286.84 MB/s   32 B/op   1 allocs/op
BenchmarkMarshal/TextTo-8             100000000     22.4 ns/op     714.91 MB/s    0 B/op   0 allocs/op
BenchmarkMarshal/Binary-8             300000000     4.02 ns/op    3981.77 MB/s    0 B/op   0 allocs/op
BenchmarkMarshal/BinaryTo-8           2000000000    1.18 ns/op   13551.75 MB/s    0 B/op   0 allocs/op
BenchmarkUnmarshal/Text-8             100000000     20.5 ns/op    1265.27 MB/s    0 B/op   0 allocs/op
BenchmarkUnmarshal/Binary-8           300000000     4.94 ns/op    3240.01 MB/s    0 B/op   0 allocs/op
BenchmarkNow-8                        100000000     15.1 ns/op     528.09 MB/s    0 B/op   0 allocs/op
BenchmarkTimestamp-8                  2000000000    0.29 ns/op   27271.59 MB/s    0 B/op   0 allocs/op
BenchmarkTime-8                       2000000000    0.58 ns/op   13717.80 MB/s    0 B/op   0 allocs/op
BenchmarkSetTime-8                    2000000000    0.89 ns/op    9023.95 MB/s    0 B/op   0 allocs/op
BenchmarkEntropy-8                    200000000     7.62 ns/op    1311.66 MB/s    0 B/op   0 allocs/op
BenchmarkSetEntropy-8                 2000000000    0.88 ns/op   11376.54 MB/s    0 B/op   0 allocs/op
BenchmarkCompare-8                    200000000     7.34 ns/op    4359.23 MB/s    0 B/op   0 allocs/op
```

## Prior Art

- [ulid/javascript](https://github.com/ulid/javascript)
- [RobThree/NUlid](https://github.com/RobThree/NUlid)
- [imdario/go-ulid](https://github.com/imdario/go-ulid)
