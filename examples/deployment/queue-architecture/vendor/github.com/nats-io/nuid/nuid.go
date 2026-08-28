// Copyright 2016-2019 The NATS Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A unique identifier generator that is high performance, very fast, and tries to be entropy pool friendly.
package nuid

import (
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"sync"
	"time"

	prand "math/rand"
)

// NUID needs to be very fast to generate and truly unique, all while being entropy pool friendly.
// We will use 12 bytes of crypto generated data (entropy draining), and 10 bytes of sequential data
// that is started at a pseudo random number and increments with a pseudo-random increment.
// Total is 22 bytes of base 62 ascii text :)

// Version of the library
const Version = "1.0.1"

const (
	digits   = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	base     = 62
	preLen   = 12
	seqLen   = 10
	maxSeq   = int64(839299365868340224) // base^seqLen == 62^10
	minInc   = int64(33)
	maxInc   = int64(333)
	totalLen = preLen + seqLen
)

type NUID struct {
	pre []byte
	seq int64
	inc int64
}

type lockedNUID struct {
	sync.Mutex
	*NUID
}

// Global NUID
var globalNUID *lockedNUID

// Seed sequential random with crypto or math/random and current time
// and generate crypto prefix.
func init() {
	r, err := rand.Int(rand.Reader, big.NewInt(math.MaxInt64))
	if err != nil {
		prand.Seed(time.Now().UnixNano())
	} else {
		prand.Seed(r.Int64())
	}
	globalNUID = &lockedNUID{NUID: New()}
	globalNUID.RandomizePrefix()
}

// New will generate a new NUID and properly initialize the prefix, sequential start, and sequential increment.
func New() *NUID {
	n := &NUID{
		seq: prand.Int63n(maxSeq),
		inc: minInc + prand.Int63n(maxInc-minInc),
		pre: make([]byte, preLen),
	}
	n.RandomizePrefix()
	return n
}

// Generate the next NUID string from the global locked NUID instance.
func Next() string {
	globalNUID.Lock()
	nuid := globalNUID.Next()
	globalNUID.Unlock()
	return nuid
}

// Generate the next NUID string.
func (n *NUID) Next() string {
	// Increment and capture.
	n.seq += n.inc
	if n.seq >= maxSeq {
		n.RandomizePrefix()
		n.resetSequential()
	}
	seq := n.seq

	// Copy prefix
	var b [totalLen]byte
	bs := b[:preLen]
	copy(bs, n.pre)

	// copy in the seq in base62.
	for i, l := len(b), seq; i > preLen; l /= base {
		i -= 1
		b[i] = digits[l%base]
	}
	return string(b[:])
}

// Resets the sequential portion of the NUID.
func (n *NUID) resetSequential() {
	n.seq = prand.Int63n(maxSeq)
	n.inc = minInc + prand.Int63n(maxInc-minInc)
}

// Generate a new prefix from crypto/rand.
// This call *can* drain entropy and will be called automatically when we exhaust the sequential range.
// Will panic if it gets an error from rand.Int()
func (n *NUID) RandomizePrefix() {
	var cb [preLen]byte
	cbs := cb[:]
	if nb, err := rand.Read(cbs); nb != preLen || err != nil {
		panic(fmt.Sprintf("nuid: failed generating crypto random number: %v\n", err))
	}

	for i := 0; i < preLen; i++ {
		n.pre[i] = digits[int(cbs[i])%base]
	}
}
