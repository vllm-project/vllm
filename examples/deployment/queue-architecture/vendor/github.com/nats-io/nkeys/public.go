// Copyright 2018-2024 The NATS Authors
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

package nkeys

import (
	"crypto/ed25519"
	"crypto/rand"
	"io"
)

// A KeyPair from a public key capable of verifying only.
type pub struct {
	pre PrefixByte
	pub ed25519.PublicKey
}

// PublicKey will return the encoded public key associated with the KeyPair.
// All KeyPairs have a public key.
func (p *pub) PublicKey() (string, error) {
	pk, err := Encode(p.pre, p.pub)
	if err != nil {
		return "", err
	}
	return string(pk), nil
}

// Seed will return an error since this is not available for public key only KeyPairs.
func (p *pub) Seed() ([]byte, error) {
	return nil, ErrPublicKeyOnly
}

// PrivateKey will return an error since this is not available for public key only KeyPairs.
func (p *pub) PrivateKey() ([]byte, error) {
	return nil, ErrPublicKeyOnly
}

// Sign will return an error since this is not available for public key only KeyPairs.
func (p *pub) Sign(input []byte) ([]byte, error) {
	return nil, ErrCannotSign
}

// Verify will verify the input against a signature utilizing the public key.
func (p *pub) Verify(input []byte, sig []byte) error {
	if !ed25519.Verify(p.pub, input, sig) {
		return ErrInvalidSignature
	}
	return nil
}

// Wipe will randomize the public key and erase the pre byte.
func (p *pub) Wipe() {
	p.pre = '0'
	io.ReadFull(rand.Reader, p.pub)
}

func (p *pub) Seal(input []byte, recipient string) ([]byte, error) {
	if p.pre == PrefixByteCurve {
		return nil, ErrCannotSeal
	}
	return nil, ErrInvalidNKeyOperation
}
func (p *pub) SealWithRand(input []byte, _recipient string, rr io.Reader) ([]byte, error) {
	if p.pre == PrefixByteCurve {
		return nil, ErrCannotSeal
	}
	return nil, ErrInvalidNKeyOperation
}

func (p *pub) Open(input []byte, sender string) ([]byte, error) {
	if p.pre == PrefixByteCurve {
		return nil, ErrCannotOpen
	}
	return nil, ErrInvalidNKeyOperation
}
