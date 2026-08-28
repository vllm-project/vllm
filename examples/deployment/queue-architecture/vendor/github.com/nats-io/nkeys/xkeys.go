// Copyright 2022-2024 The NATS Authors
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
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/binary"
	"io"

	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/nacl/box"
)

// This package will support safe use of X25519 keys for asymmetric encryption.
// We will be compatible with nacl.Box, but generate random nonces automatically.
// We may add more advanced options in the future for group recipients and better
// end to end algorithms.

const (
	curveKeyLen    = 32
	curveDecodeLen = 35
	curveNonceLen  = 24
)

type ckp struct {
	seed [curveKeyLen]byte // Private raw key.
}

// CreateCurveKeys will create a Curve typed KeyPair.
func CreateCurveKeys() (KeyPair, error) {
	return CreateCurveKeysWithRand(nil)
}

// CreateCurveKeysWithRand will create a Curve typed KeyPair
// with specified rand source.
func CreateCurveKeysWithRand(rr io.Reader) (KeyPair, error) {
	var kp ckp
	_, priv, err := ed25519.GenerateKey(rr)
	if err != nil {
		return nil, err
	}
	kp.seed = [curveKeyLen]byte(priv.Seed())
	return &kp, nil
}

// Will create a curve key pair from seed.
func FromCurveSeed(seed []byte) (KeyPair, error) {
	pb, raw, err := DecodeSeed(seed)
	if err != nil {
		return nil, err
	}
	if pb != PrefixByteCurve || len(raw) != curveKeyLen {
		return nil, ErrInvalidCurveSeed
	}
	var kp ckp
	copy(kp.seed[:], raw)
	return &kp, nil
}

// Seed will return the encoded seed.
func (pair *ckp) Seed() ([]byte, error) {
	return EncodeSeed(PrefixByteCurve, pair.seed[:])
}

// PublicKey will return the encoded public key.
func (pair *ckp) PublicKey() (string, error) {
	var pub [curveKeyLen]byte
	curve25519.ScalarBaseMult(&pub, &pair.seed)
	key, err := Encode(PrefixByteCurve, pub[:])
	return string(key), err
}

// PrivateKey will return the encoded private key.
func (pair *ckp) PrivateKey() ([]byte, error) {
	return Encode(PrefixBytePrivate, pair.seed[:])
}

func decodePubCurveKey(src string, dest []byte) error {
	var raw [curveDecodeLen]byte // should always be 35
	n, err := b32Enc.Decode(raw[:], []byte(src))
	if err != nil {
		return err
	}
	if n != curveDecodeLen {
		return ErrInvalidCurveKey
	}
	// Make sure it is what we expected.
	if prefix := PrefixByte(raw[0]); prefix != PrefixByteCurve {
		return ErrInvalidPublicKey
	}
	var crc uint16
	end := n - 2
	sum := raw[end:n]
	checksum := bytes.NewReader(sum)
	if err := binary.Read(checksum, binary.LittleEndian, &crc); err != nil {
		return err
	}

	// ensure checksum is valid
	if err := validate(raw[:end], crc); err != nil {
		return err
	}

	// Copy over, ignore prefix byte.
	copy(dest, raw[1:end])
	return nil
}

// Only version for now, but could add in X3DH in the future, etc.
const XKeyVersionV1 = "xkv1"
const vlen = len(XKeyVersionV1)

// Seal is compatible with nacl.Box.Seal() and can be used in similar situations for small messages.
// We generate the nonce from crypto rand by default.
func (pair *ckp) Seal(input []byte, recipient string) ([]byte, error) {
	return pair.SealWithRand(input, recipient, rand.Reader)
}

func (pair *ckp) SealWithRand(input []byte, recipient string, rr io.Reader) ([]byte, error) {
	var (
		rpub  [curveKeyLen]byte
		nonce [curveNonceLen]byte
		out   [vlen + curveNonceLen]byte
		err   error
	)

	if err = decodePubCurveKey(recipient, rpub[:]); err != nil {
		return nil, ErrInvalidRecipient
	}
	if _, err := io.ReadFull(rr, nonce[:]); err != nil {
		return nil, err
	}
	copy(out[:vlen], []byte(XKeyVersionV1))
	copy(out[vlen:], nonce[:])
	return box.Seal(out[:], input, &nonce, &rpub, &pair.seed), nil
}

func (pair *ckp) Open(input []byte, sender string) ([]byte, error) {
	if len(input) <= vlen+curveNonceLen {
		return nil, ErrInvalidEncrypted
	}
	var (
		spub  [curveKeyLen]byte
		nonce [curveNonceLen]byte
		err   error
	)
	if !bytes.Equal(input[:vlen], []byte(XKeyVersionV1)) {
		return nil, ErrInvalidEncVersion
	}
	copy(nonce[:], input[vlen:vlen+curveNonceLen])

	if err = decodePubCurveKey(sender, spub[:]); err != nil {
		return nil, ErrInvalidSender
	}

	decrypted, ok := box.Open(nil, input[vlen+curveNonceLen:], &nonce, &spub, &pair.seed)
	if !ok {
		return nil, ErrCouldNotDecrypt
	}
	return decrypted, nil
}

// Wipe will randomize the contents of the secret key
func (pair *ckp) Wipe() {
	io.ReadFull(rand.Reader, pair.seed[:])
}

func (pair *ckp) Sign(_ []byte) ([]byte, error) {
	return nil, ErrInvalidCurveKeyOperation
}

func (pair *ckp) Verify(_ []byte, _ []byte) error {
	return ErrInvalidCurveKeyOperation
}
