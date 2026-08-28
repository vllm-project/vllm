// Copyright 2022 The NATS Authors
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

// Errors
const (
	ErrInvalidPrefixByte        = nkeysError("nkeys: invalid prefix byte")
	ErrInvalidKey               = nkeysError("nkeys: invalid key")
	ErrInvalidPublicKey         = nkeysError("nkeys: invalid public key")
	ErrInvalidPrivateKey        = nkeysError("nkeys: invalid private key")
	ErrInvalidSeedLen           = nkeysError("nkeys: invalid seed length")
	ErrInvalidSeed              = nkeysError("nkeys: invalid seed")
	ErrInvalidEncoding          = nkeysError("nkeys: invalid encoded key")
	ErrInvalidSignature         = nkeysError("nkeys: signature verification failed")
	ErrCannotSign               = nkeysError("nkeys: can not sign, no private key available")
	ErrPublicKeyOnly            = nkeysError("nkeys: no seed or private key available")
	ErrIncompatibleKey          = nkeysError("nkeys: incompatible key")
	ErrInvalidChecksum          = nkeysError("nkeys: invalid checksum")
	ErrNoSeedFound              = nkeysError("nkeys: no nkey seed found")
	ErrInvalidNkeySeed          = nkeysError("nkeys: doesn't contain a seed nkey")
	ErrInvalidUserSeed          = nkeysError("nkeys: doesn't contain an user seed nkey")
	ErrInvalidRecipient         = nkeysError("nkeys: not a valid recipient public curve key")
	ErrInvalidSender            = nkeysError("nkeys: not a valid sender public curve key")
	ErrInvalidCurveKey          = nkeysError("nkeys: not a valid curve key")
	ErrInvalidCurveSeed         = nkeysError("nkeys: not a valid curve seed")
	ErrInvalidEncrypted         = nkeysError("nkeys: encrypted input is not valid")
	ErrInvalidEncVersion        = nkeysError("nkeys: encrypted input wrong version")
	ErrCouldNotDecrypt          = nkeysError("nkeys: could not decrypt input")
	ErrInvalidCurveKeyOperation = nkeysError("nkeys: curve key is not valid for sign/verify")
	ErrInvalidNKeyOperation     = nkeysError("nkeys: only curve key can seal/open")
	ErrCannotOpen               = nkeysError("nkeys: cannot open no private curve key available")
	ErrCannotSeal               = nkeysError("nkeys: cannot seal no private curve key available")
)

type nkeysError string

func (e nkeysError) Error() string {
	return string(e)
}
