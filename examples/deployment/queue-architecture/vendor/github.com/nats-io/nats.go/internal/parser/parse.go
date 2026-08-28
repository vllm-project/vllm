// Copyright 2020-2022 The NATS Authors
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

package parser

import (
	"errors"
	"fmt"
)

const (
	AckDomainTokenPos = iota + 2
	AckAccHashTokenPos
	AckStreamTokenPos
	AckConsumerTokenPos
	AckNumDeliveredTokenPos
	AckStreamSeqTokenPos
	AckConsumerSeqTokenPos
	AckTimestampSeqTokenPos
	AckNumPendingTokenPos
)

var ErrInvalidSubjectFormat = errors.New("invalid format of ACK subject")

// Quick parser for positive numbers in ack reply encoding.
// NOTE: This parser does not detect uint64 overflow
func ParseNum(d string) (n uint64) {
	if len(d) == 0 {
		return 0
	}

	// ASCII numbers 0-9
	const (
		asciiZero = 48
		asciiNine = 57
	)

	for _, dec := range d {
		if dec < asciiZero || dec > asciiNine {
			return 0
		}
		n = n*10 + uint64(dec) - asciiZero
	}
	return
}

func GetMetadataFields(subject string) ([]string, error) {
	v1TokenCounts, v2TokenCounts := 9, 12

	var start int
	tokens := make([]string, 0, v2TokenCounts)
	for i := 0; i < len(subject); i++ {
		if subject[i] == '.' {
			tokens = append(tokens, subject[start:i])
			start = i + 1
		}
	}
	tokens = append(tokens, subject[start:])
	//
	// Newer server will include the domain name and account hash in the subject,
	// and a token at the end.
	//
	// Old subject was:
	// $JS.ACK.<stream>.<consumer>.<delivered>.<sseq>.<cseq>.<tm>.<pending>
	//
	// New subject would be:
	// $JS.ACK.<domain>.<account hash>.<stream>.<consumer>.<delivered>.<sseq>.<cseq>.<tm>.<pending>.<a token with a random value>
	//
	// v1 has 9 tokens, v2 has 12, but we must not be strict on the 12th since
	// it may be removed in the future. Also, the library has no use for it.
	// The point is that a v2 ACK subject is valid if it has at least 11 tokens.
	//
	tokensLen := len(tokens)
	// If lower than 9 or more than 9 but less than 11, report an error
	if tokensLen < v1TokenCounts || (tokensLen > v1TokenCounts && tokensLen < v2TokenCounts-1) {
		return nil, ErrInvalidSubjectFormat
	}
	if tokens[0] != "$JS" || tokens[1] != "ACK" {
		return nil, fmt.Errorf("%w: subject should start with $JS.ACK", ErrInvalidSubjectFormat)
	}
	// For v1 style, we insert 2 empty tokens (domain and hash) so that the
	// rest of the library references known fields at a constant location.
	if tokensLen == v1TokenCounts {
		// Extend the array (we know the backend is big enough)
		tokens = append(tokens[:AckDomainTokenPos+2], tokens[AckDomainTokenPos:]...)
		// Clear the domain and hash tokens
		tokens[AckDomainTokenPos], tokens[AckAccHashTokenPos] = "", ""

	} else if tokens[AckDomainTokenPos] == "_" {
		// If domain is "_", replace with empty value.
		tokens[AckDomainTokenPos] = ""
	}
	return tokens, nil
}
