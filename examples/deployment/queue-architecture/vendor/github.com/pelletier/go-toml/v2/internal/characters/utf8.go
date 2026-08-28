package characters

import (
	"unicode/utf8"
)

type utf8Err struct {
	Index int
	Size  int
}

func (u utf8Err) Zero() bool {
	return u.Size == 0
}

// Verified that a given string is only made of valid UTF-8 characters allowed
// by the TOML spec:
//
// Any Unicode character may be used except those that must be escaped:
// quotation mark, backslash, and the control characters other than tab (U+0000
// to U+0008, U+000A to U+001F, U+007F).
//
// It is a copy of the Go 1.17 utf8.Valid implementation, tweaked to exit early
// when a character is not allowed.
//
// The returned utf8Err is Zero() if the string is valid, or contains the byte
// index and size of the invalid character.
//
// quotation mark => already checked
// backslash => already checked
// 0-0x8 => invalid
// 0x9 => tab, ok
// 0xA - 0x1F => invalid
// 0x7F => invalid
func Utf8TomlValidAlreadyEscaped(p []byte) (err utf8Err) {
	// Fast path. Check for and skip 8 bytes of ASCII characters per iteration.
	offset := 0
	for len(p) >= 8 {
		// Combining two 32 bit loads allows the same code to be used
		// for 32 and 64 bit platforms.
		// The compiler can generate a 32bit load for first32 and second32
		// on many platforms. See test/codegen/memcombine.go.
		first32 := uint32(p[0]) | uint32(p[1])<<8 | uint32(p[2])<<16 | uint32(p[3])<<24
		second32 := uint32(p[4]) | uint32(p[5])<<8 | uint32(p[6])<<16 | uint32(p[7])<<24
		if (first32|second32)&0x80808080 != 0 {
			// Found a non ASCII byte (>= RuneSelf).
			break
		}

		for i, b := range p[:8] {
			if InvalidAscii(b) {
				err.Index = offset + i
				err.Size = 1
				return
			}
		}

		p = p[8:]
		offset += 8
	}
	n := len(p)
	for i := 0; i < n; {
		pi := p[i]
		if pi < utf8.RuneSelf {
			if InvalidAscii(pi) {
				err.Index = offset + i
				err.Size = 1
				return
			}
			i++
			continue
		}
		x := first[pi]
		if x == xx {
			// Illegal starter byte.
			err.Index = offset + i
			err.Size = 1
			return
		}
		size := int(x & 7)
		if i+size > n {
			// Short or invalid.
			err.Index = offset + i
			err.Size = n - i
			return
		}
		accept := acceptRanges[x>>4]
		if c := p[i+1]; c < accept.lo || accept.hi < c {
			err.Index = offset + i
			err.Size = 2
			return
		} else if size == 2 {
		} else if c := p[i+2]; c < locb || hicb < c {
			err.Index = offset + i
			err.Size = 3
			return
		} else if size == 3 {
		} else if c := p[i+3]; c < locb || hicb < c {
			err.Index = offset + i
			err.Size = 4
			return
		}
		i += size
	}
	return
}

// Return the size of the next rune if valid, 0 otherwise.
func Utf8ValidNext(p []byte) int {
	c := p[0]

	if c < utf8.RuneSelf {
		if InvalidAscii(c) {
			return 0
		}
		return 1
	}

	x := first[c]
	if x == xx {
		// Illegal starter byte.
		return 0
	}
	size := int(x & 7)
	if size > len(p) {
		// Short or invalid.
		return 0
	}
	accept := acceptRanges[x>>4]
	if c := p[1]; c < accept.lo || accept.hi < c {
		return 0
	} else if size == 2 {
	} else if c := p[2]; c < locb || hicb < c {
		return 0
	} else if size == 3 {
	} else if c := p[3]; c < locb || hicb < c {
		return 0
	}

	return size
}

// acceptRange gives the range of valid values for the second byte in a UTF-8
// sequence.
type acceptRange struct {
	lo uint8 // lowest value for second byte.
	hi uint8 // highest value for second byte.
}

// acceptRanges has size 16 to avoid bounds checks in the code that uses it.
var acceptRanges = [16]acceptRange{
	0: {locb, hicb},
	1: {0xA0, hicb},
	2: {locb, 0x9F},
	3: {0x90, hicb},
	4: {locb, 0x8F},
}

// first is information about the first byte in a UTF-8 sequence.
var first = [256]uint8{
	//   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x00-0x0F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x10-0x1F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x20-0x2F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x30-0x3F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x40-0x4F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x50-0x5F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x60-0x6F
	as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, as, // 0x70-0x7F
	//   1   2   3   4   5   6   7   8   9   A   B   C   D   E   F
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0x80-0x8F
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0x90-0x9F
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0xA0-0xAF
	xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0xB0-0xBF
	xx, xx, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, // 0xC0-0xCF
	s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, s1, // 0xD0-0xDF
	s2, s3, s3, s3, s3, s3, s3, s3, s3, s3, s3, s3, s3, s4, s3, s3, // 0xE0-0xEF
	s5, s6, s6, s6, s7, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, xx, // 0xF0-0xFF
}

const (
	// The default lowest and highest continuation byte.
	locb = 0b10000000
	hicb = 0b10111111

	// These names of these constants are chosen to give nice alignment in the
	// table below. The first nibble is an index into acceptRanges or F for
	// special one-byte cases. The second nibble is the Rune length or the
	// Status for the special one-byte case.
	xx = 0xF1 // invalid: size 1
	as = 0xF0 // ASCII: size 1
	s1 = 0x02 // accept 0, size 2
	s2 = 0x13 // accept 1, size 3
	s3 = 0x03 // accept 0, size 3
	s4 = 0x23 // accept 2, size 3
	s5 = 0x34 // accept 3, size 4
	s6 = 0x04 // accept 0, size 4
	s7 = 0x44 // accept 4, size 4
)
