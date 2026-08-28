package flate

import (
	"fmt"

	"github.com/klauspost/compress/internal/le"
)

// fastGen maintains the table for matches,
// and the previous byte block for level 2.
// This is the generic implementation.
type fastEncL1 struct {
	fastGen
	table [tableSize]tableEntry
}

// EncodeL1 uses a similar algorithm to level 1
func (e *fastEncL1) Encode(dst *tokens, src []byte) {
	const (
		inputMargin            = 12 - 1
		minNonLiteralBlockSize = 1 + 1 + inputMargin
		hashBytes              = 5
	)
	if debugDeflate && e.cur < 0 {
		panic(fmt.Sprint("e.cur < 0: ", e.cur))
	}

	// Protect against e.cur wraparound.
	for e.cur >= bufferReset {
		if len(e.hist) == 0 {
			for i := range e.table[:] {
				e.table[i] = tableEntry{}
			}
			e.cur = maxMatchOffset
			break
		}
		// Shift down everything in the table that isn't already too far away.
		minOff := e.cur + int32(len(e.hist)) - maxMatchOffset
		for i := range e.table[:] {
			v := e.table[i].offset
			if v <= minOff {
				v = 0
			} else {
				v = v - e.cur + maxMatchOffset
			}
			e.table[i].offset = v
		}
		e.cur = maxMatchOffset
	}

	s := e.addBlock(src)

	// This check isn't in the Snappy implementation, but there, the caller
	// instead of the callee handles this case.
	if len(src) < minNonLiteralBlockSize {
		// We do not fill the token table.
		// This will be picked up by caller.
		dst.n = uint16(len(src))
		return
	}

	// Override src
	src = e.hist
	nextEmit := s

	// sLimit is when to stop looking for offset/length copies. The inputMargin
	// lets us use a fast path for emitLiteral in the main loop, while we are
	// looking for copies.
	sLimit := int32(len(src) - inputMargin)

	// nextEmit is where in src the next emitLiteral should start from.
	cv := load6432(src, s)

	for {
		const skipLog = 5
		const doEvery = 2

		nextS := s
		var candidate tableEntry
		var t int32
		for {
			nextHash := hashLen(cv, tableBits, hashBytes)
			candidate = e.table[nextHash]
			nextS = s + doEvery + (s-nextEmit)>>skipLog
			if nextS > sLimit {
				goto emitRemainder
			}

			now := load6432(src, nextS)
			e.table[nextHash] = tableEntry{offset: s + e.cur}
			nextHash = hashLen(now, tableBits, hashBytes)
			t = candidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == load3232(src, t) {
				e.table[nextHash] = tableEntry{offset: nextS + e.cur}
				break
			}

			// Do one right away...
			cv = now
			s = nextS
			nextS++
			candidate = e.table[nextHash]
			now >>= 8
			e.table[nextHash] = tableEntry{offset: s + e.cur}

			t = candidate.offset - e.cur
			if s-t < maxMatchOffset && uint32(cv) == load3232(src, t) {
				e.table[nextHash] = tableEntry{offset: nextS + e.cur}
				break
			}
			cv = now
			s = nextS
		}

		// A 4-byte match has been found. We'll later see if more than 4 bytes
		// match. But, prior to the match, src[nextEmit:s] are unmatched. Emit
		// them as literal bytes.
		for {
			// Invariant: we have a 4-byte match at s, and no need to emit any
			// literal bytes prior to s.

			// Extend the 4-byte match as long as possible.
			l := e.matchlenLong(int(s+4), int(t+4), src) + 4

			// Extend backwards
			for t > 0 && s > nextEmit && le.Load8(src, t-1) == le.Load8(src, s-1) {
				s--
				t--
				l++
			}
			if nextEmit < s {
				if false {
					emitLiteral(dst, src[nextEmit:s])
				} else {
					for _, v := range src[nextEmit:s] {
						dst.tokens[dst.n] = token(v)
						dst.litHist[v]++
						dst.n++
					}
				}
			}

			// Save the match found
			if false {
				dst.AddMatchLong(l, uint32(s-t-baseMatchOffset))
			} else {
				// Inlined...
				xoffset := uint32(s - t - baseMatchOffset)
				xlength := l
				oc := offsetCode(xoffset)
				xoffset |= oc << 16
				for xlength > 0 {
					xl := xlength
					if xl > 258 {
						if xl > 258+baseMatchLength {
							xl = 258
						} else {
							xl = 258 - baseMatchLength
						}
					}
					xlength -= xl
					xl -= baseMatchLength
					dst.extraHist[lengthCodes1[uint8(xl)]]++
					dst.offHist[oc]++
					dst.tokens[dst.n] = token(matchType | uint32(xl)<<lengthShift | xoffset)
					dst.n++
				}
			}
			s += l
			nextEmit = s
			if nextS >= s {
				s = nextS + 1
			}
			if s >= sLimit {
				// Index first pair after match end.
				if int(s+l+8) < len(src) {
					cv := load6432(src, s)
					e.table[hashLen(cv, tableBits, hashBytes)] = tableEntry{offset: s + e.cur}
				}
				goto emitRemainder
			}

			// We could immediately start working at s now, but to improve
			// compression we first update the hash table at s-2 and at s. If
			// another emitCopy is not our next move, also calculate nextHash
			// at s+1. At least on GOARCH=amd64, these three hash calculations
			// are faster as one load64 call (with some shifts) instead of
			// three load32 calls.
			x := load6432(src, s-2)
			o := e.cur + s - 2
			prevHash := hashLen(x, tableBits, hashBytes)
			e.table[prevHash] = tableEntry{offset: o}
			x >>= 16
			currHash := hashLen(x, tableBits, hashBytes)
			candidate = e.table[currHash]
			e.table[currHash] = tableEntry{offset: o + 2}

			t = candidate.offset - e.cur
			if s-t > maxMatchOffset || uint32(x) != load3232(src, t) {
				cv = x >> 8
				s++
				break
			}
		}
	}

emitRemainder:
	if int(nextEmit) < len(src) {
		// If nothing was added, don't encode literals.
		if dst.n == 0 {
			return
		}
		emitLiteral(dst, src[nextEmit:])
	}
}
