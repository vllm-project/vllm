// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"fmt"
	"io"
	"math"

	"github.com/klauspost/compress/internal/le"
)

const (
	// The largest offset code.
	offsetCodeCount = 30

	// The special code used to mark the end of a block.
	endBlockMarker = 256

	// The first length code.
	lengthCodesStart = 257

	// The number of codegen codes.
	codegenCodeCount = 19
	badCode          = 255

	// maxPredefinedTokens is the maximum number of tokens
	// where we check if fixed size is smaller.
	maxPredefinedTokens = 250

	// bufferFlushSize indicates the buffer size
	// after which bytes are flushed to the writer.
	// Should preferably be a multiple of 6, since
	// we accumulate 6 bytes between writes to the buffer.
	bufferFlushSize = 246
)

// Minimum length code that emits bits.
const lengthExtraBitsMinCode = 8

// The number of extra bits needed by length code X - LENGTH_CODES_START.
var lengthExtraBits = [32]uint8{
	/* 257 */ 0, 0, 0,
	/* 260 */ 0, 0, 0, 0, 0, 1, 1, 1, 1, 2,
	/* 270 */ 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
	/* 280 */ 4, 5, 5, 5, 5, 0,
}

// The length indicated by length code X - LENGTH_CODES_START.
var lengthBase = [32]uint8{
	0, 1, 2, 3, 4, 5, 6, 7, 8, 10,
	12, 14, 16, 20, 24, 28, 32, 40, 48, 56,
	64, 80, 96, 112, 128, 160, 192, 224, 255,
}

// Minimum offset code that emits bits.
const offsetExtraBitsMinCode = 4

// offset code word extra bits.
var offsetExtraBits = [32]int8{
	0, 0, 0, 0, 1, 1, 2, 2, 3, 3,
	4, 4, 5, 5, 6, 6, 7, 7, 8, 8,
	9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
	/* extended window */
	14, 14,
}

var offsetCombined = [32]uint32{}

func init() {
	var offsetBase = [32]uint32{
		/* normal deflate */
		0x000000, 0x000001, 0x000002, 0x000003, 0x000004,
		0x000006, 0x000008, 0x00000c, 0x000010, 0x000018,
		0x000020, 0x000030, 0x000040, 0x000060, 0x000080,
		0x0000c0, 0x000100, 0x000180, 0x000200, 0x000300,
		0x000400, 0x000600, 0x000800, 0x000c00, 0x001000,
		0x001800, 0x002000, 0x003000, 0x004000, 0x006000,

		/* extended window */
		0x008000, 0x00c000,
	}

	for i := range offsetCombined[:] {
		// Don't use extended window values...
		if offsetExtraBits[i] == 0 || offsetBase[i] > 0x006000 {
			continue
		}
		offsetCombined[i] = uint32(offsetExtraBits[i]) | (offsetBase[i] << 8)
	}
}

// The odd order in which the codegen code sizes are written.
var codegenOrder = []uint32{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15}

type huffmanBitWriter struct {
	// writer is the underlying writer.
	// Do not use it directly; use the write method, which ensures
	// that Write errors are sticky.
	writer io.Writer

	// Data waiting to be written is bytes[0:nbytes]
	// and then the low nbits of bits.
	bits            uint64
	nbits           uint8
	nbytes          uint8
	lastHuffMan     bool
	literalEncoding *huffmanEncoder
	tmpLitEncoding  *huffmanEncoder
	offsetEncoding  *huffmanEncoder
	codegenEncoding *huffmanEncoder
	err             error
	lastHeader      int
	// Set between 0 (reused block can be up to 2x the size)
	logNewTablePenalty uint
	bytes              [256 + 8]byte
	literalFreq        [lengthCodesStart + 32]uint16
	offsetFreq         [32]uint16
	codegenFreq        [codegenCodeCount]uint16

	// codegen must have an extra space for the final symbol.
	codegen [literalCount + offsetCodeCount + 1]uint8
}

// Huffman reuse.
//
// The huffmanBitWriter supports reusing huffman tables and thereby combining block sections.
//
// This is controlled by several variables:
//
// If lastHeader is non-zero the Huffman table can be reused.
// This also indicates that a Huffman table has been generated that can output all
// possible symbols.
// It also indicates that an EOB has not yet been emitted, so if a new tabel is generated
// an EOB with the previous table must be written.
//
// If lastHuffMan is set, a table for outputting literals has been generated and offsets are invalid.
//
// An incoming block estimates the output size of a new table using a 'fresh' by calculating the
// optimal size and adding a penalty in 'logNewTablePenalty'.
// A Huffman table is not optimal, which is why we add a penalty, and generating a new table
// is slower both for compression and decompression.

func newHuffmanBitWriter(w io.Writer) *huffmanBitWriter {
	return &huffmanBitWriter{
		writer:          w,
		literalEncoding: newHuffmanEncoder(literalCount),
		tmpLitEncoding:  newHuffmanEncoder(literalCount),
		codegenEncoding: newHuffmanEncoder(codegenCodeCount),
		offsetEncoding:  newHuffmanEncoder(offsetCodeCount),
	}
}

func (w *huffmanBitWriter) reset(writer io.Writer) {
	w.writer = writer
	w.bits, w.nbits, w.nbytes, w.err = 0, 0, 0, nil
	w.lastHeader = 0
	w.lastHuffMan = false
}

func (w *huffmanBitWriter) canReuse(t *tokens) (ok bool) {
	a := t.offHist[:offsetCodeCount]
	b := w.offsetEncoding.codes
	b = b[:len(a)]
	for i, v := range a {
		if v != 0 && b[i].zero() {
			return false
		}
	}

	a = t.extraHist[:literalCount-256]
	b = w.literalEncoding.codes[256:literalCount]
	b = b[:len(a)]
	for i, v := range a {
		if v != 0 && b[i].zero() {
			return false
		}
	}

	a = t.litHist[:256]
	b = w.literalEncoding.codes[:len(a)]
	for i, v := range a {
		if v != 0 && b[i].zero() {
			return false
		}
	}
	return true
}

func (w *huffmanBitWriter) flush() {
	if w.err != nil {
		w.nbits = 0
		return
	}
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}
	n := w.nbytes
	for w.nbits != 0 {
		w.bytes[n] = byte(w.bits)
		w.bits >>= 8
		if w.nbits > 8 { // Avoid underflow
			w.nbits -= 8
		} else {
			w.nbits = 0
		}
		n++
	}
	w.bits = 0
	if n > 0 {
		w.write(w.bytes[:n])
	}
	w.nbytes = 0
}

func (w *huffmanBitWriter) write(b []byte) {
	if w.err != nil {
		return
	}
	_, w.err = w.writer.Write(b)
}

func (w *huffmanBitWriter) writeBits(b int32, nb uint8) {
	w.bits |= uint64(b) << (w.nbits & 63)
	w.nbits += nb
	if w.nbits >= 48 {
		w.writeOutBits()
	}
}

func (w *huffmanBitWriter) writeBytes(bytes []byte) {
	if w.err != nil {
		return
	}
	n := w.nbytes
	if w.nbits&7 != 0 {
		w.err = InternalError("writeBytes with unfinished bits")
		return
	}
	for w.nbits != 0 {
		w.bytes[n] = byte(w.bits)
		w.bits >>= 8
		w.nbits -= 8
		n++
	}
	if n != 0 {
		w.write(w.bytes[:n])
	}
	w.nbytes = 0
	w.write(bytes)
}

// RFC 1951 3.2.7 specifies a special run-length encoding for specifying
// the literal and offset lengths arrays (which are concatenated into a single
// array).  This method generates that run-length encoding.
//
// The result is written into the codegen array, and the frequencies
// of each code is written into the codegenFreq array.
// Codes 0-15 are single byte codes. Codes 16-18 are followed by additional
// information. Code badCode is an end marker
//
//	numLiterals      The number of literals in literalEncoding
//	numOffsets       The number of offsets in offsetEncoding
//	litenc, offenc   The literal and offset encoder to use
func (w *huffmanBitWriter) generateCodegen(numLiterals int, numOffsets int, litEnc, offEnc *huffmanEncoder) {
	for i := range w.codegenFreq {
		w.codegenFreq[i] = 0
	}
	// Note that we are using codegen both as a temporary variable for holding
	// a copy of the frequencies, and as the place where we put the result.
	// This is fine because the output is always shorter than the input used
	// so far.
	codegen := w.codegen[:] // cache
	// Copy the concatenated code sizes to codegen. Put a marker at the end.
	cgnl := codegen[:numLiterals]
	for i := range cgnl {
		cgnl[i] = litEnc.codes[i].len()
	}

	cgnl = codegen[numLiterals : numLiterals+numOffsets]
	for i := range cgnl {
		cgnl[i] = offEnc.codes[i].len()
	}
	codegen[numLiterals+numOffsets] = badCode

	size := codegen[0]
	count := 1
	outIndex := 0
	for inIndex := 1; size != badCode; inIndex++ {
		// INVARIANT: We have seen "count" copies of size that have not yet
		// had output generated for them.
		nextSize := codegen[inIndex]
		if nextSize == size {
			count++
			continue
		}
		// We need to generate codegen indicating "count" of size.
		if size != 0 {
			codegen[outIndex] = size
			outIndex++
			w.codegenFreq[size]++
			count--
			for count >= 3 {
				n := min(6, count)
				codegen[outIndex] = 16
				outIndex++
				codegen[outIndex] = uint8(n - 3)
				outIndex++
				w.codegenFreq[16]++
				count -= n
			}
		} else {
			for count >= 11 {
				n := min(138, count)
				codegen[outIndex] = 18
				outIndex++
				codegen[outIndex] = uint8(n - 11)
				outIndex++
				w.codegenFreq[18]++
				count -= n
			}
			if count >= 3 {
				// count >= 3 && count <= 10
				codegen[outIndex] = 17
				outIndex++
				codegen[outIndex] = uint8(count - 3)
				outIndex++
				w.codegenFreq[17]++
				count = 0
			}
		}
		count--
		for ; count >= 0; count-- {
			codegen[outIndex] = size
			outIndex++
			w.codegenFreq[size]++
		}
		// Set up invariant for next time through the loop.
		size = nextSize
		count = 1
	}
	// Marker indicating the end of the codegen.
	codegen[outIndex] = badCode
}

func (w *huffmanBitWriter) codegens() int {
	numCodegens := len(w.codegenFreq)
	for numCodegens > 4 && w.codegenFreq[codegenOrder[numCodegens-1]] == 0 {
		numCodegens--
	}
	return numCodegens
}

func (w *huffmanBitWriter) headerSize() (size, numCodegens int) {
	numCodegens = len(w.codegenFreq)
	for numCodegens > 4 && w.codegenFreq[codegenOrder[numCodegens-1]] == 0 {
		numCodegens--
	}
	return 3 + 5 + 5 + 4 + (3 * numCodegens) +
		w.codegenEncoding.bitLength(w.codegenFreq[:]) +
		int(w.codegenFreq[16])*2 +
		int(w.codegenFreq[17])*3 +
		int(w.codegenFreq[18])*7, numCodegens
}

// dynamicSize returns the size of dynamically encoded data in bits.
func (w *huffmanBitWriter) dynamicReuseSize(litEnc, offEnc *huffmanEncoder) (size int) {
	size = litEnc.bitLength(w.literalFreq[:]) +
		offEnc.bitLength(w.offsetFreq[:])
	return size
}

// dynamicSize returns the size of dynamically encoded data in bits.
func (w *huffmanBitWriter) dynamicSize(litEnc, offEnc *huffmanEncoder, extraBits int) (size, numCodegens int) {
	header, numCodegens := w.headerSize()
	size = header +
		litEnc.bitLength(w.literalFreq[:]) +
		offEnc.bitLength(w.offsetFreq[:]) +
		extraBits
	return size, numCodegens
}

// extraBitSize will return the number of bits that will be written
// as "extra" bits on matches.
func (w *huffmanBitWriter) extraBitSize() int {
	total := 0
	for i, n := range w.literalFreq[257:literalCount] {
		total += int(n) * int(lengthExtraBits[i&31])
	}
	for i, n := range w.offsetFreq[:offsetCodeCount] {
		total += int(n) * int(offsetExtraBits[i&31])
	}
	return total
}

// fixedSize returns the size of dynamically encoded data in bits.
func (w *huffmanBitWriter) fixedSize(extraBits int) int {
	return 3 +
		fixedLiteralEncoding.bitLength(w.literalFreq[:]) +
		fixedOffsetEncoding.bitLength(w.offsetFreq[:]) +
		extraBits
}

// storedSize calculates the stored size, including header.
// The function returns the size in bits and whether the block
// fits inside a single block.
func (w *huffmanBitWriter) storedSize(in []byte) (int, bool) {
	if in == nil {
		return 0, false
	}
	if len(in) <= maxStoreBlockSize {
		return (len(in) + 5) * 8, true
	}
	return 0, false
}

func (w *huffmanBitWriter) writeCode(c hcode) {
	// The function does not get inlined if we "& 63" the shift.
	w.bits |= c.code64() << (w.nbits & 63)
	w.nbits += c.len()
	if w.nbits >= 48 {
		w.writeOutBits()
	}
}

// writeOutBits will write bits to the buffer.
func (w *huffmanBitWriter) writeOutBits() {
	bits := w.bits
	w.bits >>= 48
	w.nbits -= 48
	n := w.nbytes

	// We overwrite, but faster...
	le.Store64(w.bytes[:], n, bits)
	n += 6

	if n >= bufferFlushSize {
		if w.err != nil {
			n = 0
			return
		}
		w.write(w.bytes[:n])
		n = 0
	}

	w.nbytes = n
}

// Write the header of a dynamic Huffman block to the output stream.
//
//	numLiterals  The number of literals specified in codegen
//	numOffsets   The number of offsets specified in codegen
//	numCodegens  The number of codegens used in codegen
func (w *huffmanBitWriter) writeDynamicHeader(numLiterals int, numOffsets int, numCodegens int, isEof bool) {
	if w.err != nil {
		return
	}
	var firstBits int32 = 4
	if isEof {
		firstBits = 5
	}
	w.writeBits(firstBits, 3)
	w.writeBits(int32(numLiterals-257), 5)
	w.writeBits(int32(numOffsets-1), 5)
	w.writeBits(int32(numCodegens-4), 4)

	for i := range numCodegens {
		value := uint(w.codegenEncoding.codes[codegenOrder[i]].len())
		w.writeBits(int32(value), 3)
	}

	i := 0
	for {
		var codeWord = uint32(w.codegen[i])
		i++
		if codeWord == badCode {
			break
		}
		w.writeCode(w.codegenEncoding.codes[codeWord])

		switch codeWord {
		case 16:
			w.writeBits(int32(w.codegen[i]), 2)
			i++
		case 17:
			w.writeBits(int32(w.codegen[i]), 3)
			i++
		case 18:
			w.writeBits(int32(w.codegen[i]), 7)
			i++
		}
	}
}

// writeStoredHeader will write a stored header.
// If the stored block is only used for EOF,
// it is replaced with a fixed huffman block.
func (w *huffmanBitWriter) writeStoredHeader(length int, isEof bool) {
	if w.err != nil {
		return
	}
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}

	// To write EOF, use a fixed encoding block. 10 bits instead of 5 bytes.
	if length == 0 && isEof {
		w.writeFixedHeader(isEof)
		// EOB: 7 bits, value: 0
		w.writeBits(0, 7)
		w.flush()
		return
	}

	var flag int32
	if isEof {
		flag = 1
	}
	w.writeBits(flag, 3)
	w.flush()
	w.writeBits(int32(length), 16)
	w.writeBits(int32(^uint16(length)), 16)
}

func (w *huffmanBitWriter) writeFixedHeader(isEof bool) {
	if w.err != nil {
		return
	}
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}

	// Indicate that we are a fixed Huffman block
	var value int32 = 2
	if isEof {
		value = 3
	}
	w.writeBits(value, 3)
}

// writeBlock will write a block of tokens with the smallest encoding.
// The original input can be supplied, and if the huffman encoded data
// is larger than the original bytes, the data will be written as a
// stored block.
// If the input is nil, the tokens will always be Huffman encoded.
func (w *huffmanBitWriter) writeBlock(tokens *tokens, eof bool, input []byte) {
	if w.err != nil {
		return
	}

	tokens.AddEOB()
	if w.lastHeader > 0 {
		// We owe an EOB
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}
	numLiterals, numOffsets := w.indexTokens(tokens, false)
	w.generate()
	var extraBits int
	storedSize, storable := w.storedSize(input)
	if storable {
		extraBits = w.extraBitSize()
	}

	// Figure out smallest code.
	// Fixed Huffman baseline.
	var literalEncoding = fixedLiteralEncoding
	var offsetEncoding = fixedOffsetEncoding
	var size = math.MaxInt32
	if tokens.n < maxPredefinedTokens {
		size = w.fixedSize(extraBits)
	}

	// Dynamic Huffman?
	var numCodegens int

	// Generate codegen and codegenFrequencies, which indicates how to encode
	// the literalEncoding and the offsetEncoding.
	w.generateCodegen(numLiterals, numOffsets, w.literalEncoding, w.offsetEncoding)
	w.codegenEncoding.generate(w.codegenFreq[:], 7)
	dynamicSize, numCodegens := w.dynamicSize(w.literalEncoding, w.offsetEncoding, extraBits)

	if dynamicSize < size {
		size = dynamicSize
		literalEncoding = w.literalEncoding
		offsetEncoding = w.offsetEncoding
	}

	// Stored bytes?
	if storable && storedSize <= size {
		w.writeStoredHeader(len(input), eof)
		w.writeBytes(input)
		return
	}

	// Huffman.
	if literalEncoding == fixedLiteralEncoding {
		w.writeFixedHeader(eof)
	} else {
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
	}

	// Write the tokens.
	w.writeTokens(tokens.Slice(), literalEncoding.codes, offsetEncoding.codes)
}

// writeBlockDynamic encodes a block using a dynamic Huffman table.
// This should be used if the symbols used have a disproportionate
// histogram distribution.
// If input is supplied and the compression savings are below 1/16th of the
// input size the block is stored.
func (w *huffmanBitWriter) writeBlockDynamic(tokens *tokens, eof bool, input []byte, sync bool) {
	if w.err != nil {
		return
	}

	sync = sync || eof
	if sync {
		tokens.AddEOB()
	}

	// We cannot reuse pure huffman table, and must mark as EOF.
	if (w.lastHuffMan || eof) && w.lastHeader > 0 {
		// We will not try to reuse.
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
		w.lastHuffMan = false
	}

	// fillReuse enables filling of empty values.
	// This will make encodings always reusable without testing.
	// However, this does not appear to benefit on most cases.
	const fillReuse = false

	// Check if we can reuse...
	if !fillReuse && w.lastHeader > 0 && !w.canReuse(tokens) {
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
	}

	numLiterals, numOffsets := w.indexTokens(tokens, true)
	extraBits := 0
	ssize, storable := w.storedSize(input)

	const usePrefs = true
	if storable || w.lastHeader > 0 {
		extraBits = w.extraBitSize()
	}

	var size int

	// Check if we should reuse.
	if w.lastHeader > 0 {
		// Estimate size for using a new table.
		// Use the previous header size as the best estimate.
		newSize := w.lastHeader + tokens.EstimatedBits()
		newSize += int(w.literalEncoding.codes[endBlockMarker].len()) + newSize>>w.logNewTablePenalty

		// The estimated size is calculated as an optimal table.
		// We add a penalty to make it more realistic and re-use a bit more.
		reuseSize := w.dynamicReuseSize(w.literalEncoding, w.offsetEncoding) + extraBits

		// Check if a new table is better.
		if newSize < reuseSize {
			// Write the EOB we owe.
			w.writeCode(w.literalEncoding.codes[endBlockMarker])
			size = newSize
			w.lastHeader = 0
		} else {
			size = reuseSize
		}

		if tokens.n < maxPredefinedTokens {
			if preSize := w.fixedSize(extraBits) + 7; usePrefs && preSize < size {
				// Check if we get a reasonable size decrease.
				if storable && ssize <= size {
					w.writeStoredHeader(len(input), eof)
					w.writeBytes(input)
					return
				}
				w.writeFixedHeader(eof)
				if !sync {
					tokens.AddEOB()
				}
				w.writeTokens(tokens.Slice(), fixedLiteralEncoding.codes, fixedOffsetEncoding.codes)
				return
			}
		}
		// Check if we get a reasonable size decrease.
		if storable && ssize <= size {
			w.writeStoredHeader(len(input), eof)
			w.writeBytes(input)
			return
		}
	}

	// We want a new block/table
	if w.lastHeader == 0 {
		if fillReuse && !sync {
			w.fillTokens()
			numLiterals, numOffsets = maxNumLit, maxNumDist
		} else {
			w.literalFreq[endBlockMarker] = 1
		}

		w.generate()
		// Generate codegen and codegenFrequencies, which indicates how to encode
		// the literalEncoding and the offsetEncoding.
		w.generateCodegen(numLiterals, numOffsets, w.literalEncoding, w.offsetEncoding)
		w.codegenEncoding.generate(w.codegenFreq[:], 7)

		var numCodegens int
		if fillReuse && !sync {
			// Reindex for accurate size...
			w.indexTokens(tokens, true)
		}
		size, numCodegens = w.dynamicSize(w.literalEncoding, w.offsetEncoding, extraBits)

		// Store predefined, if we don't get a reasonable improvement.
		if tokens.n < maxPredefinedTokens {
			if preSize := w.fixedSize(extraBits); usePrefs && preSize <= size {
				// Store bytes, if we don't get an improvement.
				if storable && ssize <= preSize {
					w.writeStoredHeader(len(input), eof)
					w.writeBytes(input)
					return
				}
				w.writeFixedHeader(eof)
				if !sync {
					tokens.AddEOB()
				}
				w.writeTokens(tokens.Slice(), fixedLiteralEncoding.codes, fixedOffsetEncoding.codes)
				return
			}
		}

		if storable && ssize <= size {
			// Store bytes, if we don't get an improvement.
			w.writeStoredHeader(len(input), eof)
			w.writeBytes(input)
			return
		}

		// Write Huffman table.
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
		if !sync {
			w.lastHeader, _ = w.headerSize()
		}
		w.lastHuffMan = false
	}

	if sync {
		w.lastHeader = 0
	}
	// Write the tokens.
	w.writeTokens(tokens.Slice(), w.literalEncoding.codes, w.offsetEncoding.codes)
}

func (w *huffmanBitWriter) fillTokens() {
	for i, v := range w.literalFreq[:literalCount] {
		if v == 0 {
			w.literalFreq[i] = 1
		}
	}
	for i, v := range w.offsetFreq[:offsetCodeCount] {
		if v == 0 {
			w.offsetFreq[i] = 1
		}
	}
}

// indexTokens indexes a slice of tokens, and updates
// literalFreq and offsetFreq, and generates literalEncoding
// and offsetEncoding.
// The number of literal and offset tokens is returned.
func (w *huffmanBitWriter) indexTokens(t *tokens, alwaysEOB bool) (numLiterals, numOffsets int) {
	//copy(w.literalFreq[:], t.litHist[:])
	*(*[256]uint16)(w.literalFreq[:]) = t.litHist
	//copy(w.literalFreq[256:], t.extraHist[:])
	*(*[32]uint16)(w.literalFreq[256:]) = t.extraHist
	w.offsetFreq = t.offHist

	if t.n == 0 {
		return
	}
	if alwaysEOB {
		w.literalFreq[endBlockMarker] = 1
	}

	// get the number of literals
	numLiterals = len(w.literalFreq)
	for w.literalFreq[numLiterals-1] == 0 {
		numLiterals--
	}
	// get the number of offsets
	numOffsets = len(w.offsetFreq)
	for numOffsets > 0 && w.offsetFreq[numOffsets-1] == 0 {
		numOffsets--
	}
	if numOffsets == 0 {
		// We haven't found a single match. If we want to go with the dynamic encoding,
		// we should count at least one offset to be sure that the offset huffman tree could be encoded.
		w.offsetFreq[0] = 1
		numOffsets = 1
	}
	return
}

func (w *huffmanBitWriter) generate() {
	w.literalEncoding.generate(w.literalFreq[:literalCount], 15)
	w.offsetEncoding.generate(w.offsetFreq[:offsetCodeCount], 15)
}

// writeTokens writes a slice of tokens to the output.
// codes for literal and offset encoding must be supplied.
func (w *huffmanBitWriter) writeTokens(tokens []token, leCodes, oeCodes []hcode) {
	if w.err != nil {
		return
	}
	if len(tokens) == 0 {
		return
	}

	// Only last token should be endBlockMarker.
	var deferEOB bool
	if tokens[len(tokens)-1] == endBlockMarker {
		tokens = tokens[:len(tokens)-1]
		deferEOB = true
	}

	// Create slices up to the next power of two to avoid bounds checks.
	lits := leCodes[:256]
	offs := oeCodes[:32]
	lengths := leCodes[lengthCodesStart:]
	lengths = lengths[:32]

	// Go 1.16 LOVES having these on stack.
	bits, nbits, nbytes := w.bits, w.nbits, w.nbytes

	for _, t := range tokens {
		if t < 256 {
			//w.writeCode(lits[t.literal()])
			c := lits[t]
			bits |= c.code64() << (nbits & 63)
			nbits += c.len()
			if nbits >= 48 {
				le.Store64(w.bytes[:], nbytes, bits)
				bits >>= 48
				nbits -= 48
				nbytes += 6
				if nbytes >= bufferFlushSize {
					if w.err != nil {
						nbytes = 0
						return
					}
					_, w.err = w.writer.Write(w.bytes[:nbytes])
					nbytes = 0
				}
			}
			continue
		}

		// Write the length
		length := t.length()
		lengthCode := lengthCode(length) & 31
		if false {
			w.writeCode(lengths[lengthCode])
		} else {
			// inlined
			c := lengths[lengthCode]
			bits |= c.code64() << (nbits & 63)
			nbits += c.len()
			if nbits >= 48 {
				le.Store64(w.bytes[:], nbytes, bits)
				bits >>= 48
				nbits -= 48
				nbytes += 6
				if nbytes >= bufferFlushSize {
					if w.err != nil {
						nbytes = 0
						return
					}
					_, w.err = w.writer.Write(w.bytes[:nbytes])
					nbytes = 0
				}
			}
		}

		if lengthCode >= lengthExtraBitsMinCode {
			extraLengthBits := lengthExtraBits[lengthCode]
			//w.writeBits(extraLength, extraLengthBits)
			extraLength := int32(length - lengthBase[lengthCode])
			bits |= uint64(extraLength) << (nbits & 63)
			nbits += extraLengthBits
			if nbits >= 48 {
				le.Store64(w.bytes[:], nbytes, bits)
				bits >>= 48
				nbits -= 48
				nbytes += 6
				if nbytes >= bufferFlushSize {
					if w.err != nil {
						nbytes = 0
						return
					}
					_, w.err = w.writer.Write(w.bytes[:nbytes])
					nbytes = 0
				}
			}
		}
		// Write the offset
		offset := t.offset()
		offsetCode := (offset >> 16) & 31
		if false {
			w.writeCode(offs[offsetCode])
		} else {
			// inlined
			c := offs[offsetCode]
			bits |= c.code64() << (nbits & 63)
			nbits += c.len()
			if nbits >= 48 {
				le.Store64(w.bytes[:], nbytes, bits)
				bits >>= 48
				nbits -= 48
				nbytes += 6
				if nbytes >= bufferFlushSize {
					if w.err != nil {
						nbytes = 0
						return
					}
					_, w.err = w.writer.Write(w.bytes[:nbytes])
					nbytes = 0
				}
			}
		}

		if offsetCode >= offsetExtraBitsMinCode {
			offsetComb := offsetCombined[offsetCode]
			//w.writeBits(extraOffset, extraOffsetBits)
			bits |= uint64((offset-(offsetComb>>8))&matchOffsetOnlyMask) << (nbits & 63)
			nbits += uint8(offsetComb)
			if nbits >= 48 {
				le.Store64(w.bytes[:], nbytes, bits)
				bits >>= 48
				nbits -= 48
				nbytes += 6
				if nbytes >= bufferFlushSize {
					if w.err != nil {
						nbytes = 0
						return
					}
					_, w.err = w.writer.Write(w.bytes[:nbytes])
					nbytes = 0
				}
			}
		}
	}
	// Restore...
	w.bits, w.nbits, w.nbytes = bits, nbits, nbytes

	if deferEOB {
		w.writeCode(leCodes[endBlockMarker])
	}
}

// huffOffset is a static offset encoder used for huffman only encoding.
// It can be reused since we will not be encoding offset values.
var huffOffset *huffmanEncoder

func init() {
	w := newHuffmanBitWriter(nil)
	w.offsetFreq[0] = 1
	huffOffset = newHuffmanEncoder(offsetCodeCount)
	huffOffset.generate(w.offsetFreq[:offsetCodeCount], 15)
}

// writeBlockHuff encodes a block of bytes as either
// Huffman encoded literals or uncompressed bytes if the
// results only gains very little from compression.
func (w *huffmanBitWriter) writeBlockHuff(eof bool, input []byte, sync bool) {
	if w.err != nil {
		return
	}

	// Clear histogram
	for i := range w.literalFreq[:] {
		w.literalFreq[i] = 0
	}
	if !w.lastHuffMan {
		for i := range w.offsetFreq[:] {
			w.offsetFreq[i] = 0
		}
	}

	const numLiterals = endBlockMarker + 1
	const numOffsets = 1

	// Add everything as literals
	// We have to estimate the header size.
	// Assume header is around 70 bytes:
	// https://stackoverflow.com/a/25454430
	const guessHeaderSizeBits = 70 * 8
	histogram(input, w.literalFreq[:numLiterals])
	ssize, storable := w.storedSize(input)
	if storable && len(input) > 1024 {
		// Quick check for incompressible content.
		abs := float64(0)
		avg := float64(len(input)) / 256
		max := float64(len(input) * 2)
		for _, v := range w.literalFreq[:256] {
			diff := float64(v) - avg
			abs += diff * diff
			if abs > max {
				break
			}
		}
		if abs < max {
			if debugDeflate {
				fmt.Println("stored", abs, "<", max)
			}
			// No chance we can compress this...
			w.writeStoredHeader(len(input), eof)
			w.writeBytes(input)
			return
		}
	}
	w.literalFreq[endBlockMarker] = 1
	w.tmpLitEncoding.generate(w.literalFreq[:numLiterals], 15)
	estBits := w.tmpLitEncoding.canReuseBits(w.literalFreq[:numLiterals])
	if estBits < math.MaxInt32 {
		estBits += w.lastHeader
		if w.lastHeader == 0 {
			estBits += guessHeaderSizeBits
		}
		estBits += estBits >> w.logNewTablePenalty
	}

	// Store bytes, if we don't get a reasonable improvement.
	if storable && ssize <= estBits {
		if debugDeflate {
			fmt.Println("stored,", ssize, "<=", estBits)
		}
		w.writeStoredHeader(len(input), eof)
		w.writeBytes(input)
		return
	}

	if w.lastHeader > 0 {
		reuseSize := w.literalEncoding.canReuseBits(w.literalFreq[:256])

		if estBits < reuseSize {
			if debugDeflate {
				fmt.Println("NOT reusing, reuse:", reuseSize/8, "> new:", estBits/8, "header est:", w.lastHeader/8, "bytes")
			}
			// We owe an EOB
			w.writeCode(w.literalEncoding.codes[endBlockMarker])
			w.lastHeader = 0
		} else if debugDeflate {
			fmt.Println("reusing, reuse:", reuseSize/8, "> new:", estBits/8, "- header est:", w.lastHeader/8)
		}
	}

	count := 0
	if w.lastHeader == 0 {
		// Use the temp encoding, so swap.
		w.literalEncoding, w.tmpLitEncoding = w.tmpLitEncoding, w.literalEncoding
		// Generate codegen and codegenFrequencies, which indicates how to encode
		// the literalEncoding and the offsetEncoding.
		w.generateCodegen(numLiterals, numOffsets, w.literalEncoding, huffOffset)
		w.codegenEncoding.generate(w.codegenFreq[:], 7)
		numCodegens := w.codegens()

		// Huffman.
		w.writeDynamicHeader(numLiterals, numOffsets, numCodegens, eof)
		w.lastHuffMan = true
		w.lastHeader, _ = w.headerSize()
		if debugDeflate {
			count += w.lastHeader
			fmt.Println("header:", count/8)
		}
	}

	encoding := w.literalEncoding.codes[:256]
	// Go 1.16 LOVES having these on stack. At least 1.5x the speed.
	bits, nbits, nbytes := w.bits, w.nbits, w.nbytes

	if debugDeflate {
		count -= int(nbytes)*8 + int(nbits)
	}
	// Unroll, write 3 codes/loop.
	// Fastest number of unrolls.
	for len(input) > 3 {
		// We must have at least 48 bits free.
		if nbits >= 8 {
			n := nbits >> 3
			le.Store64(w.bytes[:], nbytes, bits)
			bits >>= (n * 8) & 63
			nbits -= n * 8
			nbytes += n
		}
		if nbytes >= bufferFlushSize {
			if w.err != nil {
				nbytes = 0
				return
			}
			if debugDeflate {
				count += int(nbytes) * 8
			}
			_, w.err = w.writer.Write(w.bytes[:nbytes])
			nbytes = 0
		}
		a, b := encoding[input[0]], encoding[input[1]]
		bits |= a.code64() << (nbits & 63)
		bits |= b.code64() << ((nbits + a.len()) & 63)
		c := encoding[input[2]]
		nbits += b.len() + a.len()
		bits |= c.code64() << (nbits & 63)
		nbits += c.len()
		input = input[3:]
	}

	// Remaining...
	for _, t := range input {
		if nbits >= 48 {
			le.Store64(w.bytes[:], nbytes, bits)
			bits >>= 48
			nbits -= 48
			nbytes += 6
			if nbytes >= bufferFlushSize {
				if w.err != nil {
					nbytes = 0
					return
				}
				if debugDeflate {
					count += int(nbytes) * 8
				}
				_, w.err = w.writer.Write(w.bytes[:nbytes])
				nbytes = 0
			}
		}
		// Bitwriting inlined, ~30% speedup
		c := encoding[t]
		bits |= c.code64() << (nbits & 63)

		nbits += c.len()
		if debugDeflate {
			count += int(c.len())
		}
	}
	// Restore...
	w.bits, w.nbits, w.nbytes = bits, nbits, nbytes

	if debugDeflate {
		nb := count + int(nbytes)*8 + int(nbits)
		fmt.Println("wrote", nb, "bits,", nb/8, "bytes.")
	}
	// Flush if needed to have space.
	if w.nbits >= 48 {
		w.writeOutBits()
	}

	if eof || sync {
		w.writeCode(w.literalEncoding.codes[endBlockMarker])
		w.lastHeader = 0
		w.lastHuffMan = false
	}
}
