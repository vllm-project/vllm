package proto

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"

	"github.com/redis/go-redis/v9/internal/util"
)

// DefaultBufferSize is the default size for read/write buffers (32 KiB).
const DefaultBufferSize = 32 * 1024

// redis resp protocol data type.
const (
	RespStatus    = '+' // +<string>\r\n
	RespError     = '-' // -<string>\r\n
	RespString    = '$' // $<length>\r\n<bytes>\r\n
	RespInt       = ':' // :<number>\r\n
	RespNil       = '_' // _\r\n
	RespFloat     = ',' // ,<floating-point-number>\r\n (golang float)
	RespBool      = '#' // true: #t\r\n false: #f\r\n
	RespBlobError = '!' // !<length>\r\n<bytes>\r\n
	RespVerbatim  = '=' // =<length>\r\nFORMAT:<bytes>\r\n
	RespBigInt    = '(' // (<big number>\r\n
	RespArray     = '*' // *<len>\r\n... (same as resp2)
	RespMap       = '%' // %<len>\r\n(key)\r\n(value)\r\n... (golang map)
	RespSet       = '~' // ~<len>\r\n... (same as Array)
	RespAttr      = '|' // |<len>\r\n(key)\r\n(value)\r\n... + command reply
	RespPush      = '>' // ><len>\r\n... (same as Array)
)

// Not used temporarily.
// Redis has not used these two data types for the time being, and will implement them later.
// Streamed           = "EOF:"
// StreamedAggregated = '?'

//------------------------------------------------------------------------------

const Nil = RedisError("redis: nil") // nolint:errname

type RedisError string

func (e RedisError) Error() string { return string(e) }

func (RedisError) RedisError() {}

func ParseErrorReply(line []byte) error {
	msg := string(line[1:])
	return parseTypedRedisError(msg)
}

//------------------------------------------------------------------------------

type Reader struct {
	rd *bufio.Reader
}

func NewReader(rd io.Reader) *Reader {
	return &Reader{
		rd: bufio.NewReaderSize(rd, DefaultBufferSize),
	}
}

func NewReaderSize(rd io.Reader, size int) *Reader {
	return &Reader{
		rd: bufio.NewReaderSize(rd, size),
	}
}

func (r *Reader) Buffered() int {
	return r.rd.Buffered()
}

// Size returns the size of the underlying read buffer.
func (r *Reader) Size() int {
	return r.rd.Size()
}

func (r *Reader) Peek(n int) ([]byte, error) {
	return r.rd.Peek(n)
}

func (r *Reader) Reset(rd io.Reader) {
	r.rd.Reset(rd)
}

// PeekReplyType returns the data type of the next response without advancing the Reader,
// and discard the attribute type.
func (r *Reader) PeekReplyType() (byte, error) {
	b, err := r.rd.Peek(1)
	if err != nil {
		return 0, err
	}
	if b[0] == RespAttr {
		if err = r.DiscardNext(); err != nil {
			return 0, err
		}
		return r.PeekReplyType()
	}
	return b[0], nil
}

// MinRESP3ReadBufferSize is the minimum buffer size used when RESP3 push
// notifications must be inspected without consuming them.
const MinRESP3ReadBufferSize = 128

// ErrPushNotificationNameTooLong is returned when the push header does not fit
// in the bounded peek window. Callers should consume the frame with ReadReply.
var ErrPushNotificationNameTooLong = errors.New("redis: push notification name exceeds peek window")

// PeekPushNotificationName returns the notification name of the next RESP3
// push frame without consuming it. The caller is expected to have already
// verified that the next reply is a push notification (e.g. via PeekReplyType
// returning RespPush).
//
// To identify the name the method may block reading more bytes from the
// underlying connection, but only ever waits for one byte beyond the valid
// frame prefix it has already seen. That byte is guaranteed to arrive: an
// incomplete prefix means the server is still committed to sending the rest
// of the frame. Demanding any fixed amount instead can deadlock — a complete
// frame such as a subscribe confirmation for a short channel name can be
// smaller than the fixed window, and once it is buffered the server has
// nothing more to send (issue #3935). Blocking for in-flight bytes is
// preferred to a truncated peek, which would silently misidentify the
// notification and cause the caller's ReadReply to consume (and drop) the
// frame; see issue #3839.
func (r *Reader) PeekPushNotificationName() (string, error) {
	c, err := r.rd.Peek(1)
	if err != nil {
		return "", err
	}
	if c[0] != RespPush {
		return "", fmt.Errorf("redis: can't peek push notification name, next reply is not a push notification")
	}

	const maxPushHeaderPeek = 4096

	for {
		// Parse from what is already buffered; this never blocks.
		avail := r.rd.Buffered()
		if avail > maxPushHeaderPeek {
			avail = maxPushHeaderPeek
		}
		buf, peekErr := r.rd.Peek(avail)
		if peekErr != nil {
			return "", peekErr
		}
		name, complete, parseErr := parsePushNotificationName(buf)
		if parseErr != nil {
			return "", parseErr
		}
		if complete {
			return name, nil
		}
		if avail >= maxPushHeaderPeek {
			return "", ErrPushNotificationNameTooLong
		}
		// Valid but incomplete prefix: the rest of the frame is in flight.
		// Block for exactly one more byte — the read that delivers it picks
		// up whatever else has already arrived — then re-parse.
		if _, err := r.rd.Peek(avail + 1); err != nil {
			if errors.Is(err, bufio.ErrBufferFull) {
				return "", ErrPushNotificationNameTooLong
			}
			return "", err
		}
	}
}

// parsePushNotificationName extracts the notification name from a buffered
// RESP3 push frame prefix. The three return values are:
//
//   - (name, true, nil): the full name is in buf.
//   - ("", false, nil):  buf is a valid prefix but too short to determine the
//     name; the caller should fetch more bytes and retry.
//   - ("", _, err):      buf is malformed.
//
// This split lets PeekPushNotificationName tell "incomplete header" apart
// from "corrupt frame" without ever returning a truncated string.
func parsePushNotificationName(buf []byte) (string, bool, error) {
	// Need at least ">N\r" before any meaningful work.
	if len(buf) < 3 {
		return "", false, nil
	}
	if buf[0] != RespPush {
		return "", false, fmt.Errorf("redis: can't parse push notification: %q", buf)
	}

	// Skip the array length line ">N\r\n".
	const arrayLenStart = 1 // first byte after the '>' marker
	pos, ok, err := skipDigitsThenCRLF(buf, arrayLenStart)
	if err != nil {
		return "", false, fmt.Errorf("redis: can't parse push notification: %w", err)
	}
	if !ok {
		return "", false, nil
	}
	// Reject ">\r\n": RESP requires at least one digit for the array length.
	// Without this check the empty length looks like a valid prefix and the
	// caller would block fetching more bytes for a frame that is already
	// malformed.
	if pos-2 == arrayLenStart {
		return "", false, fmt.Errorf("redis: empty push notification array length")
	}

	// First element type byte: '$' (bulk) or '+' (simple-string).
	if pos >= len(buf) {
		return "", false, nil
	}
	typeOfName := buf[pos]
	if typeOfName != RespString && typeOfName != RespStatus {
		return "", false, fmt.Errorf("redis: can't parse push notification name: %q", buf[pos:])
	}
	pos++

	if typeOfName == RespString {
		// Read "$M\r\n" then the M-byte name.
		lenStart := pos
		next, ok, err := skipDigitsThenCRLF(buf, pos)
		if err != nil {
			return "", false, fmt.Errorf("redis: can't parse push notification name length: %w", err)
		}
		if !ok {
			return "", false, nil
		}
		if next-2 == lenStart {
			return "", false, fmt.Errorf("redis: empty push notification name length")
		}
		nameLen, err := util.Atoi(buf[lenStart : next-2])
		if err != nil {
			return "", false, fmt.Errorf("redis: invalid push notification name length %q: %w", buf[lenStart:next-2], err)
		}
		if nameLen < 0 {
			return "", false, fmt.Errorf("redis: negative push notification name length: %d", nameLen)
		}
		// Compare against the remaining bytes instead of computing
		// next+nameLen: a hugely advertised length on malformed input could
		// overflow int, wrap negative, slip past an "end > len(buf)" guard and
		// panic the slice below. next <= len(buf) here, so the subtraction is
		// safe.
		if nameLen > len(buf)-next {
			return "", false, nil
		}
		return util.BytesToString(buf[next : next+nameLen]), true, nil
	}

	// RespStatus: scan for the terminating CRLF.
	for i := pos; i < len(buf)-1; i++ {
		if buf[i] == '\r' && buf[i+1] == '\n' {
			return util.BytesToString(buf[pos:i]), true, nil
		}
	}
	return "", false, nil
}

// skipDigitsThenCRLF advances past zero-or-more ASCII digits and the
// terminating "\r\n" starting at offset start in buf. It returns the position
// after the "\r\n" and true on success; (pos, false, nil) if buf is too
// short; or an error if a non-digit non-CR byte is encountered before the CRLF.
func skipDigitsThenCRLF(buf []byte, start int) (int, bool, error) {
	for pos := start; pos < len(buf)-1; pos++ {
		if buf[pos] == '\r' && buf[pos+1] == '\n' {
			return pos + 2, true, nil
		}
		if buf[pos] < '0' || buf[pos] > '9' {
			return pos, false, fmt.Errorf("expected digit or CRLF, got %q", buf[pos])
		}
	}
	return len(buf), false, nil
}

// ReadLine Return a valid reply, it will check the protocol or redis error,
// and discard the attribute type.
func (r *Reader) ReadLine() ([]byte, error) {
	line, err := r.readLine()
	if err != nil {
		return nil, err
	}
	switch line[0] {
	case RespError:
		return nil, ParseErrorReply(line)
	case RespNil:
		return nil, Nil
	case RespBlobError:
		var blobErr string
		blobErr, err = r.readStringReply(line)
		if err == nil {
			err = parseTypedRedisError(blobErr)
		}
		return nil, err
	case RespAttr:
		if err = r.Discard(line); err != nil {
			return nil, err
		}
		return r.ReadLine()
	}

	// Compatible with RESP2
	if IsNilReply(line) {
		return nil, Nil
	}

	return line, nil
}

// readLine returns an error if:
//   - there is a pending read error;
//   - or line does not end with \r\n.
func (r *Reader) readLine() ([]byte, error) {
	b, err := r.rd.ReadSlice('\n')
	if err != nil {
		if err != bufio.ErrBufferFull {
			return nil, err
		}

		full := make([]byte, len(b))
		copy(full, b)

		b, err = r.rd.ReadBytes('\n')
		if err != nil {
			return nil, err
		}

		full = append(full, b...) //nolint:makezero
		b = full
	}
	if len(b) <= 2 || b[len(b)-1] != '\n' || b[len(b)-2] != '\r' {
		return nil, fmt.Errorf("redis: invalid reply: %q", b)
	}
	return b[:len(b)-2], nil
}

func (r *Reader) ReadReply() (interface{}, error) {
	line, err := r.ReadLine()
	if err != nil {
		return nil, err
	}

	switch line[0] {
	case RespStatus:
		return string(line[1:]), nil
	case RespInt:
		return util.ParseInt(line[1:], 10, 64)
	case RespFloat:
		return r.readFloat(line)
	case RespBool:
		return r.readBool(line)
	case RespBigInt:
		return r.readBigInt(line)

	case RespString:
		return r.readStringReply(line)
	case RespVerbatim:
		return r.readVerb(line)

	case RespArray, RespSet, RespPush:
		return r.readSlice(line)
	case RespMap:
		return r.readMap(line)
	}
	return nil, fmt.Errorf("redis: can't parse %.100q", line)
}

func (r *Reader) readFloat(line []byte) (float64, error) {
	v := util.BytesToString(line[1:])
	switch v {
	case "inf":
		return math.Inf(1), nil
	case "-inf":
		return math.Inf(-1), nil
	case "nan", "-nan":
		return math.NaN(), nil
	}
	return strconv.ParseFloat(v, 64)
}

func (r *Reader) readBool(line []byte) (bool, error) {
	switch util.BytesToString(line[1:]) {
	case "t":
		return true, nil
	case "f":
		return false, nil
	}
	return false, fmt.Errorf("redis: can't parse bool reply: %q", line)
}

func (r *Reader) readBigInt(line []byte) (*big.Int, error) {
	i := new(big.Int)
	if i, ok := i.SetString(util.BytesToString(line[1:]), 10); ok {
		return i, nil
	}
	return nil, fmt.Errorf("redis: can't parse bigInt reply: %q", line)
}

func (r *Reader) readStringReply(line []byte) (string, error) {
	n, err := replyLen(line)
	if err != nil {
		return "", err
	}

	b := make([]byte, n+2)
	_, err = io.ReadFull(r.rd, b)
	if err != nil {
		return "", err
	}

	return util.BytesToString(b[:n]), nil
}

func (r *Reader) readVerb(line []byte) (string, error) {
	s, err := r.readStringReply(line)
	if err != nil {
		return "", err
	}
	if len(s) < 4 || s[3] != ':' {
		return "", fmt.Errorf("redis: can't parse verbatim string reply: %q", line)
	}
	return s[4:], nil
}

func (r *Reader) readSlice(line []byte) ([]interface{}, error) {
	n, err := replyLen(line)
	if err != nil {
		return nil, err
	}

	val := make([]interface{}, n)
	for i := 0; i < len(val); i++ {
		v, err := r.ReadReply()
		if err != nil {
			if err == Nil {
				val[i] = nil
				continue
			}
			if err, ok := err.(RedisError); ok {
				val[i] = err
				continue
			}
			return nil, err
		}
		val[i] = v
	}
	return val, nil
}

func (r *Reader) readMap(line []byte) (map[interface{}]interface{}, error) {
	n, err := replyLen(line)
	if err != nil {
		return nil, err
	}
	m := make(map[interface{}]interface{}, n)
	for i := 0; i < n; i++ {
		k, err := r.ReadReply()
		if err != nil {
			return nil, err
		}

		// Reject unhashable keys (arrays/maps) before they are used as a map
		// key, which would otherwise panic. This check must run before the
		// value is read so it also guards the Nil and RedisError paths below,
		// which write the key into the map and continue.
		switch k.(type) {
		case []interface{}, map[interface{}]interface{}:
			return nil, fmt.Errorf("redis: RESP3 map key must be a scalar type, got %T", k)
		}

		v, err := r.ReadReply()
		if err != nil {
			if err == Nil {
				m[k] = nil
				continue
			}
			if err, ok := err.(RedisError); ok {
				m[k] = err
				continue
			}
			return nil, err
		}

		m[k] = v
	}
	return m, nil
}

// -------------------------------

func (r *Reader) ReadInt() (int64, error) {
	line, err := r.ReadLine()
	if err != nil {
		return 0, err
	}
	switch line[0] {
	case RespInt, RespStatus:
		return util.ParseInt(line[1:], 10, 64)
	case RespString:
		s, err := r.readStringReply(line)
		if err != nil {
			return 0, err
		}
		return util.ParseInt([]byte(s), 10, 64)
	case RespBigInt:
		b, err := r.readBigInt(line)
		if err != nil {
			return 0, err
		}
		if !b.IsInt64() {
			return 0, fmt.Errorf("bigInt(%s) value out of range", b.String())
		}
		return b.Int64(), nil
	}
	return 0, fmt.Errorf("redis: can't parse int reply: %.100q", line)
}

func (r *Reader) ReadUint() (uint64, error) {
	line, err := r.ReadLine()
	if err != nil {
		return 0, err
	}
	switch line[0] {
	case RespInt, RespStatus:
		return util.ParseUint(line[1:], 10, 64)
	case RespString:
		s, err := r.readStringReply(line)
		if err != nil {
			return 0, err
		}
		return util.ParseUint([]byte(s), 10, 64)
	case RespBigInt:
		b, err := r.readBigInt(line)
		if err != nil {
			return 0, err
		}
		if !b.IsUint64() {
			return 0, fmt.Errorf("bigInt(%s) value out of range", b.String())
		}
		return b.Uint64(), nil
	}
	return 0, fmt.Errorf("redis: can't parse uint reply: %.100q", line)
}

func (r *Reader) ReadFloat() (float64, error) {
	line, err := r.ReadLine()
	if err != nil {
		return 0, err
	}
	switch line[0] {
	case RespFloat:
		return r.readFloat(line)
	case RespStatus:
		return strconv.ParseFloat(util.BytesToString(line[1:]), 64)
	case RespString:
		s, err := r.readStringReply(line)
		if err != nil {
			return 0, err
		}
		return strconv.ParseFloat(s, 64)
	}
	return 0, fmt.Errorf("redis: can't parse float reply: %.100q", line)
}

// ReadStringInto reads a string-typed reply directly into buf, avoiding the
// per-call allocation that ReadString incurs. It returns the number of bytes
// written to buf.
//
// Supported reply types:
//   - $<n>\r\n<payload>\r\n  bulk string (the GET path; payload is read
//     straight into buf via bufio.Reader — for payloads larger than the
//     bufio buffer this is effectively zero-copy from the socket)
//   - +<status>\r\n          simple string, copied from the header line
//   - :<int>\r\n             integer, copied as its ASCII representation
//   - ,<float>\r\n           float, copied as its ASCII representation
//
// Errors, nil, push notifications, and RESP3 attributes are intercepted
// by ReadLine and surfaced through err. RESP3 verbatim strings
// (=<n>\r\n<txt:payload>\r\n) are intentionally not handled — they are
// never returned by GET-family commands, and including them re-introduces
// a hazard class where the response-type byte read from a stale `line[0]`
// after a bufio refill can be misinterpreted as the verbatim format tag.
//
// If the bulk payload does not fit in buf, an error is returned and the
// payload plus the trailing CRLF are drained from the reader so the
// connection stays aligned for the next reply. For simple-string / integer
// / float responses the payload lives in the (already-consumed) header
// line, so no drain is needed.
func (r *Reader) ReadStringInto(buf []byte) (int, error) {
	line, err := r.ReadLine()
	if err != nil {
		return 0, err
	}

	switch line[0] {
	case RespStatus:
		// Simple string — data is in the line itself.
		s := line[1:]
		if len(s) > len(buf) {
			return 0, fmt.Errorf("redis: buffer too small: need %d bytes, have %d", len(s), len(buf))
		}
		return copy(buf, s), nil

	case RespString:
		n, err := replyLen(line)
		if err != nil {
			return 0, err
		}
		if n > len(buf) {
			// Drain the payload + trailing \r\n so the next read on this
			// connection sees the start of the next reply rather than the
			// tail of this one. Otherwise the unread bytes corrupt the
			// stream and the bad connection gets handed back to the pool.
			if _, derr := r.rd.Discard(n + 2); derr != nil {
				return 0, derr
			}
			return 0, fmt.Errorf("redis: buffer too small: need %d bytes, have %d", n, len(buf))
		}
		// Read data directly into the user's buffer through the bufio.Reader.
		// bufio.Reader.Read first drains its internal buffer, then for
		// remaining data larger than its buffer size reads directly from the
		// underlying reader (socket) — effectively zero-copy.
		//
		// Fast path: when the caller VISIBLY hands over room for the trailing
		// CRLF too (len(buf) >= n+2), read the payload and the CRLF in a
		// single io.ReadFull. For large values this is one direct socket read
		// instead of a big read followed by a tiny separate Discard(2) read,
		// which is what makes GetToBuffer beat a regular Get (no payload
		// allocation and the same number of reads). The 2 trailing bytes land
		// past the returned length and are ignored.
		//
		// The gate is on len, NOT cap: a sub-slice of a larger buffer (e.g.
		// packed segments big[i*slot:(i+1)*slot]) exposes trailing capacity
		// that belongs to the caller's NEXT segment — writing the CRLF there
		// would silently corrupt caller-owned memory outside the slice they
		// passed. Callers who want the fast path pass len == payload+2 (the
		// returned length is still the payload length).
		if len(buf) >= n+2 {
			full := buf[:n+2]
			if _, err := io.ReadFull(r.rd, full); err != nil {
				return 0, err
			}
			return n, nil
		}
		// Slow path: buffer is exactly large enough for the payload only, so
		// read the payload into it and discard the CRLF separately.
		if _, err := io.ReadFull(r.rd, buf[:n]); err != nil {
			return 0, err
		}
		if _, err := r.rd.Discard(2); err != nil {
			return 0, err
		}
		return n, nil

	case RespInt, RespFloat:
		s := line[1:]
		if len(s) > len(buf) {
			return 0, fmt.Errorf("redis: buffer too small: need %d bytes, have %d", len(s), len(buf))
		}
		return copy(buf, s), nil
	}

	return 0, fmt.Errorf("redis: can't parse reply=%.100q reading string into buffer", line)
}

func (r *Reader) ReadString() (string, error) {
	line, err := r.ReadLine()
	if err != nil {
		return "", err
	}

	switch line[0] {
	case RespStatus, RespInt, RespFloat:
		return string(line[1:]), nil
	case RespString:
		return r.readStringReply(line)
	case RespBool:
		b, err := r.readBool(line)
		return strconv.FormatBool(b), err
	case RespVerbatim:
		return r.readVerb(line)
	case RespBigInt:
		b, err := r.readBigInt(line)
		if err != nil {
			return "", err
		}
		return b.String(), nil
	}
	return "", fmt.Errorf("redis: can't parse reply=%.100q reading string", line)
}

func (r *Reader) ReadBool() (bool, error) {
	s, err := r.ReadString()
	if err != nil {
		return false, err
	}
	return s == "OK" || s == "1" || s == "true", nil
}

func (r *Reader) ReadSlice() ([]interface{}, error) {
	line, err := r.ReadLine()
	if err != nil {
		return nil, err
	}
	return r.readSlice(line)
}

// ReadFixedArrayLen read fixed array length.
func (r *Reader) ReadFixedArrayLen(fixedLen int) error {
	n, err := r.ReadArrayLen()
	if err != nil {
		return err
	}
	if n != fixedLen {
		return fmt.Errorf("redis: got %d elements in the array, wanted %d", n, fixedLen)
	}
	return nil
}

// ReadArrayLen Read and return the length of the array.
func (r *Reader) ReadArrayLen() (int, error) {
	line, err := r.ReadLine()
	if err != nil {
		return 0, err
	}
	switch line[0] {
	case RespArray, RespSet, RespPush:
		return replyLen(line)
	default:
		return 0, fmt.Errorf("redis: can't parse array/set/push reply: %.100q", line)
	}
}

// ReadFixedMapLen reads fixed map length.
func (r *Reader) ReadFixedMapLen(fixedLen int) error {
	n, err := r.ReadMapLen()
	if err != nil {
		return err
	}
	if n != fixedLen {
		return fmt.Errorf("redis: got %d elements in the map, wanted %d", n, fixedLen)
	}
	return nil
}

// ReadMapLen reads the length of the map type.
// If responding to the array type (RespArray/RespSet/RespPush),
// it must be a multiple of 2 and return n/2.
// Other types will return an error.
func (r *Reader) ReadMapLen() (int, error) {
	line, err := r.ReadLine()
	if err != nil {
		return 0, err
	}
	switch line[0] {
	case RespMap:
		return replyLen(line)
	case RespArray, RespSet, RespPush:
		// Some commands and RESP2 protocol may respond to array types.
		n, err := replyLen(line)
		if err != nil {
			return 0, err
		}
		if n%2 != 0 {
			return 0, fmt.Errorf("redis: the length of the array must be a multiple of 2, got: %d", n)
		}
		return n / 2, nil
	default:
		return 0, fmt.Errorf("redis: can't parse map reply: %.100q", line)
	}
}

// DiscardNext read and discard the data represented by the next line.
func (r *Reader) DiscardNext() error {
	line, err := r.readLine()
	if err != nil {
		return err
	}
	return r.Discard(line)
}

// Discard the data represented by line.
func (r *Reader) Discard(line []byte) (err error) {
	if len(line) == 0 {
		return errors.New("redis: invalid line")
	}
	switch line[0] {
	case RespStatus, RespError, RespInt, RespNil, RespFloat, RespBool, RespBigInt:
		return nil
	}

	n, err := replyLen(line)
	if err != nil {
		if err == Nil {
			// A nil reply ($-1, =-1, !-1, *-1, %-1) carries no payload; the
			// header line was already consumed by readLine, so there is
			// nothing to discard. Falling through would Discard(n+2)==2 bytes
			// that belong to the next reply and desync the stream, matching
			// how readRawReplyBuf/readRawReplyWriteTo already treat Nil.
			return nil
		}
		return err
	}

	switch line[0] {
	case RespBlobError, RespString, RespVerbatim:
		// +\r\n
		_, err = r.rd.Discard(n + 2)
		return err
	case RespArray, RespSet, RespPush:
		for i := 0; i < n; i++ {
			if err = r.DiscardNext(); err != nil {
				return err
			}
		}
		return nil
	case RespMap, RespAttr:
		// Iterate over the n key/value pairs rather than n*2 elements: a count
		// above MaxInt/2 makes n*2 overflow to a negative loop bound, which
		// would skip the body entirely and return nil, leaving the map bytes in
		// the stream for the next reply to consume (a silent desync).
		for i := 0; i < n; i++ {
			if err = r.DiscardNext(); err != nil {
				return err
			}
			if err = r.DiscardNext(); err != nil {
				return err
			}
		}
		return nil
	}

	return fmt.Errorf("redis: can't parse %.100q", line)
}

func replyLen(line []byte) (n int, err error) {
	n, err = util.Atoi(line[1:])
	if err != nil {
		return 0, err
	}

	if n < -1 {
		return 0, fmt.Errorf("redis: invalid reply: %q", line)
	}

	switch line[0] {
	case RespString, RespVerbatim, RespBlobError,
		RespArray, RespSet, RespPush, RespMap, RespAttr:
		if n == -1 {
			return 0, Nil
		}
	}
	return n, nil
}

// IsNilReply detects redis.Nil of RESP2.
func IsNilReply(line []byte) bool {
	return len(line) == 3 &&
		(line[0] == RespString || line[0] == RespArray) &&
		line[1] == '-' && line[2] == '1'
}

// ReadRawReply reads the next RESP reply and returns it as raw bytes without parsing.
func (r *Reader) ReadRawReply() ([]byte, error) {
	return r.readRawReplyBuf(nil)
}

func (r *Reader) readRawReplyBuf(buf []byte) ([]byte, error) {
	line, err := r.readLine()
	if err != nil {
		return buf, err
	}

	buf = append(buf, line...)
	buf = append(buf, '\r', '\n')

	switch line[0] {
	case RespStatus, RespError, RespInt, RespNil, RespFloat, RespBool, RespBigInt:
		return buf, nil

	case RespString, RespVerbatim, RespBlobError:
		n, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return buf, nil
			}
			return buf, err
		}
		curLen := len(buf)
		buf = append(buf, make([]byte, n+2)...)
		_, err = io.ReadFull(r.rd, buf[curLen:])
		return buf, err

	case RespArray, RespSet, RespPush:
		n, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return buf, nil
			}
			return buf, err
		}
		for i := 0; i < n; i++ {
			buf, err = r.readRawReplyBuf(buf)
			if err != nil {
				return buf, err
			}
		}
		return buf, nil

	case RespMap:
		n, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return buf, nil
			}
			return buf, err
		}
		for i := 0; i < n; i++ {
			for pair := 0; pair < 2; pair++ {
				buf, err = r.readRawReplyBuf(buf)
				if err != nil {
					return buf, err
				}
			}
		}
		return buf, nil

	case RespAttr:
		// Per RESP3 spec, an attribute is always followed by the actual command reply.
		// We need to read the attribute's key-value pairs AND the following reply.
		n, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return buf, nil
			}
			return buf, err
		}
		// Read the attribute key-value pairs. Iterate over pairs rather than
		// n*2 elements so a count above MaxInt/2 can't overflow int to a
		// negative loop bound and skip the body.
		for i := 0; i < n; i++ {
			for pair := 0; pair < 2; pair++ {
				buf, err = r.readRawReplyBuf(buf)
				if err != nil {
					return buf, err
				}
			}
		}
		// Read the command reply that follows the attribute
		return r.readRawReplyBuf(buf)
	}

	return buf, fmt.Errorf("redis: can't read raw reply: %.100q", line)
}

var crlf = []byte{'\r', '\n'}

// ReadRawReplyWriteTo streams the next RESP reply directly to w without intermediate allocations.
// Returns the number of bytes written and any error encountered.
func (r *Reader) ReadRawReplyWriteTo(w io.Writer) (int64, error) {
	return r.readRawReplyWriteTo(w)
}

func (r *Reader) readRawReplyWriteTo(w io.Writer) (int64, error) {
	line, err := r.readLine()
	if err != nil {
		return 0, err
	}

	var written int64
	n, err := w.Write(line)
	written += int64(n)
	if err != nil {
		return written, err
	}
	n, err = w.Write(crlf)
	written += int64(n)
	if err != nil {
		return written, err
	}

	switch line[0] {
	case RespStatus, RespError, RespInt, RespNil, RespFloat, RespBool, RespBigInt:
		return written, nil

	case RespString, RespVerbatim, RespBlobError:
		dataLen, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return written, nil
			}
			return written, err
		}
		copied, err := io.CopyN(w, r.rd, int64(dataLen)+2)
		written += copied
		return written, err

	case RespArray, RespSet, RespPush:
		count, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return written, nil
			}
			return written, err
		}
		for i := 0; i < count; i++ {
			n, err := r.readRawReplyWriteTo(w)
			written += n
			if err != nil {
				return written, err
			}
		}
		return written, nil

	case RespMap:
		count, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return written, nil
			}
			return written, err
		}
		for i := 0; i < count; i++ {
			for pair := 0; pair < 2; pair++ {
				n, err := r.readRawReplyWriteTo(w)
				written += n
				if err != nil {
					return written, err
				}
			}
		}
		return written, nil

	case RespAttr:
		// Per RESP3 spec, an attribute is always followed by the actual command reply.
		// We need to read the attribute's key-value pairs AND the following reply.
		count, err := replyLen(line)
		if err != nil {
			if err == Nil {
				return written, nil
			}
			return written, err
		}
		// Read the attribute key-value pairs. Iterate over pairs rather than
		// count*2 elements so a count above MaxInt/2 can't overflow int to a
		// negative loop bound and skip the body.
		for i := 0; i < count; i++ {
			for pair := 0; pair < 2; pair++ {
				n, err := r.readRawReplyWriteTo(w)
				written += n
				if err != nil {
					return written, err
				}
			}
		}
		// Read the command reply that follows the attribute
		n, err := r.readRawReplyWriteTo(w)
		written += n
		return written, err
	}

	return written, fmt.Errorf("redis: can't read raw reply: %.100q", line)
}
