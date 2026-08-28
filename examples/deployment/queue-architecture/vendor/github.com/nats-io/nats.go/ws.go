// Copyright 2021-2023 The NATS Authors
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

package nats

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	mrand "math/rand"
	"net/http"
	"net/url"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/klauspost/compress/flate"
)

type wsOpCode int

const (
	// From https://tools.ietf.org/html/rfc6455#section-5.2
	wsTextMessage   = wsOpCode(1)
	wsBinaryMessage = wsOpCode(2)
	wsCloseMessage  = wsOpCode(8)
	wsPingMessage   = wsOpCode(9)
	wsPongMessage   = wsOpCode(10)

	wsFinalBit = 1 << 7
	wsRsv1Bit  = 1 << 6 // Used for compression, from https://tools.ietf.org/html/rfc7692#section-6
	wsRsv2Bit  = 1 << 5
	wsRsv3Bit  = 1 << 4

	wsMaskBit = 1 << 7

	wsContinuationFrame     = 0
	wsMaxFrameHeaderSize    = 14
	wsMaxControlPayloadSize = 125
	wsCloseSatusSize        = 2

	// wsMaxMsgPayloadMultiple is the multiplier applied to MaxPayload to
	// determine the maximum WebSocket frame size.
	wsMaxMsgPayloadMultiple = 8
	// wsMaxMsgPayloadLimit is the absolute cap on WebSocket frame size (64MB).
	wsMaxMsgPayloadLimit = 64 * 1024 * 1024

	// From https://tools.ietf.org/html/rfc6455#section-11.7
	wsCloseStatusNormalClosure      = 1000
	wsCloseStatusNoStatusReceived   = 1005
	wsCloseStatusAbnormalClosure    = 1006
	wsCloseStatusInvalidPayloadData = 1007

	wsScheme    = "ws"
	wsSchemeTLS = "wss"

	wsPMCExtension      = "permessage-deflate" // per-message compression
	wsPMCSrvNoCtx       = "server_no_context_takeover"
	wsPMCCliNoCtx       = "client_no_context_takeover"
	wsPMCReqHeaderValue = wsPMCExtension + "; " + wsPMCSrvNoCtx + "; " + wsPMCCliNoCtx
)

// From https://tools.ietf.org/html/rfc6455#section-1.3
var wsGUID = []byte("258EAFA5-E914-47DA-95CA-C5AB0DC85B11")

var compressFinalBlock = []byte{0x00, 0x00, 0xff, 0xff, 0x01, 0x00, 0x00, 0xff, 0xff}

type websocketReader struct {
	r        io.Reader
	pending  [][]byte
	compress bool
	ib       []byte
	ff       bool
	fc       bool
	nl       bool
	dc       *wsDecompressor
	nc       *Conn
	closeErr error
}

type wsDecompressor struct {
	flate io.ReadCloser
	bufs  [][]byte
	off   int
}

type websocketWriter struct {
	w          io.Writer
	compress   bool
	compressor *flate.Writer
	ctrlFrames [][]byte // pending frames that should be sent at the next Write()
	cm         []byte   // close message that needs to be sent when everything else has been sent
	cmDone     bool     // a close message has been added or sent (never going back to false)
	noMoreSend bool     // if true, even if there is a Write() call, we should not send anything
}

func (d *wsDecompressor) Read(dst []byte) (int, error) {
	if len(dst) == 0 {
		return 0, nil
	}
	if len(d.bufs) == 0 {
		return 0, io.EOF
	}
	copied := 0
	rem := len(dst)
	for buf := d.bufs[0]; buf != nil && rem > 0; {
		n := min(len(buf[d.off:]), rem)
		copy(dst[copied:], buf[d.off:d.off+n])
		copied += n
		rem -= n
		d.off += n
		buf = d.nextBuf()
	}
	return copied, nil
}

func (d *wsDecompressor) nextBuf() []byte {
	// We still have remaining data in the first buffer
	if d.off != len(d.bufs[0]) {
		return d.bufs[0]
	}
	// We read the full first buffer. Reset offset.
	d.off = 0
	// We were at the last buffer, so we are done.
	if len(d.bufs) == 1 {
		d.bufs = nil
		return nil
	}
	// Here we move to the next buffer.
	d.bufs = d.bufs[1:]
	return d.bufs[0]
}

func (d *wsDecompressor) ReadByte() (byte, error) {
	if len(d.bufs) == 0 {
		return 0, io.EOF
	}
	b := d.bufs[0][d.off]
	d.off++
	d.nextBuf()
	return b, nil
}

func (d *wsDecompressor) addBuf(b []byte) {
	d.bufs = append(d.bufs, b)
}

func (d *wsDecompressor) decompress() ([]byte, error) {
	d.off = 0
	// As per https://tools.ietf.org/html/rfc7692#section-7.2.2
	// add 0x00, 0x00, 0xff, 0xff and then a final block so that flate reader
	// does not report unexpected EOF.
	d.bufs = append(d.bufs, compressFinalBlock)
	// Create or reset the decompressor with his object (wsDecompressor)
	// that provides Read() and ReadByte() APIs that will consume from
	// the compressed buffers (d.bufs).
	if d.flate == nil {
		d.flate = flate.NewReader(d)
	} else {
		d.flate.(flate.Resetter).Reset(d, nil)
	}
	b, err := io.ReadAll(d.flate)
	// Now reset the compressed buffers list
	d.bufs = nil
	return b, err
}

func wsNewReader(r io.Reader) *websocketReader {
	return &websocketReader{r: r, ff: true}
}

// maxFrameSize returns the maximum allowed WebSocket frame size based on the
// negotiated MaxPayload. This mirrors the server-side wsMaxMessageSize logic.
func (r *websocketReader) maxFrameSize() uint64 {
	if r.nc != nil {
		mp := r.nc.info.MaxPayload
		if mp > 0 && uint64(mp) <= wsMaxMsgPayloadLimit/wsMaxMsgPayloadMultiple {
			return uint64(mp) * wsMaxMsgPayloadMultiple
		}
	}
	return wsMaxMsgPayloadLimit
}

// From now on, reads will be from the readLoop and we will need to
// acquire the connection lock should we have to send/write a control
// message from handleControlFrame.
//
// Note: this runs under the connection lock.
func (r *websocketReader) doneWithConnect() {
	r.nl = true
}

func (r *websocketReader) Read(p []byte) (int, error) {
	var err error
	var buf []byte

	if l := len(r.ib); l > 0 {
		buf = r.ib
		r.ib = nil
	} else {
		if len(r.pending) > 0 {
			return r.drainPending(p), nil
		}

		// If we have a deferred close error (from a previous Read that
		// had both data frames and a close frame), return it now that
		// pending data has been drained.
		if r.closeErr != nil {
			err := r.closeErr
			r.closeErr = nil
			return 0, err
		}

		// Get some data from the underlying reader.
		n, err := r.r.Read(p)
		if err != nil {
			return 0, err
		}
		buf = p[:n]
	}

	// Now parse this and decode frames. We will possibly read more to
	// ensure that we get a full frame.
	var (
		tmpBuf []byte
		pos    int
		max    = len(buf)
		rem    = 0
	)
	for pos < max {
		b0 := buf[pos]
		frameType := wsOpCode(b0 & 0xF)
		final := b0&wsFinalBit != 0
		compressed := b0&wsRsv1Bit != 0
		pos++

		tmpBuf, pos, err = wsGet(r.r, buf, pos, 1)
		if err != nil {
			return 0, err
		}
		b1 := tmpBuf[0]

		// Store size in case it is < 125
		rem = int(b1 & 0x7F)

		switch frameType {
		case wsPingMessage, wsPongMessage, wsCloseMessage:
			if rem > wsMaxControlPayloadSize {
				return 0, fmt.Errorf(
					"control frame length bigger than maximum allowed of %v bytes",
					wsMaxControlPayloadSize)
			}
			if compressed {
				return 0, errors.New("control frame should not be compressed")
			}
			if !final {
				return 0, errors.New("control frame does not have final bit set")
			}
		case wsTextMessage, wsBinaryMessage:
			if !r.ff {
				return 0, errors.New("new message started before final frame for previous message was received")
			}
			r.ff = final
			r.fc = compressed
		case wsContinuationFrame:
			// Compressed bit must be only set in the first frame
			if r.ff || compressed {
				return 0, errors.New("invalid continuation frame")
			}
			r.ff = final
		default:
			return 0, fmt.Errorf("unknown opcode %v", frameType)
		}

		// If the encoded size is <= 125, then `rem` is simply the remainder size of the
		// frame. If it is 126, then the actual size is encoded as a uint16. For larger
		// frames, `rem` will initially be 127 and the actual size is encoded as a uint64.
		switch rem {
		case 126:
			tmpBuf, pos, err = wsGet(r.r, buf, pos, 2)
			if err != nil {
				return 0, err
			}
			rem = int(binary.BigEndian.Uint16(tmpBuf))
		case 127:
			tmpBuf, pos, err = wsGet(r.r, buf, pos, 8)
			if err != nil {
				return 0, err
			}
			rem64 := binary.BigEndian.Uint64(tmpBuf)
			if rem64&(1<<63) != 0 {
				return 0, errors.New("invalid websocket frame: MSB set in 64-bit payload length")
			}
			if rem64 > r.maxFrameSize() {
				return 0, fmt.Errorf("websocket frame too large: %d", rem64)
			}
			rem = int(rem64)
		}

		// Handle control messages in place...
		if wsIsControlFrame(frameType) {
			pos, err = r.handleControlFrame(frameType, buf, pos, rem)
			if err != nil {
				// If we already have pending data (e.g. a -ERR message
				// that arrived before this close frame), defer the error
				// so the pending data can be returned to the caller first.
				if len(r.pending) > 0 {
					r.closeErr = err
					break
				}
				return 0, err
			}
			rem = 0
			continue
		}

		var b []byte
		// This ensures that we get the full payload for this frame.
		b, pos, err = wsGet(r.r, buf, pos, rem)
		if err != nil {
			return 0, err
		}
		// We read the full frame.
		rem = 0
		addToPending := true
		if r.fc {
			// Don't add to pending if we are not dealing with the final frame.
			addToPending = r.ff
			// Add the compressed payload buffer to the list.
			r.addCBuf(b)
			// Decompress only when this is the final frame.
			if r.ff {
				b, err = r.dc.decompress()
				if err != nil {
					return 0, err
				}
				r.fc = false
			}
		} else if r.compress {
			b = bytes.Clone(b)
		}
		// Add to the pending list if dealing with uncompressed frames or
		// after we have received the full compressed message and decompressed it.
		if addToPending {
			r.pending = append(r.pending, b)
		}
	}
	// In case of compression, there may be nothing to drain
	if len(r.pending) > 0 {
		return r.drainPending(p), nil
	}
	return 0, nil
}

func (r *websocketReader) addCBuf(b []byte) {
	if r.dc == nil {
		r.dc = &wsDecompressor{}
	}
	// Add a copy of the incoming buffer to the list of compressed buffers.
	r.dc.addBuf(append([]byte(nil), b...))
}

func (r *websocketReader) drainPending(p []byte) int {
	var n int
	var max = len(p)

	for i, buf := range r.pending {
		if n+len(buf) <= max {
			copy(p[n:], buf)
			n += len(buf)
		} else {
			// Is there room left?
			if n < max {
				// Write the partial and update this slice.
				rem := max - n
				copy(p[n:], buf[:rem])
				n += rem
				r.pending[i] = buf[rem:]
			}
			// These are the remaining slices that will need to be used at
			// the next Read() call.
			r.pending = r.pending[i:]
			return n
		}
	}
	r.pending = r.pending[:0]
	return n
}

func wsGet(r io.Reader, buf []byte, pos, needed int) ([]byte, int, error) {
	avail := len(buf) - pos
	if avail >= needed {
		return buf[pos : pos+needed], pos + needed, nil
	}
	b := make([]byte, needed)
	start := copy(b, buf[pos:])
	for start != needed {
		n, err := r.Read(b[start:cap(b)])
		start += n
		if err != nil {
			return b, start, err
		}
	}
	return b, pos + avail, nil
}

func (r *websocketReader) handleControlFrame(frameType wsOpCode, buf []byte, pos, rem int) (int, error) {
	var payload []byte
	var err error

	if rem > 0 {
		payload, pos, err = wsGet(r.r, buf, pos, rem)
		if err != nil {
			return pos, err
		}
	}
	switch frameType {
	case wsCloseMessage:
		status := wsCloseStatusNoStatusReceived
		var body string
		lp := len(payload)
		// If there is a payload, the status is represented as a 2-byte
		// unsigned integer (in network byte order). Then, there may be an
		// optional body.
		hasStatus, hasBody := lp >= wsCloseSatusSize, lp > wsCloseSatusSize
		if hasStatus {
			// Decode the status
			status = int(binary.BigEndian.Uint16(payload[:wsCloseSatusSize]))
			// Now if there is a body, capture it and make sure this is a valid UTF-8.
			if hasBody {
				body = string(payload[wsCloseSatusSize:])
				if !utf8.ValidString(body) {
					// https://tools.ietf.org/html/rfc6455#section-5.5.1
					// If body is present, it must be a valid utf8
					status = wsCloseStatusInvalidPayloadData
					body = "invalid utf8 body in close frame"
				}
			}
		}
		r.nc.wsEnqueueCloseMsg(r.nl, status, body)
		// Return io.EOF so that readLoop will close the connection as client closed
		// after processing pending buffers.
		return pos, io.EOF
	case wsPingMessage:
		r.nc.wsEnqueueControlMsg(r.nl, wsPongMessage, payload)
	case wsPongMessage:
		// Nothing to do..
	}
	return pos, nil
}

func (w *websocketWriter) Write(p []byte) (int, error) {
	if w.noMoreSend {
		return 0, nil
	}
	var total int
	var n int
	var err error
	// If there are control frames, they can be sent now. Actually spec says
	// that they should be sent ASAP, so we will send before any application data.
	if len(w.ctrlFrames) > 0 {
		n, err = w.writeCtrlFrames()
		if err != nil {
			return n, err
		}
		total += n
	}
	// Do the following only if there is something to send.
	// We will end with checking for need to send close message.
	if len(p) > 0 {
		if w.compress {
			buf := &bytes.Buffer{}
			if w.compressor == nil {
				w.compressor, _ = flate.NewWriter(buf, flate.BestSpeed)
			} else {
				w.compressor.Reset(buf)
			}
			if n, err = w.compressor.Write(p); err != nil {
				return n, err
			}
			if err = w.compressor.Flush(); err != nil {
				return n, err
			}
			b := buf.Bytes()
			p = b[:len(b)-4]
		}
		fh, key := wsCreateFrameHeader(w.compress, wsBinaryMessage, len(p))
		wsMaskBuf(key, p)
		n, err = w.w.Write(fh)
		total += n
		if err == nil {
			n, err = w.w.Write(p)
			total += n
		}
	}
	if err == nil && w.cm != nil {
		n, err = w.writeCloseMsg()
		total += n
	}
	return total, err
}

func (w *websocketWriter) writeCtrlFrames() (int, error) {
	var (
		n     int
		total int
		i     int
		err   error
	)
	for ; i < len(w.ctrlFrames); i++ {
		buf := w.ctrlFrames[i]
		n, err = w.w.Write(buf)
		total += n
		if err != nil {
			break
		}
	}
	if i != len(w.ctrlFrames) {
		w.ctrlFrames = w.ctrlFrames[i+1:]
	} else {
		w.ctrlFrames = w.ctrlFrames[:0]
	}
	return total, err
}

func (w *websocketWriter) writeCloseMsg() (int, error) {
	n, err := w.w.Write(w.cm)
	w.cm, w.noMoreSend = nil, true
	return n, err
}

func wsMaskBuf(key, buf []byte) {
	for i := 0; i < len(buf); i++ {
		buf[i] ^= key[i&3]
	}
}

// Create the frame header.
// Encodes the frame type and optional compression flag, and the size of the payload.
func wsCreateFrameHeader(compressed bool, frameType wsOpCode, l int) ([]byte, []byte) {
	fh := make([]byte, wsMaxFrameHeaderSize)
	n, key := wsFillFrameHeader(fh, compressed, frameType, l)
	return fh[:n], key
}

func wsFillFrameHeader(fh []byte, compressed bool, frameType wsOpCode, l int) (int, []byte) {
	var n int
	b := byte(frameType)
	b |= wsFinalBit
	if compressed {
		b |= wsRsv1Bit
	}
	b1 := byte(wsMaskBit)
	switch {
	case l <= 125:
		n = 2
		fh[0] = b
		fh[1] = b1 | byte(l)
	case l < 65536:
		n = 4
		fh[0] = b
		fh[1] = b1 | 126
		binary.BigEndian.PutUint16(fh[2:], uint16(l))
	default:
		n = 10
		fh[0] = b
		fh[1] = b1 | 127
		binary.BigEndian.PutUint64(fh[2:], uint64(l))
	}
	var key []byte
	var keyBuf [4]byte
	if _, err := io.ReadFull(rand.Reader, keyBuf[:4]); err != nil {
		kv := mrand.Int31()
		binary.LittleEndian.PutUint32(keyBuf[:4], uint32(kv))
	}
	copy(fh[n:], keyBuf[:4])
	key = fh[n : n+4]
	n += 4
	return n, key
}

func (nc *Conn) wsInitHandshake(u *url.URL) error {
	compress := nc.Opts.Compression
	tlsRequired := u.Scheme == wsSchemeTLS || nc.Opts.Secure || nc.Opts.TLSConfig != nil || nc.Opts.TLSCertCB != nil || nc.Opts.RootCAsCB != nil
	// Do TLS here as needed.
	if tlsRequired {
		if err := nc.makeTLSConn(); err != nil {
			return err
		}
	} else {
		nc.bindToNewConn()
	}

	var err error

	// For http request, we need the passed URL to contain either http or https scheme.
	scheme := "http"
	if tlsRequired {
		scheme = "https"
	}
	ustr := fmt.Sprintf("%s://%s", scheme, u.Host)

	if nc.Opts.ProxyPath != "" {
		proxyPath := nc.Opts.ProxyPath
		if !strings.HasPrefix(proxyPath, "/") {
			proxyPath = "/" + proxyPath
		}
		ustr += proxyPath
	} else if u.Path != "" {
		ustr += u.Path
	}
	if u.RawQuery != "" {
		ustr += "?" + u.RawQuery
	}

	u, err = url.Parse(ustr)
	if err != nil {
		return err
	}
	req := &http.Request{
		Method:     "GET",
		URL:        u,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Host:       u.Host,
	}
	wsKey, err := wsMakeChallengeKey()
	if err != nil {
		return err
	}

	req.Header["Upgrade"] = []string{"websocket"}
	req.Header["Connection"] = []string{"Upgrade"}
	req.Header["Sec-WebSocket-Key"] = []string{wsKey}
	req.Header["Sec-WebSocket-Version"] = []string{"13"}
	if compress {
		req.Header.Add("Sec-WebSocket-Extensions", wsPMCReqHeaderValue)
	}
	if err := nc.wsUpdateConnectionHeaders(req); err != nil {
		return err
	}
	if err := req.Write(nc.conn); err != nil {
		return err
	}

	var resp *http.Response

	br := bufio.NewReaderSize(nc.conn, 4096)
	nc.conn.SetReadDeadline(time.Now().Add(nc.Opts.Timeout))
	resp, err = http.ReadResponse(br, req)
	if err == nil &&
		(resp.StatusCode != 101 ||
			!strings.EqualFold(resp.Header.Get("Upgrade"), "websocket") ||
			!strings.EqualFold(resp.Header.Get("Connection"), "upgrade") ||
			resp.Header.Get("Sec-Websocket-Accept") != wsAcceptKey(wsKey)) {

		err = errors.New("invalid websocket connection")
	}
	// Check compression extension...
	if err == nil && compress {
		// Check that not only permessage-deflate extension is present, but that
		// we also have server and client no context take over.
		srvCompress, noCtxTakeover := wsPMCExtensionSupport(resp.Header)

		// If server does not support compression, then simply disable it in our side.
		if !srvCompress {
			compress = false
		} else if !noCtxTakeover {
			err = errors.New("compression negotiation error")
		}
	}
	if resp != nil {
		resp.Body.Close()
	}
	nc.conn.SetReadDeadline(time.Time{})
	if err != nil {
		return err
	}

	wsr := wsNewReader(nc.br.r)
	wsr.nc = nc
	wsr.compress = compress
	// We have to slurp whatever is in the bufio reader and copy to br.r
	if n := br.Buffered(); n != 0 {
		wsr.ib, _ = br.Peek(n)
	}
	nc.br.r = wsr
	nc.bw.w = &websocketWriter{w: nc.bw.w, compress: compress}
	nc.ws = true
	return nil
}

func (nc *Conn) wsClose() {
	nc.mu.Lock()
	defer nc.mu.Unlock()
	if !nc.ws {
		return
	}
	nc.wsEnqueueCloseMsgLocked(wsCloseStatusNormalClosure, _EMPTY_)
}

func (nc *Conn) wsEnqueueCloseMsg(needsLock bool, status int, payload string) {
	// In some low-level unit tests it will happen...
	if nc == nil {
		return
	}
	if needsLock {
		nc.mu.Lock()
		defer nc.mu.Unlock()
	}
	nc.wsEnqueueCloseMsgLocked(status, payload)
}

func (nc *Conn) wsEnqueueCloseMsgLocked(status int, payload string) {
	wr, ok := nc.bw.w.(*websocketWriter)
	if !ok || wr.cmDone {
		return
	}
	statusAndPayloadLen := 2 + len(payload)
	frame := make([]byte, 2+4+statusAndPayloadLen)
	n, key := wsFillFrameHeader(frame, false, wsCloseMessage, statusAndPayloadLen)
	// Set the status
	binary.BigEndian.PutUint16(frame[n:], uint16(status))
	// If there is a payload, copy
	if len(payload) > 0 {
		copy(frame[n+2:], payload)
	}
	// Mask status + payload
	wsMaskBuf(key, frame[n:n+statusAndPayloadLen])
	wr.cm = frame
	wr.cmDone = true
	nc.bw.flush()
	if c := wr.compressor; c != nil {
		c.Close()
	}
}

func (nc *Conn) wsEnqueueControlMsg(needsLock bool, frameType wsOpCode, payload []byte) {
	// In some low-level unit tests it will happen...
	if nc == nil {
		return
	}
	if needsLock {
		nc.mu.Lock()
		defer nc.mu.Unlock()
	}
	wr, ok := nc.bw.w.(*websocketWriter)
	if !ok {
		return
	}
	fh, key := wsCreateFrameHeader(false, frameType, len(payload))
	wr.ctrlFrames = append(wr.ctrlFrames, fh)
	if len(payload) > 0 {
		wsMaskBuf(key, payload)
		wr.ctrlFrames = append(wr.ctrlFrames, payload)
	}
	nc.bw.flush()
}

func (nc *Conn) wsUpdateConnectionHeaders(req *http.Request) error {
	var headers http.Header
	var err error
	if nc.Opts.WebSocketConnectionHeadersHandler != nil {
		headers, err = nc.Opts.WebSocketConnectionHeadersHandler()
		if err != nil {
			return err
		}
	} else {
		headers = nc.Opts.WebSocketConnectionHeaders
	}
	for key, values := range headers {
		for _, val := range values {
			req.Header.Add(key, val)
		}
	}
	return nil
}

func wsPMCExtensionSupport(header http.Header) (bool, bool) {
	for _, extensionList := range header["Sec-Websocket-Extensions"] {
		extensions := strings.Split(extensionList, ",")
		for _, extension := range extensions {
			extension = strings.Trim(extension, " \t")
			params := strings.Split(extension, ";")
			for i, p := range params {
				p = strings.Trim(p, " \t")
				if strings.EqualFold(p, wsPMCExtension) {
					var snc bool
					var cnc bool
					for j := i + 1; j < len(params); j++ {
						p = params[j]
						p = strings.Trim(p, " \t")
						if strings.EqualFold(p, wsPMCSrvNoCtx) {
							snc = true
						} else if strings.EqualFold(p, wsPMCCliNoCtx) {
							cnc = true
						}
						if snc && cnc {
							return true, true
						}
					}
					return true, false
				}
			}
		}
	}
	return false, false
}

func wsMakeChallengeKey() (string, error) {
	p := make([]byte, 16)
	if _, err := io.ReadFull(rand.Reader, p); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(p), nil
}

func wsAcceptKey(key string) string {
	h := sha1.New()
	h.Write([]byte(key))
	h.Write(wsGUID)
	return base64.StdEncoding.EncodeToString(h.Sum(nil))
}

// Returns true if the op code corresponds to a control frame.
func wsIsControlFrame(frameType wsOpCode) bool {
	return frameType >= wsCloseMessage
}

func isWebsocketScheme(u *url.URL) bool {
	return u.Scheme == wsScheme || u.Scheme == wsSchemeTLS
}
