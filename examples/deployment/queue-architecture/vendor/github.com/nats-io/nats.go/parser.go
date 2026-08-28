// Copyright 2012-2023 The NATS Authors
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
	"fmt"
)

type msgArg struct {
	subject []byte
	reply   []byte
	sid     int64
	hdr     int
	size    int
}

const MAX_CONTROL_LINE_SIZE = 4096

type parseState struct {
	state     int
	as        int
	drop      int
	hdr       int
	ma        msgArg
	argBuf    []byte
	msgBuf    []byte
	msgCopied bool
	scratch   [MAX_CONTROL_LINE_SIZE]byte
}

const (
	OP_START = iota
	OP_PLUS
	OP_PLUS_O
	OP_PLUS_OK
	OP_MINUS
	OP_MINUS_E
	OP_MINUS_ER
	OP_MINUS_ERR
	OP_MINUS_ERR_SPC
	MINUS_ERR_ARG
	OP_M
	OP_MS
	OP_MSG
	OP_MSG_SPC
	MSG_ARG
	MSG_PAYLOAD
	MSG_END
	OP_H
	OP_P
	OP_PI
	OP_PIN
	OP_PING
	OP_PO
	OP_PON
	OP_PONG
	OP_I
	OP_IN
	OP_INF
	OP_INFO
	OP_INFO_SPC
	INFO_ARG
)

// parse is the fast protocol parser engine.
func (nc *Conn) parse(buf []byte) error {
	var i int
	var b byte

	// Move to loop instead of range syntax to allow jumping of i
	for i = 0; i < len(buf); i++ {
		b = buf[i]

		switch nc.ps.state {
		case OP_START:
			switch b {
			case 'M', 'm':
				nc.ps.state = OP_M
				nc.ps.hdr = -1
				nc.ps.ma.hdr = -1
			case 'H', 'h':
				nc.ps.state = OP_H
				nc.ps.hdr = 0
				nc.ps.ma.hdr = 0
			case 'P', 'p':
				nc.ps.state = OP_P
			case '+':
				nc.ps.state = OP_PLUS
			case '-':
				nc.ps.state = OP_MINUS
			case 'I', 'i':
				nc.ps.state = OP_I
			default:
				goto parseErr
			}
		case OP_H:
			switch b {
			case 'M', 'm':
				nc.ps.state = OP_M
			default:
				goto parseErr
			}
		case OP_M:
			switch b {
			case 'S', 's':
				nc.ps.state = OP_MS
			default:
				goto parseErr
			}
		case OP_MS:
			switch b {
			case 'G', 'g':
				nc.ps.state = OP_MSG
			default:
				goto parseErr
			}
		case OP_MSG:
			switch b {
			case ' ', '\t':
				nc.ps.state = OP_MSG_SPC
			default:
				goto parseErr
			}
		case OP_MSG_SPC:
			switch b {
			case ' ', '\t':
				continue
			default:
				nc.ps.state = MSG_ARG
				nc.ps.as = i
			}
		case MSG_ARG:
			switch b {
			case '\r':
				nc.ps.drop = 1
			case '\n':
				var arg []byte
				if nc.ps.argBuf != nil {
					arg = nc.ps.argBuf
				} else {
					arg = buf[nc.ps.as : i-nc.ps.drop]
				}
				if err := nc.processMsgArgs(arg); err != nil {
					return err
				}
				nc.ps.drop, nc.ps.as, nc.ps.state = 0, i+1, MSG_PAYLOAD

				// jump ahead with the index. If this overruns
				// what is left we fall out and process a split buffer.
				i = nc.ps.as + nc.ps.ma.size - 1
			default:
				if nc.ps.argBuf != nil {
					nc.ps.argBuf = append(nc.ps.argBuf, b)
				}
			}
		case MSG_PAYLOAD:
			if nc.ps.msgBuf != nil {
				if len(nc.ps.msgBuf) >= nc.ps.ma.size {
					nc.processMsg(nc.ps.msgBuf)
					nc.ps.argBuf, nc.ps.msgBuf, nc.ps.msgCopied, nc.ps.state = nil, nil, false, MSG_END
				} else {
					// copy as much as we can to the buffer and skip ahead.
					toCopy := nc.ps.ma.size - len(nc.ps.msgBuf)
					avail := len(buf) - i

					if avail < toCopy {
						toCopy = avail
					}

					if toCopy > 0 {
						start := len(nc.ps.msgBuf)
						// This is needed for copy to work.
						nc.ps.msgBuf = nc.ps.msgBuf[:start+toCopy]
						copy(nc.ps.msgBuf[start:], buf[i:i+toCopy])
						// Update our index
						i = (i + toCopy) - 1
					} else {
						nc.ps.msgBuf = append(nc.ps.msgBuf, b)
					}
				}
			} else if i-nc.ps.as >= nc.ps.ma.size {
				nc.processMsg(buf[nc.ps.as:i])
				nc.ps.argBuf, nc.ps.msgBuf, nc.ps.msgCopied, nc.ps.state = nil, nil, false, MSG_END
			}
		case MSG_END:
			switch b {
			case '\n':
				nc.ps.drop, nc.ps.as, nc.ps.state = 0, i+1, OP_START
			default:
				continue
			}
		case OP_PLUS:
			switch b {
			case 'O', 'o':
				nc.ps.state = OP_PLUS_O
			default:
				goto parseErr
			}
		case OP_PLUS_O:
			switch b {
			case 'K', 'k':
				nc.ps.state = OP_PLUS_OK
			default:
				goto parseErr
			}
		case OP_PLUS_OK:
			switch b {
			case '\n':
				nc.processOK()
				nc.ps.drop, nc.ps.state = 0, OP_START
			}
		case OP_MINUS:
			switch b {
			case 'E', 'e':
				nc.ps.state = OP_MINUS_E
			default:
				goto parseErr
			}
		case OP_MINUS_E:
			switch b {
			case 'R', 'r':
				nc.ps.state = OP_MINUS_ER
			default:
				goto parseErr
			}
		case OP_MINUS_ER:
			switch b {
			case 'R', 'r':
				nc.ps.state = OP_MINUS_ERR
			default:
				goto parseErr
			}
		case OP_MINUS_ERR:
			switch b {
			case ' ', '\t':
				nc.ps.state = OP_MINUS_ERR_SPC
			default:
				goto parseErr
			}
		case OP_MINUS_ERR_SPC:
			switch b {
			case ' ', '\t':
				continue
			default:
				nc.ps.state = MINUS_ERR_ARG
				nc.ps.as = i
			}
		case MINUS_ERR_ARG:
			switch b {
			case '\r':
				nc.ps.drop = 1
			case '\n':
				var arg []byte
				if nc.ps.argBuf != nil {
					arg = nc.ps.argBuf
					nc.ps.argBuf = nil
				} else {
					arg = buf[nc.ps.as : i-nc.ps.drop]
				}
				nc.processErr(string(arg))
				nc.ps.drop, nc.ps.as, nc.ps.state = 0, i+1, OP_START
			default:
				if nc.ps.argBuf != nil {
					nc.ps.argBuf = append(nc.ps.argBuf, b)
				}
			}
		case OP_P:
			switch b {
			case 'I', 'i':
				nc.ps.state = OP_PI
			case 'O', 'o':
				nc.ps.state = OP_PO
			default:
				goto parseErr
			}
		case OP_PO:
			switch b {
			case 'N', 'n':
				nc.ps.state = OP_PON
			default:
				goto parseErr
			}
		case OP_PON:
			switch b {
			case 'G', 'g':
				nc.ps.state = OP_PONG
			default:
				goto parseErr
			}
		case OP_PONG:
			switch b {
			case '\n':
				nc.processPong()
				nc.ps.drop, nc.ps.state = 0, OP_START
			}
		case OP_PI:
			switch b {
			case 'N', 'n':
				nc.ps.state = OP_PIN
			default:
				goto parseErr
			}
		case OP_PIN:
			switch b {
			case 'G', 'g':
				nc.ps.state = OP_PING
			default:
				goto parseErr
			}
		case OP_PING:
			switch b {
			case '\n':
				nc.processPing()
				nc.ps.drop, nc.ps.state = 0, OP_START
			}
		case OP_I:
			switch b {
			case 'N', 'n':
				nc.ps.state = OP_IN
			default:
				goto parseErr
			}
		case OP_IN:
			switch b {
			case 'F', 'f':
				nc.ps.state = OP_INF
			default:
				goto parseErr
			}
		case OP_INF:
			switch b {
			case 'O', 'o':
				nc.ps.state = OP_INFO
			default:
				goto parseErr
			}
		case OP_INFO:
			switch b {
			case ' ', '\t':
				nc.ps.state = OP_INFO_SPC
			default:
				goto parseErr
			}
		case OP_INFO_SPC:
			switch b {
			case ' ', '\t':
				continue
			default:
				nc.ps.state = INFO_ARG
				nc.ps.as = i
			}
		case INFO_ARG:
			switch b {
			case '\r':
				nc.ps.drop = 1
			case '\n':
				var arg []byte
				if nc.ps.argBuf != nil {
					arg = nc.ps.argBuf
					nc.ps.argBuf = nil
				} else {
					arg = buf[nc.ps.as : i-nc.ps.drop]
				}
				nc.processAsyncInfo(arg)
				nc.ps.drop, nc.ps.as, nc.ps.state = 0, i+1, OP_START
			default:
				if nc.ps.argBuf != nil {
					nc.ps.argBuf = append(nc.ps.argBuf, b)
				}
			}
		default:
			goto parseErr
		}
	}
	// Check for split buffer scenarios
	if (nc.ps.state == MSG_ARG || nc.ps.state == MINUS_ERR_ARG || nc.ps.state == INFO_ARG) && nc.ps.argBuf == nil {
		nc.ps.argBuf = nc.ps.scratch[:0]
		nc.ps.argBuf = append(nc.ps.argBuf, buf[nc.ps.as:i-nc.ps.drop]...)
		// FIXME, check max len
	}
	// Check for split msg
	if nc.ps.state == MSG_PAYLOAD && nc.ps.msgBuf == nil {
		// We need to clone the msgArg if it is still referencing the
		// read buffer and we are not able to process the msg.
		if nc.ps.argBuf == nil {
			nc.cloneMsgArg()
		}

		// If we will overflow the scratch buffer, just create a
		// new buffer to hold the split message.
		if nc.ps.ma.size > cap(nc.ps.scratch)-len(nc.ps.argBuf) {
			lrem := len(buf[nc.ps.as:])

			nc.ps.msgBuf = make([]byte, lrem, nc.ps.ma.size)
			copy(nc.ps.msgBuf, buf[nc.ps.as:])
			nc.ps.msgCopied = true
		} else {
			nc.ps.msgBuf = nc.ps.scratch[len(nc.ps.argBuf):len(nc.ps.argBuf)]
			nc.ps.msgBuf = append(nc.ps.msgBuf, (buf[nc.ps.as:])...)
		}
	}

	return nil

parseErr:
	return fmt.Errorf("nats: Parse Error [%d]: '%s'", nc.ps.state, buf[i:])
}

// cloneMsgArg is used when the split buffer scenario has the pubArg in the existing read buffer, but
// we need to hold onto it into the next read.
func (nc *Conn) cloneMsgArg() {
	nc.ps.argBuf = nc.ps.scratch[:0]
	nc.ps.argBuf = append(nc.ps.argBuf, nc.ps.ma.subject...)
	nc.ps.argBuf = append(nc.ps.argBuf, nc.ps.ma.reply...)
	nc.ps.ma.subject = nc.ps.argBuf[:len(nc.ps.ma.subject)]
	if nc.ps.ma.reply != nil {
		nc.ps.ma.reply = nc.ps.argBuf[len(nc.ps.ma.subject):]
	}
}

const argsLenMax = 4

func (nc *Conn) processMsgArgs(arg []byte) error {
	// Use separate function for header based messages.
	if nc.ps.hdr >= 0 {
		return nc.processHeaderMsgArgs(arg)
	}

	// Unroll splitArgs to avoid runtime/heap issues
	a := [argsLenMax][]byte{}
	args := a[:0]
	start := -1
	for i, b := range arg {
		switch b {
		case ' ', '\t', '\r', '\n':
			if start >= 0 {
				args = append(args, arg[start:i])
				start = -1
			}
		default:
			if start < 0 {
				start = i
			}
		}
	}
	if start >= 0 {
		args = append(args, arg[start:])
	}

	switch len(args) {
	case 3:
		nc.ps.ma.subject = args[0]
		nc.ps.ma.sid = parseInt64(args[1])
		nc.ps.ma.reply = nil
		nc.ps.ma.size = int(parseInt64(args[2]))
	case 4:
		nc.ps.ma.subject = args[0]
		nc.ps.ma.sid = parseInt64(args[1])
		nc.ps.ma.reply = args[2]
		nc.ps.ma.size = int(parseInt64(args[3]))
	default:
		return fmt.Errorf("nats: processMsgArgs Parse Error: '%s'", arg)
	}
	if nc.ps.ma.sid < 0 {
		return fmt.Errorf("nats: processMsgArgs Bad or Missing Sid: '%s'", arg)
	}
	if nc.ps.ma.size < 0 {
		return fmt.Errorf("nats: processMsgArgs Bad or Missing Size: '%s'", arg)
	}
	return nil
}

// processHeaderMsgArgs is for a header based message.
func (nc *Conn) processHeaderMsgArgs(arg []byte) error {
	// Unroll splitArgs to avoid runtime/heap issues
	a := [argsLenMax][]byte{}
	args := a[:0]
	start := -1
	for i, b := range arg {
		switch b {
		case ' ', '\t', '\r', '\n':
			if start >= 0 {
				args = append(args, arg[start:i])
				start = -1
			}
		default:
			if start < 0 {
				start = i
			}
		}
	}
	if start >= 0 {
		args = append(args, arg[start:])
	}

	switch len(args) {
	case 4:
		nc.ps.ma.subject = args[0]
		nc.ps.ma.sid = parseInt64(args[1])
		nc.ps.ma.reply = nil
		nc.ps.ma.hdr = int(parseInt64(args[2]))
		nc.ps.ma.size = int(parseInt64(args[3]))
	case 5:
		nc.ps.ma.subject = args[0]
		nc.ps.ma.sid = parseInt64(args[1])
		nc.ps.ma.reply = args[2]
		nc.ps.ma.hdr = int(parseInt64(args[3]))
		nc.ps.ma.size = int(parseInt64(args[4]))
	default:
		return fmt.Errorf("nats: processHeaderMsgArgs Parse Error: '%s'", arg)
	}
	if nc.ps.ma.sid < 0 {
		return fmt.Errorf("nats: processHeaderMsgArgs Bad or Missing Sid: '%s'", arg)
	}
	if nc.ps.ma.hdr < 0 || nc.ps.ma.hdr > nc.ps.ma.size {
		return fmt.Errorf("nats: processHeaderMsgArgs Bad or Missing Header Size: '%s'", arg)
	}
	if nc.ps.ma.size < 0 {
		return fmt.Errorf("nats: processHeaderMsgArgs Bad or Missing Size: '%s'", arg)
	}
	return nil
}

// ASCII numbers 0-9
const (
	ascii_0 = 48
	ascii_9 = 57
)

// parseInt64 expects decimal positive numbers. We
// return -1 to signal error
func parseInt64(d []byte) (n int64) {
	if len(d) == 0 {
		return -1
	}
	for _, dec := range d {
		if dec < ascii_0 || dec > ascii_9 {
			return -1
		}
		n = n*10 + (int64(dec) - ascii_0)
	}
	return n
}
