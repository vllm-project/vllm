package proto

import (
	"encoding"
	"fmt"
	"io"
	"net"
	"strconv"
	"time"

	"github.com/redis/go-redis/v9/internal/util"
)

type writer interface {
	io.Writer
	io.ByteWriter
	// WriteString implement io.StringWriter.
	WriteString(s string) (n int, err error)
}

type Writer struct {
	writer

	lenBuf []byte
	numBuf []byte
}

func NewWriter(wr writer) *Writer {
	return &Writer{
		writer: wr,

		lenBuf: make([]byte, 64),
		numBuf: make([]byte, 64),
	}
}

func (w *Writer) WriteArgs(args []interface{}) error {
	if err := w.WriteByte(RespArray); err != nil {
		return err
	}

	if err := w.writeLen(len(args)); err != nil {
		return err
	}

	for _, arg := range args {
		if err := w.WriteArg(arg); err != nil {
			return err
		}
	}

	return nil
}

func (w *Writer) writeLen(n int) error {
	w.lenBuf = strconv.AppendUint(w.lenBuf[:0], uint64(n), 10)
	w.lenBuf = append(w.lenBuf, '\r', '\n')
	_, err := w.Write(w.lenBuf)
	return err
}

func (w *Writer) WriteArg(v interface{}) error {
	switch v := v.(type) {
	case nil:
		return w.string("")
	case string:
		return w.string(v)
	case *string:
		if v == nil {
			return w.string("")
		}
		return w.string(*v)
	case []byte:
		return w.bytes(v)
	case int:
		return w.int(int64(v))
	case *int:
		if v == nil {
			return w.int(0)
		}
		return w.int(int64(*v))
	case int8:
		return w.int(int64(v))
	case *int8:
		if v == nil {
			return w.int(0)
		}
		return w.int(int64(*v))
	case int16:
		return w.int(int64(v))
	case *int16:
		if v == nil {
			return w.int(0)
		}
		return w.int(int64(*v))
	case int32:
		return w.int(int64(v))
	case *int32:
		if v == nil {
			return w.int(0)
		}
		return w.int(int64(*v))
	case int64:
		return w.int(v)
	case *int64:
		if v == nil {
			return w.int(0)
		}
		return w.int(*v)
	case uint:
		return w.uint(uint64(v))
	case *uint:
		if v == nil {
			return w.uint(0)
		}
		return w.uint(uint64(*v))
	case uint8:
		return w.uint(uint64(v))
	case *uint8:
		if v == nil {
			return w.uint(0)
		}
		return w.uint(uint64(*v))
	case uint16:
		return w.uint(uint64(v))
	case *uint16:
		if v == nil {
			return w.uint(0)
		}
		return w.uint(uint64(*v))
	case uint32:
		return w.uint(uint64(v))
	case *uint32:
		if v == nil {
			return w.uint(0)
		}
		return w.uint(uint64(*v))
	case uint64:
		return w.uint(v)
	case *uint64:
		if v == nil {
			return w.uint(0)
		}
		return w.uint(*v)
	case float32:
		return w.float(float64(v))
	case *float32:
		if v == nil {
			return w.float(0)
		}
		return w.float(float64(*v))
	case float64:
		return w.float(v)
	case *float64:
		if v == nil {
			return w.float(0)
		}
		return w.float(*v)
	case bool:
		if v {
			return w.int(1)
		}
		return w.int(0)
	case *bool:
		if v == nil {
			return w.int(0)
		}
		if *v {
			return w.int(1)
		}
		return w.int(0)
	case time.Time:
		w.numBuf = v.AppendFormat(w.numBuf[:0], time.RFC3339Nano)
		return w.bytes(w.numBuf)
	case *time.Time:
		if v == nil {
			v = &time.Time{}
		}
		w.numBuf = v.AppendFormat(w.numBuf[:0], time.RFC3339Nano)
		return w.bytes(w.numBuf)
	case time.Duration:
		return w.int(v.Nanoseconds())
	case *time.Duration:
		if v == nil {
			return w.int(0)
		}
		return w.int(v.Nanoseconds())
	case encoding.BinaryMarshaler:
		b, err := v.MarshalBinary()
		if err != nil {
			return err
		}
		return w.bytes(b)
	case net.IP:
		return w.bytes(v)
	default:
		return fmt.Errorf(
			"redis: can't marshal %T (implement encoding.BinaryMarshaler)", v)
	}
}

func (w *Writer) bytes(b []byte) error {
	if err := w.WriteByte(RespString); err != nil {
		return err
	}

	if err := w.writeLen(len(b)); err != nil {
		return err
	}

	if _, err := w.Write(b); err != nil {
		return err
	}

	return w.crlf()
}

func (w *Writer) string(s string) error {
	return w.bytes(util.StringToBytes(s))
}

func (w *Writer) uint(n uint64) error {
	w.numBuf = strconv.AppendUint(w.numBuf[:0], n, 10)
	return w.bytes(w.numBuf)
}

func (w *Writer) int(n int64) error {
	w.numBuf = strconv.AppendInt(w.numBuf[:0], n, 10)
	return w.bytes(w.numBuf)
}

func (w *Writer) float(f float64) error {
	w.numBuf = strconv.AppendFloat(w.numBuf[:0], f, 'f', -1, 64)
	return w.bytes(w.numBuf)
}

func (w *Writer) crlf() error {
	if err := w.WriteByte('\r'); err != nil {
		return err
	}
	return w.WriteByte('\n')
}
