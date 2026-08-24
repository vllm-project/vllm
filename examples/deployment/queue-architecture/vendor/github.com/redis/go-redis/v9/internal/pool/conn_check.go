//go:build linux || darwin || dragonfly || freebsd || netbsd || openbsd || solaris || illumos

package pool

import (
	"errors"
	"io"
	"net"
	"syscall"
	"time"
)

var errUnexpectedRead = errors.New("unexpected read from socket")

// connCheck checks if the connection is still alive and if there is data in the socket
// it will try to peek at the next byte without consuming it since we may want to work with it
// later on (e.g. push notifications)
func connCheck(conn net.Conn) error {
	// Reset previous timeout.
	_ = conn.SetDeadline(time.Time{})

	// Health checks deliberately inspect only the outer connection. Unwrapping a
	// buffered transport such as crypto/tls.Conn can reveal an encrypted
	// post-handshake record and make isHealthyConn call PeekReplyType on the TLS
	// stream. With the deadline cleared above, TLS may consume that control record
	// and then wait forever for application data.
	sysConn, ok := conn.(syscall.Conn)
	if !ok {
		return nil
	}
	return checkSyscallConn(sysConn)
}

func checkSyscallConn(sysConn syscall.Conn) error {
	rawConn, err := sysConn.SyscallConn()
	if err != nil {
		return err
	}

	var sysErr error

	if err := rawConn.Read(func(fd uintptr) bool {
		var buf [1]byte
		// Use MSG_PEEK to peek at data without consuming it
		n, _, err := syscall.Recvfrom(int(fd), buf[:], syscall.MSG_PEEK|syscall.MSG_DONTWAIT)

		switch {
		case n == 0 && err == nil:
			sysErr = io.EOF
		case n > 0:
			sysErr = errUnexpectedRead
		case err == syscall.EAGAIN || err == syscall.EWOULDBLOCK:
			sysErr = nil
		default:
			sysErr = err
		}
		return true
	}); err != nil {
		return err
	}

	return sysErr
}

// underlyingSyscallConn unwraps connections that expose their transport through
// NetConn (notably crypto/tls.Conn). Limit the walk so a broken wrapper cannot
// loop forever.
func underlyingSyscallConn(conn net.Conn) (syscall.Conn, bool) {
	for range 8 {
		if sysConn, ok := conn.(syscall.Conn); ok {
			return sysConn, true
		}
		unwrapper, ok := conn.(interface{ NetConn() net.Conn })
		if !ok {
			return nil, false
		}
		conn = unwrapper.NetConn()
		if conn == nil {
			return nil, false
		}
	}
	return nil, false
}

// maybeHasData checks if there is data in the socket without consuming it
func maybeHasData(conn net.Conn) bool {
	hasData, _ := checkForData(conn)
	return hasData
}

func checkForData(conn net.Conn) (bool, error) {
	// Unlike the general health check, CSC only uses this as a readiness hint
	// before a bounded read, so unwrapping TLS is safe and avoids blocking.
	_ = conn.SetDeadline(time.Time{})
	sysConn, ok := underlyingSyscallConn(conn)
	if !ok {
		return false, nil
	}
	switch err := checkSyscallConn(sysConn); err {
	case nil:
		return false, nil
	case errUnexpectedRead:
		return true, nil
	default:
		return false, err
	}
}

// needsCscReadProbe reports whether a command read may leave data hidden from
// maybeHasData. On Unix a direct syscall.Conn has no intermediate buffering;
// TLS and opaque wrappers need one bounded post-command probe.
func needsCscReadProbe(conn net.Conn) bool {
	_, direct := conn.(syscall.Conn)
	return !direct
}

// needsCscPeriodicProbe reports whether the platform can inspect the transport
// at all. Opaque wrappers get a throttled bounded fallback so invalidations that
// arrive after the post-command probe are still eventually consumed.
func needsCscPeriodicProbe(conn net.Conn) bool {
	_, ok := underlyingSyscallConn(conn)
	return !ok
}
