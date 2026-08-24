//go:build !linux && !darwin && !dragonfly && !freebsd && !netbsd && !openbsd && !solaris && !illumos

package pool

import (
	"errors"
	"net"
)

// errUnexpectedRead is placeholder error variable for non-unix build constraints
var errUnexpectedRead = errors.New("unexpected read from socket")

func connCheck(_ net.Conn) error {
	return nil
}

// There is no portable non-consuming readiness check on this platform.
// Returning true would force every idle CSC connection through a timed read on
// every drainer tick. The CSC drainer uses needsCscPeriodicProbe instead.
func maybeHasData(_ net.Conn) bool {
	return false
}

func checkForData(_ net.Conn) (bool, error) {
	return false, nil
}

func needsCscReadProbe(_ net.Conn) bool {
	return true
}

func needsCscPeriodicProbe(_ net.Conn) bool {
	return true
}
