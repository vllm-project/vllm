package mapstructure

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"reflect"
	"strconv"
	"strings"
	"time"
)

// Error interface is implemented by all errors emitted by mapstructure.
//
// Use [errors.As] to check if an error implements this interface.
type Error interface {
	error

	mapstructure()
}

// DecodeError is a generic error type that holds information about
// a decoding error together with the name of the field that caused the error.
type DecodeError struct {
	name string
	err  error
}

func newDecodeError(name string, err error) *DecodeError {
	return &DecodeError{
		name: name,
		err:  err,
	}
}

func (e *DecodeError) Name() string {
	return e.name
}

func (e *DecodeError) Unwrap() error {
	return e.err
}

func (e *DecodeError) Error() string {
	return fmt.Sprintf("'%s' %s", e.name, e.err)
}

func (*DecodeError) mapstructure() {}

// ParseError is an error type that indicates a value could not be parsed
// into the expected type.
type ParseError struct {
	Expected reflect.Value
	Value    any
	Err      error
}

func (e *ParseError) Error() string {
	return fmt.Sprintf("cannot parse value as '%s': %s", e.Expected.Type(), e.Err)
}

func (*ParseError) mapstructure() {}

// UnconvertibleTypeError is an error type that indicates a value could not be
// converted to the expected type.
type UnconvertibleTypeError struct {
	Expected reflect.Value
	Value    any
}

func (e *UnconvertibleTypeError) Error() string {
	return fmt.Sprintf(
		"expected type '%s', got unconvertible type '%s'",
		e.Expected.Type(),
		reflect.TypeOf(e.Value),
	)
}

func (*UnconvertibleTypeError) mapstructure() {}

func wrapStrconvNumError(err error) error {
	if err == nil {
		return nil
	}

	if err, ok := err.(*strconv.NumError); ok {
		return &strconvNumError{Err: err}
	}

	return err
}

type strconvNumError struct {
	Err *strconv.NumError
}

func (e *strconvNumError) Error() string {
	return "strconv." + e.Err.Func + ": " + e.Err.Err.Error()
}

func (e *strconvNumError) Unwrap() error { return e.Err }

func wrapUrlError(err error) error {
	if err == nil {
		return nil
	}

	if err, ok := err.(*url.Error); ok {
		return &urlError{Err: err}
	}

	return err
}

type urlError struct {
	Err *url.Error
}

func (e *urlError) Error() string {
	return fmt.Sprintf("%s", e.Err.Err)
}

func (e *urlError) Unwrap() error { return e.Err }

func wrapNetParseError(err error) error {
	if err == nil {
		return nil
	}

	if err, ok := err.(*net.ParseError); ok {
		return &netParseError{Err: err}
	}

	return err
}

type netParseError struct {
	Err *net.ParseError
}

func (e *netParseError) Error() string {
	return "invalid " + e.Err.Type
}

func (e *netParseError) Unwrap() error { return e.Err }

func wrapTimeParseError(err error) error {
	if err == nil {
		return nil
	}

	if err, ok := err.(*time.ParseError); ok {
		return &timeParseError{Err: err}
	}

	return err
}

type timeParseError struct {
	Err *time.ParseError
}

func (e *timeParseError) Error() string {
	if e.Err.Message == "" {
		return fmt.Sprintf("parsing time as %q: cannot parse as %q", e.Err.Layout, e.Err.LayoutElem)
	}

	return "parsing time " + e.Err.Message
}

func (e *timeParseError) Unwrap() error { return e.Err }

func wrapNetIPParseAddrError(err error) error {
	if err == nil {
		return nil
	}

	if errMsg := err.Error(); strings.HasPrefix(errMsg, "ParseAddr") {
		errPieces := strings.Split(errMsg, ": ")

		return fmt.Errorf("ParseAddr: %s", errPieces[len(errPieces)-1])
	}

	return err
}

func wrapNetIPParseAddrPortError(err error) error {
	if err == nil {
		return nil
	}

	errMsg := err.Error()
	if strings.HasPrefix(errMsg, "invalid port ") {
		return errors.New("invalid port")
	} else if strings.HasPrefix(errMsg, "invalid ip:port ") {
		return errors.New("invalid ip:port")
	}

	return err
}

func wrapNetIPParsePrefixError(err error) error {
	if err == nil {
		return nil
	}

	if errMsg := err.Error(); strings.HasPrefix(errMsg, "netip.ParsePrefix") {
		errPieces := strings.Split(errMsg, ": ")

		return fmt.Errorf("netip.ParsePrefix: %s", errPieces[len(errPieces)-1])
	}

	return err
}

func wrapTimeParseDurationError(err error) error {
	if err == nil {
		return nil
	}

	errMsg := err.Error()
	if strings.HasPrefix(errMsg, "time: unknown unit ") {
		return errors.New("time: unknown unit")
	} else if strings.HasPrefix(errMsg, "time: ") {
		idx := strings.LastIndex(errMsg, " ")

		return errors.New(errMsg[:idx])
	}

	return err
}

func wrapTimeParseLocationError(err error) error {
	if err == nil {
		return nil
	}
	errMsg := err.Error()
	if strings.Contains(errMsg, "unknown time zone") || strings.HasPrefix(errMsg, "time: unknown format") {
		return fmt.Errorf("invalid time zone format: %w", err)
	}

	return err
}
