//go:build go1.20

package errors

import "errors"

func Join(errs ...error) error {
	return errors.Join(errs...)
}
