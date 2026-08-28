package viper

import (
	"errors"

	"github.com/spf13/afero"
)

// WithFinder sets a custom [Finder].
func WithFinder(f Finder) Option {
	return optionFunc(func(v *Viper) {
		if f == nil {
			return
		}

		v.finder = f
	})
}

// Finder looks for files and directories in an [afero.Fs] filesystem.
type Finder interface {
	Find(fsys afero.Fs) ([]string, error)
}

// Finders combines multiple finders into one.
func Finders(finders ...Finder) Finder {
	return &combinedFinder{finders: finders}
}

// combinedFinder is a Finder that combines multiple finders.
type combinedFinder struct {
	finders []Finder
}

// Find implements the [Finder] interface.
func (c *combinedFinder) Find(fsys afero.Fs) ([]string, error) {
	var results []string
	var errs []error

	for _, finder := range c.finders {
		if finder == nil {
			continue
		}

		r, err := finder.Find(fsys)
		if err != nil {
			errs = append(errs, err)
			continue
		}

		results = append(results, r...)
	}

	return results, errors.Join(errs...)
}
