// Package locafero looks for files and directories in an {fs.Fs} filesystem.
package locafero

import (
	"errors"
	"io/fs"
	"path/filepath"
	"strings"

	"github.com/sourcegraph/conc/pool"
	"github.com/spf13/afero"
)

// Finder looks for files and directories in an [afero.Fs] filesystem.
type Finder struct {
	// Paths represents a list of locations that the [Finder] will search in.
	//
	// They are essentially the root directories or starting points for the search.
	//
	// Examples:
	//   - home/user
	//   - etc
	Paths []string

	// Names are specific entries that the [Finder] will look for within the given Paths.
	//
	// It provides the capability to search for entries with depth,
	// meaning it can target deeper locations within the directory structure.
	//
	// It also supports glob syntax (as defined by [filepath.Match]), offering greater flexibility in search patterns.
	//
	// Examples:
	//   - config.yaml
	//   - home/*/config.yaml
	//   - home/*/config.*
	Names []string

	// Type restricts the kind of entries returned by the [Finder].
	//
	// This parameter helps in differentiating and filtering out files from directories or vice versa.
	Type FileType
}

// Find looks for files and directories in an [afero.Fs] filesystem.
func (f Finder) Find(fsys afero.Fs) ([]string, error) {
	// Arbitrary go routine limit (TODO: make this a parameter)
	p := pool.NewWithResults[[]searchResult]().WithMaxGoroutines(5).WithErrors().WithFirstError()

	for _, searchPath := range f.Paths {
		for _, searchName := range f.Names {
			p.Go(func() ([]searchResult, error) {
				// If the name contains any glob character, perform a glob match
				if strings.ContainsAny(searchName, globMatch) {
					return globWalkSearch(fsys, searchPath, searchName, f.Type)
				}

				return statSearch(fsys, searchPath, searchName, f.Type)
			})
		}
	}

	searchResults, err := flatten(p.Wait())
	if err != nil {
		return nil, err
	}

	// Return early if no results were found
	if len(searchResults) == 0 {
		return nil, nil
	}

	results := make([]string, 0, len(searchResults))

	for _, searchResult := range searchResults {
		results = append(results, searchResult.path)
	}

	return results, nil
}

type searchResult struct {
	path string
	info fs.FileInfo
}

func flatten[T any](results [][]T, err error) ([]T, error) {
	if err != nil {
		return nil, err
	}

	var flattened []T

	for _, r := range results {
		flattened = append(flattened, r...)
	}

	return flattened, nil
}

func globWalkSearch(
	fsys afero.Fs,
	searchPath string,
	searchName string,
	searchType FileType,
) ([]searchResult, error) {
	var results []searchResult

	err := afero.Walk(fsys, searchPath, func(p string, fileInfo fs.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip the root path
		if p == searchPath {
			return nil
		}

		var result error

		// Stop reading subdirectories
		// TODO: add depth detection here
		if fileInfo.IsDir() && filepath.Dir(p) == searchPath {
			result = fs.SkipDir
		}

		// Skip unmatching type
		if !searchType.match(fileInfo) {
			return result
		}

		match, err := filepath.Match(searchName, fileInfo.Name())
		if err != nil {
			return err
		}

		if match {
			results = append(results, searchResult{p, fileInfo})
		}

		return result
	})
	if err != nil {
		return results, err
	}

	return results, nil
}

func statSearch(
	fsys afero.Fs,
	searchPath string,
	searchName string,
	searchType FileType,
) ([]searchResult, error) {
	filePath := filepath.Join(searchPath, searchName)

	fileInfo, err := fsys.Stat(filePath)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	// Skip unmatching type
	if !searchType.match(fileInfo) {
		return nil, nil
	}

	return []searchResult{{filePath, fileInfo}}, nil
}
