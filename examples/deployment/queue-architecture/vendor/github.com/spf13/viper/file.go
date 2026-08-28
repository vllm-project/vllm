package viper

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/sagikazarmark/locafero"
	"github.com/spf13/afero"
)

// ExperimentalFinder tells Viper to use the new Finder interface for finding configuration files.
func ExperimentalFinder() Option {
	return optionFunc(func(v *Viper) {
		v.experimentalFinder = true
	})
}

// Search for a config file.
func (v *Viper) findConfigFile() (string, error) {
	finder := v.finder

	if finder == nil && v.experimentalFinder {
		var names []string

		if v.configType != "" {
			names = locafero.NameWithOptionalExtensions(v.configName, SupportedExts...)
		} else {
			names = locafero.NameWithExtensions(v.configName, SupportedExts...)
		}

		finder = locafero.Finder{
			Paths: v.configPaths,
			Names: names,
			Type:  locafero.FileTypeFile,
		}
	}

	if finder != nil {
		return v.findConfigFileWithFinder(finder)
	}

	return v.findConfigFileOld()
}

func (v *Viper) findConfigFileWithFinder(finder Finder) (string, error) {
	results, err := finder.Find(v.fs)
	if err != nil {
		return "", err
	}

	if len(results) == 0 {
		return "", ConfigFileNotFoundError{v.configName, fmt.Sprintf("%s", v.configPaths)}
	}

	// We call clean on the final result to ensure that the path is in its canonical form.
	// This is mostly for consistent path handling and to make sure tests pass.
	return results[0], nil
}

// Search all configPaths for any config file.
// Returns the first path that exists (and is a config file).
func (v *Viper) findConfigFileOld() (string, error) {
	v.logger.Info("searching for config in paths", "paths", v.configPaths)

	for _, cp := range v.configPaths {
		file := v.searchInPath(cp)
		if file != "" {
			return file, nil
		}
	}
	return "", ConfigFileNotFoundError{v.configName, fmt.Sprintf("%s", v.configPaths)}
}

func (v *Viper) searchInPath(in string) (filename string) {
	v.logger.Debug("searching for config in path", "path", in)
	for _, ext := range SupportedExts {
		v.logger.Debug("checking if file exists", "file", filepath.Join(in, v.configName+"."+ext))
		if b, _ := exists(v.fs, filepath.Join(in, v.configName+"."+ext)); b {
			v.logger.Debug("found file", "file", filepath.Join(in, v.configName+"."+ext))
			return filepath.Join(in, v.configName+"."+ext)
		}
	}

	if v.configType != "" {
		if b, _ := exists(v.fs, filepath.Join(in, v.configName)); b {
			return filepath.Join(in, v.configName)
		}
	}

	return ""
}

// exists checks if file exists.
func exists(fs afero.Fs, path string) (bool, error) {
	stat, err := fs.Stat(path)
	if err == nil {
		return !stat.IsDir(), nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}
