# Changelog

## [1.5.0] - 2023-08-15

### Fixed

- Use io.Reader instead of custom Reader

## [1.5.0] - 2023-08-15

### Added

- Support for reading UTF16 files

### Fixed

- Scanner error handling
- Reader error handling

## [1.4.2] - 2023-01-11

### Fixed

- Env var initialization

### Changed

- More consitent line splitting

## [1.4.1] - 2022-08-23

### Fixed

- Missing file close

### Changed

- Updated dependencies

## [1.4.0] - 2022-06-02

### Added

- Add `Marshal` and `Unmarshal` helpers

### Changed

- The CI will now run a linter and the tests on PRs.

## [1.3.0] - 2022-05-23

### Added

- Support = within double-quoted strings
- Add support for multiline values

### Changed

- `OverLoad` prefer environment variables over local variables

## [1.2.0] - 2019-08-03

### Added

- Add `Must` helper to raise an error as panic. It can be used with `Load` and `OverLoad`.
- Add more tests to be 100% coverage.
- Add CHANGELOG
- Add more OS for the test: OSX and Windows

### Changed

- Reduce complexity and improve source code for having `A+` score in [goreportcard](https://goreportcard.com/report/github.com/subosito/gotenv).
- Updated README with mentions to all available functions

### Removed

- Remove `ErrFormat`
- Remove `MustLoad` and `MustOverload`, replaced with `Must` helper.

## [1.1.1] - 2018-06-05

### Changed

- Replace `os.Getenv` with `os.LookupEnv` to ensure that the environment variable is not set, by [radding](https://github.com/radding)

## [1.1.0] - 2017-03-20

### Added

- Supports carriage return in env
- Handle files with UTF-8 BOM

### Changed

- Whitespace handling

### Fixed

- Incorrect variable expansion
- Handling escaped '$' characters

## [1.0.0] - 2014-10-05

First stable release.

