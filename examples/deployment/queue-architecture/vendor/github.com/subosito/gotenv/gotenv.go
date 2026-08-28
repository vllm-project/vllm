// Package gotenv provides functionality to dynamically load the environment variables
package gotenv

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/text/encoding/unicode"
	"golang.org/x/text/transform"
)

const (
	// Pattern for detecting valid line format
	linePattern = `\A\s*(?:export\s+)?([\w\.]+)(?:\s*=\s*|:\s+?)('(?:\'|[^'])*'|"(?:\"|[^"])*"|[^#\n]+)?\s*(?:\s*\#.*)?\z`

	// Pattern for detecting valid variable within a value
	variablePattern = `(\\)?(\$)(\{?([A-Z0-9_]+)?\}?)`
)

// Byte order mark character
var (
	bomUTF8    = []byte("\xEF\xBB\xBF")
	bomUTF16LE = []byte("\xFF\xFE")
	bomUTF16BE = []byte("\xFE\xFF")
)

// Env holds key/value pair of valid environment variable
type Env map[string]string

// Load is a function to load a file or multiple files and then export the valid variables into environment variables if they do not exist.
// When it's called with no argument, it will load `.env` file on the current path and set the environment variables.
// Otherwise, it will loop over the filenames parameter and set the proper environment variables.
func Load(filenames ...string) error {
	return loadenv(false, filenames...)
}

// OverLoad is a function to load a file or multiple files and then export and override the valid variables into environment variables.
func OverLoad(filenames ...string) error {
	return loadenv(true, filenames...)
}

// Must is wrapper function that will panic when supplied function returns an error.
func Must(fn func(filenames ...string) error, filenames ...string) {
	if err := fn(filenames...); err != nil {
		panic(err.Error())
	}
}

// Apply is a function to load an io Reader then export the valid variables into environment variables if they do not exist.
func Apply(r io.Reader) error {
	return parset(r, false)
}

// OverApply is a function to load an io Reader then export and override the valid variables into environment variables.
func OverApply(r io.Reader) error {
	return parset(r, true)
}

func loadenv(override bool, filenames ...string) error {
	if len(filenames) == 0 {
		filenames = []string{".env"}
	}

	for _, filename := range filenames {
		f, err := os.Open(filename)
		if err != nil {
			return err
		}

		err = parset(f, override)
		f.Close()
		if err != nil {
			return err
		}
	}

	return nil
}

// parse and set :)
func parset(r io.Reader, override bool) error {
	env, err := strictParse(r, override)
	if err != nil {
		return err
	}

	for key, val := range env {
		setenv(key, val, override)
	}

	return nil
}

func setenv(key, val string, override bool) {
	if override {
		os.Setenv(key, val)
	} else {
		if _, present := os.LookupEnv(key); !present {
			os.Setenv(key, val)
		}
	}
}

// Parse is a function to parse line by line any io.Reader supplied and returns the valid Env key/value pair of valid variables.
// It expands the value of a variable from the environment variable but does not set the value to the environment itself.
// This function is skipping any invalid lines and only processing the valid one.
func Parse(r io.Reader) Env {
	env, _ := strictParse(r, false)
	return env
}

// StrictParse is a function to parse line by line any io.Reader supplied and returns the valid Env key/value pair of valid variables.
// It expands the value of a variable from the environment variable but does not set the value to the environment itself.
// This function is returning an error if there are any invalid lines.
func StrictParse(r io.Reader) (Env, error) {
	return strictParse(r, false)
}

// Read is a function to parse a file line by line and returns the valid Env key/value pair of valid variables.
// It expands the value of a variable from the environment variable but does not set the value to the environment itself.
// This function is skipping any invalid lines and only processing the valid one.
func Read(filename string) (Env, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return strictParse(f, false)
}

// Unmarshal reads a string line by line and returns the valid Env key/value pair of valid variables.
// It expands the value of a variable from the environment variable but does not set the value to the environment itself.
// This function is returning an error if there are any invalid lines.
func Unmarshal(str string) (Env, error) {
	return strictParse(strings.NewReader(str), false)
}

// Marshal outputs the given environment as a env file.
// Variables will be sorted by name.
func Marshal(env Env) (string, error) {
	lines := make([]string, 0, len(env))
	for k, v := range env {
		if d, err := strconv.Atoi(v); err == nil {
			lines = append(lines, fmt.Sprintf(`%s=%d`, k, d))
		} else {
			lines = append(lines, fmt.Sprintf(`%s=%q`, k, v))
		}
	}
	sort.Strings(lines)
	return strings.Join(lines, "\n"), nil
}

// Write serializes the given environment and writes it to a file
func Write(env Env, filename string) error {
	content, err := Marshal(env)
	if err != nil {
		return err
	}
	// ensure the path exists
	if err := os.MkdirAll(filepath.Dir(filename), 0o775); err != nil {
		return err
	}
	// create or truncate the file
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	_, err = file.WriteString(content + "\n")
	if err != nil {
		return err
	}

	return file.Sync()
}

// splitLines is a valid SplitFunc for a bufio.Scanner. It will split lines on CR ('\r'), LF ('\n') or CRLF (any of the three sequences).
// If a CR is immediately followed by a LF, it is treated as a CRLF (one single line break).
func splitLines(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if atEOF && len(data) == 0 {
		return 0, nil, bufio.ErrFinalToken
	}

	idx := bytes.IndexAny(data, "\r\n")
	switch {
	case atEOF && idx < 0:
		return len(data), data, bufio.ErrFinalToken

	case idx < 0:
		return 0, nil, nil
	}

	// consume CR or LF
	eol := idx + 1
	// detect CRLF
	if len(data) > eol && data[eol-1] == '\r' && data[eol] == '\n' {
		eol++
	}

	return eol, data[:idx], nil
}

func strictParse(r io.Reader, override bool) (Env, error) {
	env := make(Env)

	buf := new(bytes.Buffer)
	tee := io.TeeReader(r, buf)

	// There can be a maximum of 3 BOM bytes.
	bomByteBuffer := make([]byte, 3)
	_, err := tee.Read(bomByteBuffer)
	if err != nil && err != io.EOF {
		return env, err
	}

	z := io.MultiReader(buf, r)

	// We chooes a different scanner depending on file encoding.
	var scanner *bufio.Scanner

	if bytes.HasPrefix(bomByteBuffer, bomUTF8) {
		scanner = bufio.NewScanner(transform.NewReader(z, unicode.UTF8BOM.NewDecoder()))
	} else if bytes.HasPrefix(bomByteBuffer, bomUTF16LE) {
		scanner = bufio.NewScanner(transform.NewReader(z, unicode.UTF16(unicode.LittleEndian, unicode.ExpectBOM).NewDecoder()))
	} else if bytes.HasPrefix(bomByteBuffer, bomUTF16BE) {
		scanner = bufio.NewScanner(transform.NewReader(z, unicode.UTF16(unicode.BigEndian, unicode.ExpectBOM).NewDecoder()))
	} else {
		scanner = bufio.NewScanner(z)
	}

	scanner.Split(splitLines)

	for scanner.Scan() {
		if err := scanner.Err(); err != nil {
			return env, err
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" || line[0] == '#' {
			continue
		}

		quote := ""
		// look for the delimiter character
		idx := strings.Index(line, "=")
		if idx == -1 {
			idx = strings.Index(line, ":")
		}
		// look for a quote character
		if idx > 0 && idx < len(line)-1 {
			val := strings.TrimSpace(line[idx+1:])
			if val[0] == '"' || val[0] == '\'' {
				quote = val[:1]
				// look for the closing quote character within the same line
				idx = strings.LastIndex(strings.TrimSpace(val[1:]), quote)
				if idx >= 0 && val[idx] != '\\' {
					quote = ""
				}
			}
		}
		// look for the closing quote character
		for quote != "" && scanner.Scan() {
			l := scanner.Text()
			line += "\n" + l
			idx := strings.LastIndex(l, quote)
			if idx > 0 && l[idx-1] == '\\' {
				// foud a matching quote character but it's escaped
				continue
			}
			if idx >= 0 {
				// foud a matching quote
				quote = ""
			}
		}

		if quote != "" {
			return env, fmt.Errorf("missing quotes")
		}

		err := parseLine(line, env, override)
		if err != nil {
			return env, err
		}
	}

	return env, scanner.Err()
}

var (
	lineRgx     = regexp.MustCompile(linePattern)
	unescapeRgx = regexp.MustCompile(`\\([^$])`)
	varRgx      = regexp.MustCompile(variablePattern)
)

func parseLine(s string, env Env, override bool) error {
	rm := lineRgx.FindStringSubmatch(s)

	if len(rm) == 0 {
		return checkFormat(s, env)
	}

	key := strings.TrimSpace(rm[1])
	val := strings.TrimSpace(rm[2])

	var hsq, hdq bool

	// check if the value is quoted
	if l := len(val); l >= 2 {
		l -= 1
		// has double quotes
		hdq = val[0] == '"' && val[l] == '"'
		// has single quotes
		hsq = val[0] == '\'' && val[l] == '\''

		// remove quotes '' or ""
		if hsq || hdq {
			val = val[1:l]
		}
	}

	if hdq {
		val = strings.ReplaceAll(val, `\n`, "\n")
		val = strings.ReplaceAll(val, `\r`, "\r")

		// Unescape all characters except $ so variables can be escaped properly
		val = unescapeRgx.ReplaceAllString(val, "$1")
	}

	if !hsq {
		fv := func(s string) string {
			return varReplacement(s, hsq, env, override)
		}
		val = varRgx.ReplaceAllStringFunc(val, fv)
	}

	env[key] = val
	return nil
}

func parseExport(st string, env Env) error {
	if strings.HasPrefix(st, "export") {
		vs := strings.SplitN(st, " ", 2)

		if len(vs) > 1 {
			if _, ok := env[vs[1]]; !ok {
				return fmt.Errorf("line `%s` has an unset variable", st)
			}
		}
	}

	return nil
}

var varNameRgx = regexp.MustCompile(`(\$)(\{?([A-Z0-9_]+)\}?)`)

func varReplacement(s string, hsq bool, env Env, override bool) string {
	if s == "" {
		return s
	}

	if s[0] == '\\' {
		// the dollar sign is escaped
		return s[1:]
	}

	if hsq {
		return s
	}

	mn := varNameRgx.FindStringSubmatch(s)

	if len(mn) == 0 {
		return s
	}

	v := mn[3]

	if replace, ok := os.LookupEnv(v); ok && !override {
		return replace
	}

	if replace, ok := env[v]; ok {
		return replace
	}

	return os.Getenv(v)
}

func checkFormat(s string, env Env) error {
	st := strings.TrimSpace(s)

	if st == "" || st[0] == '#' {
		return nil
	}

	if err := parseExport(st, env); err != nil {
		return err
	}

	return fmt.Errorf("line `%s` doesn't match format", s)
}
