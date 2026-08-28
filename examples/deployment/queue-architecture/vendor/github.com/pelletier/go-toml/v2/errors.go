package toml

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/pelletier/go-toml/v2/internal/danger"
	"github.com/pelletier/go-toml/v2/unstable"
)

// DecodeError represents an error encountered during the parsing or decoding
// of a TOML document.
//
// In addition to the error message, it contains the position in the document
// where it happened, as well as a human-readable representation that shows
// where the error occurred in the document.
type DecodeError struct {
	message string
	line    int
	column  int
	key     Key

	human string
}

// StrictMissingError occurs in a TOML document that does not have a
// corresponding field in the target value. It contains all the missing fields
// in Errors.
//
// Emitted by Decoder when DisallowUnknownFields() was called.
type StrictMissingError struct {
	// One error per field that could not be found.
	Errors []DecodeError
}

// Error returns the canonical string for this error.
func (s *StrictMissingError) Error() string {
	return "strict mode: fields in the document are missing in the target struct"
}

// String returns a human readable description of all errors.
func (s *StrictMissingError) String() string {
	var buf strings.Builder

	for i, e := range s.Errors {
		if i > 0 {
			buf.WriteString("\n---\n")
		}

		buf.WriteString(e.String())
	}

	return buf.String()
}

type Key []string

// Error returns the error message contained in the DecodeError.
func (e *DecodeError) Error() string {
	return "toml: " + e.message
}

// String returns the human-readable contextualized error. This string is multi-line.
func (e *DecodeError) String() string {
	return e.human
}

// Position returns the (line, column) pair indicating where the error
// occurred in the document. Positions are 1-indexed.
func (e *DecodeError) Position() (row int, column int) {
	return e.line, e.column
}

// Key that was being processed when the error occurred. The key is present only
// if this DecodeError is part of a StrictMissingError.
func (e *DecodeError) Key() Key {
	return e.key
}

// decodeErrorFromHighlight creates a DecodeError referencing a highlighted
// range of bytes from document.
//
// highlight needs to be a sub-slice of document, or this function panics.
//
// The function copies all bytes used in DecodeError, so that document and
// highlight can be freely deallocated.
//
//nolint:funlen
func wrapDecodeError(document []byte, de *unstable.ParserError) *DecodeError {
	offset := danger.SubsliceOffset(document, de.Highlight)

	errMessage := de.Error()
	errLine, errColumn := positionAtEnd(document[:offset])
	before, after := linesOfContext(document, de.Highlight, offset, 3)

	var buf strings.Builder

	maxLine := errLine + len(after) - 1
	lineColumnWidth := len(strconv.Itoa(maxLine))

	// Write the lines of context strictly before the error.
	for i := len(before) - 1; i > 0; i-- {
		line := errLine - i
		buf.WriteString(formatLineNumber(line, lineColumnWidth))
		buf.WriteString("|")

		if len(before[i]) > 0 {
			buf.WriteString(" ")
			buf.Write(before[i])
		}

		buf.WriteRune('\n')
	}

	// Write the document line that contains the error.

	buf.WriteString(formatLineNumber(errLine, lineColumnWidth))
	buf.WriteString("| ")

	if len(before) > 0 {
		buf.Write(before[0])
	}

	buf.Write(de.Highlight)

	if len(after) > 0 {
		buf.Write(after[0])
	}

	buf.WriteRune('\n')

	// Write the line with the error message itself (so it does not have a line
	// number).

	buf.WriteString(strings.Repeat(" ", lineColumnWidth))
	buf.WriteString("| ")

	if len(before) > 0 {
		buf.WriteString(strings.Repeat(" ", len(before[0])))
	}

	buf.WriteString(strings.Repeat("~", len(de.Highlight)))

	if len(errMessage) > 0 {
		buf.WriteString(" ")
		buf.WriteString(errMessage)
	}

	// Write the lines of context strictly after the error.

	for i := 1; i < len(after); i++ {
		buf.WriteRune('\n')
		line := errLine + i
		buf.WriteString(formatLineNumber(line, lineColumnWidth))
		buf.WriteString("|")

		if len(after[i]) > 0 {
			buf.WriteString(" ")
			buf.Write(after[i])
		}
	}

	return &DecodeError{
		message: errMessage,
		line:    errLine,
		column:  errColumn,
		key:     de.Key,
		human:   buf.String(),
	}
}

func formatLineNumber(line int, width int) string {
	format := "%" + strconv.Itoa(width) + "d"

	return fmt.Sprintf(format, line)
}

func linesOfContext(document []byte, highlight []byte, offset int, linesAround int) ([][]byte, [][]byte) {
	return beforeLines(document, offset, linesAround), afterLines(document, highlight, offset, linesAround)
}

func beforeLines(document []byte, offset int, linesAround int) [][]byte {
	var beforeLines [][]byte

	// Walk the document backward from the highlight to find previous lines
	// of context.
	rest := document[:offset]
backward:
	for o := len(rest) - 1; o >= 0 && len(beforeLines) <= linesAround && len(rest) > 0; {
		switch {
		case rest[o] == '\n':
			// handle individual lines
			beforeLines = append(beforeLines, rest[o+1:])
			rest = rest[:o]
			o = len(rest) - 1
		case o == 0:
			// add the first line only if it's non-empty
			beforeLines = append(beforeLines, rest)

			break backward
		default:
			o--
		}
	}

	return beforeLines
}

func afterLines(document []byte, highlight []byte, offset int, linesAround int) [][]byte {
	var afterLines [][]byte

	// Walk the document forward from the highlight to find the following
	// lines of context.
	rest := document[offset+len(highlight):]
forward:
	for o := 0; o < len(rest) && len(afterLines) <= linesAround; {
		switch {
		case rest[o] == '\n':
			// handle individual lines
			afterLines = append(afterLines, rest[:o])
			rest = rest[o+1:]
			o = 0

		case o == len(rest)-1:
			// add last line only if it's non-empty
			afterLines = append(afterLines, rest)

			break forward
		default:
			o++
		}
	}

	return afterLines
}

func positionAtEnd(b []byte) (row int, column int) {
	row = 1
	column = 1

	for _, c := range b {
		if c == '\n' {
			row++
			column = 1
		} else {
			column++
		}
	}

	return
}
