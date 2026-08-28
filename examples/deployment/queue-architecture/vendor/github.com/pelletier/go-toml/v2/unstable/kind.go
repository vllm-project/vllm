package unstable

import "fmt"

// Kind represents the type of TOML structure contained in a given Node.
type Kind int

const (
	// Meta
	Invalid Kind = iota
	Comment
	Key

	// Top level structures
	Table
	ArrayTable
	KeyValue

	// Containers values
	Array
	InlineTable

	// Values
	String
	Bool
	Float
	Integer
	LocalDate
	LocalTime
	LocalDateTime
	DateTime
)

// String implementation of fmt.Stringer.
func (k Kind) String() string {
	switch k {
	case Invalid:
		return "Invalid"
	case Comment:
		return "Comment"
	case Key:
		return "Key"
	case Table:
		return "Table"
	case ArrayTable:
		return "ArrayTable"
	case KeyValue:
		return "KeyValue"
	case Array:
		return "Array"
	case InlineTable:
		return "InlineTable"
	case String:
		return "String"
	case Bool:
		return "Bool"
	case Float:
		return "Float"
	case Integer:
		return "Integer"
	case LocalDate:
		return "LocalDate"
	case LocalTime:
		return "LocalTime"
	case LocalDateTime:
		return "LocalDateTime"
	case DateTime:
		return "DateTime"
	}
	panic(fmt.Errorf("Kind.String() not implemented for '%d'", k))
}
