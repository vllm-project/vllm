package unstable

// The Unmarshaler interface may be implemented by types to customize their
// behavior when being unmarshaled from a TOML document.
type Unmarshaler interface {
	UnmarshalTOML(value *Node) error
}
