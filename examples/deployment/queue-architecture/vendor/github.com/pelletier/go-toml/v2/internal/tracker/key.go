package tracker

import "github.com/pelletier/go-toml/v2/unstable"

// KeyTracker is a tracker that keeps track of the current Key as the AST is
// walked.
type KeyTracker struct {
	k []string
}

// UpdateTable sets the state of the tracker with the AST table node.
func (t *KeyTracker) UpdateTable(node *unstable.Node) {
	t.reset()
	t.Push(node)
}

// UpdateArrayTable sets the state of the tracker with the AST array table node.
func (t *KeyTracker) UpdateArrayTable(node *unstable.Node) {
	t.reset()
	t.Push(node)
}

// Push the given key on the stack.
func (t *KeyTracker) Push(node *unstable.Node) {
	it := node.Key()
	for it.Next() {
		t.k = append(t.k, string(it.Node().Data))
	}
}

// Pop key from stack.
func (t *KeyTracker) Pop(node *unstable.Node) {
	it := node.Key()
	for it.Next() {
		t.k = t.k[:len(t.k)-1]
	}
}

// Key returns the current key
func (t *KeyTracker) Key() []string {
	k := make([]string, len(t.k))
	copy(k, t.k)
	return k
}

func (t *KeyTracker) reset() {
	t.k = t.k[:0]
}
