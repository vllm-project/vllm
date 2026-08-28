package unstable

// root contains a full AST.
//
// It is immutable once constructed with Builder.
type root struct {
	nodes []Node
}

// Iterator over the top level nodes.
func (r *root) Iterator() Iterator {
	it := Iterator{}
	if len(r.nodes) > 0 {
		it.node = &r.nodes[0]
	}
	return it
}

func (r *root) at(idx reference) *Node {
	return &r.nodes[idx]
}

type reference int

const invalidReference reference = -1

func (r reference) Valid() bool {
	return r != invalidReference
}

type builder struct {
	tree    root
	lastIdx int
}

func (b *builder) Tree() *root {
	return &b.tree
}

func (b *builder) NodeAt(ref reference) *Node {
	return b.tree.at(ref)
}

func (b *builder) Reset() {
	b.tree.nodes = b.tree.nodes[:0]
	b.lastIdx = 0
}

func (b *builder) Push(n Node) reference {
	b.lastIdx = len(b.tree.nodes)
	b.tree.nodes = append(b.tree.nodes, n)
	return reference(b.lastIdx)
}

func (b *builder) PushAndChain(n Node) reference {
	newIdx := len(b.tree.nodes)
	b.tree.nodes = append(b.tree.nodes, n)
	if b.lastIdx >= 0 {
		b.tree.nodes[b.lastIdx].next = newIdx - b.lastIdx
	}
	b.lastIdx = newIdx
	return reference(b.lastIdx)
}

func (b *builder) AttachChild(parent reference, child reference) {
	b.tree.nodes[parent].child = int(child) - int(parent)
}

func (b *builder) Chain(from reference, to reference) {
	b.tree.nodes[from].next = int(to) - int(from)
}
