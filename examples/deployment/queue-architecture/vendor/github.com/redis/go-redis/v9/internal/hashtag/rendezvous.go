package hashtag

import "github.com/cespare/xxhash/v2"

// RendezvousHash implements HRW (Highest Random Weight) hashing.
type RendezvousHash struct {
	nodes []node
}

type node struct {
	name string
	hash uint64
}

// NewRendezvousHash builds a hash from shard names.
func NewRendezvousHash(shards []string) *RendezvousHash {
	n := make([]node, len(shards))
	for i, s := range shards {
		n[i] = node{
			name: s,
			hash: xxhash.Sum64String(s),
		}
	}
	return &RendezvousHash{nodes: n}
}

// Get returns the shard name for the given key.
func (r *RendezvousHash) Get(key string) string {
	if len(r.nodes) == 0 {
		return ""
	}

	kh := xxhash.Sum64String(key)

	bestIdx := 0
	bestScore := mix64(kh ^ r.nodes[0].hash)

	for i := 1; i < len(r.nodes); i++ {
		if score := mix64(kh ^ r.nodes[i].hash); score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return r.nodes[bestIdx].name
}

// mix64 is a xorshift-based mixing function.
func mix64(x uint64) uint64 {
	x ^= x >> 12
	x ^= x << 25
	x ^= x >> 27
	return x * 2685821657736338717
}
