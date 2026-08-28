package nkeys

import (
	"bytes"
	"regexp"
	"strings"
)

var userConfigRE = regexp.MustCompile(`\s*(?:(?:[-]{3,}.*[-]{3,}\r?\n)([\w\-.=]+)(?:\r?\n[-]{3,}.*[-]{3,}\r?\n))`)

// ParseDecoratedJWT takes a creds file and returns the JWT portion.
func ParseDecoratedJWT(contents []byte) (string, error) {
	items := userConfigRE.FindAllSubmatch(contents, -1)
	if len(items) == 0 {
		return string(contents), nil
	}
	// First result should be the user JWT.
	// We copy here so that if the file contained a seed file too we wipe appropriately.
	raw := items[0][1]
	tmp := make([]byte, len(raw))
	copy(tmp, raw)
	return strings.TrimSpace(string(tmp)), nil
}

// ParseDecoratedNKey takes a creds file, finds the NKey portion and creates a
// key pair from it.
func ParseDecoratedNKey(contents []byte) (KeyPair, error) {
	var seed []byte

	items := userConfigRE.FindAllSubmatch(contents, -1)
	if len(items) > 1 {
		seed = items[1][1]
	} else {
		lines := bytes.Split(contents, []byte("\n"))
		for _, line := range lines {
			if bytes.HasPrefix(bytes.TrimSpace(line), []byte("SO")) ||
				bytes.HasPrefix(bytes.TrimSpace(line), []byte("SA")) ||
				bytes.HasPrefix(bytes.TrimSpace(line), []byte("SU")) {
				seed = line
				break
			}
		}
	}
	if seed == nil {
		return nil, ErrNoSeedFound
	}
	if !bytes.HasPrefix(seed, []byte("SO")) &&
		!bytes.HasPrefix(seed, []byte("SA")) &&
		!bytes.HasPrefix(seed, []byte("SU")) {
		return nil, ErrInvalidNkeySeed
	}
	kp, err := FromSeed(seed)
	if err != nil {
		return nil, err
	}
	return kp, nil
}

// ParseDecoratedUserNKey takes a creds file, finds the NKey portion and creates a
// key pair from it. Similar to ParseDecoratedNKey but fails for non-user keys.
func ParseDecoratedUserNKey(contents []byte) (KeyPair, error) {
	nk, err := ParseDecoratedNKey(contents)
	if err != nil {
		return nil, err
	}
	seed, err := nk.Seed()
	if err != nil {
		return nil, err
	}
	if !bytes.HasPrefix(seed, []byte("SU")) {
		return nil, ErrInvalidUserSeed
	}
	kp, err := FromSeed(seed)
	if err != nil {
		return nil, err
	}
	return kp, nil
}
