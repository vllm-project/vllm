package redis

import (
	"context"
)

// ScanIterator is used to incrementally iterate over a collection of elements.
type ScanIterator struct {
	cmd *ScanCmd
	pos int
}

// Err returns the last iterator error, if any.
func (it *ScanIterator) Err() error {
	return it.cmd.Err()
}

// Next advances the cursor and returns true if more values can be read.
func (it *ScanIterator) Next(ctx context.Context) bool {
	// Instantly return on errors.
	if it.cmd.Err() != nil {
		return false
	}

	// Advance cursor, check if we are still within range.
	if it.pos < len(it.cmd.page) {
		it.pos++
		return true
	}

	for {
		// Return if there is no more data to fetch.
		if it.cmd.cursor == 0 {
			return false
		}

		// Fetch next page.
		switch it.cmd.args[0] {
		case "scan", "qscan":
			it.cmd.args[1] = it.cmd.cursor
		default:
			it.cmd.args[2] = it.cmd.cursor
		}

		err := it.cmd.process(ctx, it.cmd)
		if err != nil {
			return false
		}
		// Await the fetch before reading page/cursor: on the deferred
		// autopipeline face process() only enqueues, and reading the raw
		// fields of a not-yet-executed command would spin re-issuing SCANs
		// with a stale cursor forever. Err() blocks until executed there and
		// is a no-op read everywhere else.
		if err := it.cmd.Err(); err != nil {
			return false
		}

		it.pos = 1

		// Redis can occasionally return empty page.
		if len(it.cmd.page) > 0 {
			return true
		}
	}
}

// Val returns the key/field at the current cursor position.
func (it *ScanIterator) Val() string {
	var v string
	if it.cmd.Err() == nil && it.pos > 0 && it.pos <= len(it.cmd.page) {
		v = it.cmd.page[it.pos-1]
	}
	return v
}
