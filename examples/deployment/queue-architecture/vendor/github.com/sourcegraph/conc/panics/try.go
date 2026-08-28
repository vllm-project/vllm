package panics

// Try executes f, catching and returning any panic it might spawn.
//
// The recovered panic can be propagated with panic(), or handled as a normal error with
// (*panics.Recovered).AsError().
func Try(f func()) *Recovered {
	var c Catcher
	c.Try(f)
	return c.Recovered()
}
