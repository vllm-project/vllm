package maintnotifications

// State represents the current state of a maintenance operation
type State int

const (
	// StateIdle indicates no upgrade is in progress
	StateIdle State = iota

	// StateHandoff indicates a connection handoff is in progress
	StateMoving
)

// String returns a string representation of the state.
func (s State) String() string {
	switch s {
	case StateIdle:
		return "idle"
	case StateMoving:
		return "moving"
	default:
		return "unknown"
	}
}
