/// Shared streaming state for parsers that emit one active tool call at a time.
///
/// This intentionally captures only the stable bookkeeping shared by several
/// parser families:
/// - which tool call is currently active
/// - whether that tool's name has already been emitted
/// - which argument bytes have already been streamed for that tool
///
/// The parser still owns format-specific concerns such as its text buffer,
/// token/tag detection, and syntax parsing.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct StreamingToolState {
    next_tool_index: usize,
    active_tool: Option<ActiveToolState>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct ActiveToolState {
    tool_index: usize,
    name_sent: bool,
    streamed_arguments: String,
}

impl StreamingToolState {
    /// Start tracking a new active tool call and return its stable stream index.
    pub(crate) fn begin_tool_call(&mut self) -> usize {
        let tool_index = self.next_tool_index;
        self.next_tool_index += 1;
        self.active_tool = Some(ActiveToolState {
            tool_index,
            ..ActiveToolState::default()
        });
        tool_index
    }

    /// Return the current active tool index, if any.
    pub(crate) fn active_tool_index(&self) -> Option<usize> {
        self.active_tool.as_ref().map(|tool| tool.tool_index)
    }

    /// Return whether the active tool's name has been emitted already.
    pub(crate) fn active_tool_name_sent(&self) -> bool {
        self.active_tool.as_ref().is_some_and(|tool| tool.name_sent)
    }

    /// Mark the active tool name as emitted.
    pub(crate) fn mark_active_tool_name_sent(&mut self) {
        if let Some(tool) = self.active_tool.as_mut() {
            tool.name_sent = true;
        }
    }

    /// Return the streamed argument prefix for the active tool.
    pub(crate) fn active_streamed_arguments(&self) -> Option<&str> {
        self.active_tool
            .as_ref()
            .map(|tool| tool.streamed_arguments.as_str())
    }

    /// Replace the streamed argument prefix tracked for the active tool.
    pub(crate) fn set_active_streamed_arguments(&mut self, arguments: impl Into<String>) {
        if let Some(tool) = self.active_tool.as_mut() {
            tool.streamed_arguments = arguments.into();
        }
    }

    /// Clear state specific to the currently active tool.
    pub(crate) fn clear_active_tool(&mut self) {
        self.active_tool = None;
    }

    /// Reset the whole streaming state back to the beginning of a turn.
    pub(crate) fn reset(&mut self) {
        self.next_tool_index = 0;
        self.active_tool = None;
    }
}

#[cfg(test)]
mod tests {
    use super::StreamingToolState;

    #[test]
    fn streaming_tool_state_tracks_active_tool_lifecycle() {
        let mut state = StreamingToolState::default();

        let first = state.begin_tool_call();
        assert_eq!(first, 0);
        assert_eq!(state.active_tool_index(), Some(0));
        assert!(!state.active_tool_name_sent());
        assert_eq!(state.active_streamed_arguments(), Some(""));

        state.mark_active_tool_name_sent();
        state.set_active_streamed_arguments("{\"x\"");
        assert!(state.active_tool_name_sent());
        assert_eq!(state.active_streamed_arguments(), Some("{\"x\""));

        state.clear_active_tool();
        assert_eq!(state.active_tool_index(), None);
        assert!(!state.active_tool_name_sent());
        assert_eq!(state.active_streamed_arguments(), None);

        let second = state.begin_tool_call();
        assert_eq!(second, 1);

        state.reset();
        assert_eq!(state.active_tool_index(), None);
        let restarted = state.begin_tool_call();
        assert_eq!(restarted, 0);
    }
}
