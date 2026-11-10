// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Recursion-depth tracking for synchronous parsers.

use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

use winnow::error::{ContextError, ErrMode, ModalResult, StrContext};

/// Maximum number of recursive parser frames active on one thread.
pub(crate) const MAX_PARSER_RECURSION_DEPTH: usize = 128;

thread_local! {
    static PARSER_RECURSION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Bounds recursive parser calls on the current thread.
///
/// Keep the guard alive for the duration of one recursive parser frame. Its
/// thread-local depth is restored on return, parse errors, and unwinding.
pub(crate) struct ParserRecursionGuard {
    previous_depth: usize,
    // Dropping on another thread would restore the wrong TLS counter.
    _not_send: PhantomData<Rc<()>>,
}

impl ParserRecursionGuard {
    /// Enters one recursive parser frame.
    ///
    /// Returns a cut error when the current thread has reached the recursion
    /// limit. The returned guard must stay alive for the full recursive call;
    /// dropping it restores the preceding depth.
    pub(crate) fn enter() -> ModalResult<Self> {
        PARSER_RECURSION_DEPTH.with(|depth| {
            let previous_depth = depth.get();
            if previous_depth >= MAX_PARSER_RECURSION_DEPTH {
                let mut error = ContextError::new();
                error.push(StrContext::Label("parser recursion limit exceeded"));
                return Err(ErrMode::Cut(error));
            }

            depth.set(previous_depth + 1);
            Ok(Self {
                previous_depth,
                _not_send: PhantomData,
            })
        })
    }
}

impl Drop for ParserRecursionGuard {
    fn drop(&mut self) {
        PARSER_RECURSION_DEPTH.with(|depth| depth.set(self.previous_depth));
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use winnow::error::ErrMode;

    use super::{MAX_PARSER_RECURSION_DEPTH, ParserRecursionGuard};

    fn enter_recursively(depth: usize) -> winnow::error::ModalResult<()> {
        if depth == 0 {
            return Ok(());
        }

        let _guard = ParserRecursionGuard::enter()?;
        enter_recursively(depth - 1)
    }

    #[test]
    fn enforces_limit_and_restores_depth() {
        assert!(enter_recursively(MAX_PARSER_RECURSION_DEPTH).is_ok());
        assert!(matches!(
            enter_recursively(MAX_PARSER_RECURSION_DEPTH + 1),
            Err(ErrMode::Cut(_))
        ));
        assert!(enter_recursively(MAX_PARSER_RECURSION_DEPTH).is_ok());
    }

    #[test]
    fn restores_depth_after_unwind() {
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _guard = ParserRecursionGuard::enter().unwrap();
            panic!("test unwind");
        }));

        assert!(result.is_err());
        assert!(enter_recursively(MAX_PARSER_RECURSION_DEPTH).is_ok());
    }
}
