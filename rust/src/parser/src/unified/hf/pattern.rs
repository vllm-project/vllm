// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! PCRE2 matching for response-template boundaries, anchors, and content.
//!
//! Hard partial matching holds back a boundary while more input could change it.
//! Searches keep the entire subject and a start offset so lookbehind and anchors
//! see the same context regardless of chunk boundaries.

use std::ffi::CStr;
use std::ptr::{self, NonNull};
use std::sync::Arc;

use pcre2_sys::*;
use serde_json::{Map, Value};

use super::{Result, unsupported};
use crate::unified::{self, parsing_failed};

pub(super) type Captures = Map<String, Value>;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Resolution {
    Matched {
        start: usize,
        len: usize,
        captures: Captures,
    },
    Pending {
        start: usize,
    },
    NoMatch,
}

#[derive(Debug, Clone)]
pub(super) struct Pattern(Arc<Code>);

#[derive(Debug)]
struct Code {
    raw: NonNull<pcre2_code_8>,
    names: Vec<(String, usize)>,
    capture_count: u32,
    jit: bool,
}

// Compilation, including JIT, finishes before sharing; match state is per parser.
unsafe impl Send for Code {}
unsafe impl Sync for Code {}

impl Drop for Code {
    fn drop(&mut self) {
        // SAFETY: this allocation belongs to Code and is freed exactly once.
        unsafe { pcre2_code_free_8(self.raw.as_ptr()) }
    }
}

/// Scratch space owned by one parser, reused across its boundary searches.
#[derive(Default)]
pub(super) struct MatchState {
    data: Option<MatchData>,
    capture_count: u32,
}

impl MatchState {
    pub fn new(capture_count: u32) -> Self {
        Self {
            data: None,
            capture_count,
        }
    }
}

struct MatchData {
    data: NonNull<pcre2_match_data_8>,
    context: NonNull<pcre2_match_context_8>,
}

// Match data is exclusively accessed through its owning parser's mutable borrow.
unsafe impl Send for MatchData {}

impl Drop for MatchData {
    fn drop(&mut self) {
        // SAFETY: both allocations are exclusively owned and are no longer used.
        unsafe {
            pcre2_match_data_free_8(self.data.as_ptr());
            pcre2_match_context_free_8(self.context.as_ptr());
        }
    }
}

impl Pattern {
    /// Compile a UTF-8 pattern with DOTALL and Unicode character properties.
    pub fn new(scope: &str, pattern: &str) -> Result<Self> {
        let mut error = 0;
        let mut offset = 0;
        // SAFETY: pattern is live for its explicit length; PCRE2 copies it.
        let raw = unsafe {
            pcre2_compile_8(
                pattern.as_ptr(),
                pattern.len(),
                PCRE2_DOTALL | PCRE2_UTF | PCRE2_UCP | PCRE2_NEVER_BACKSLASH_C,
                &mut error,
                &mut offset,
                ptr::null_mut(),
            )
        };
        let raw = NonNull::new(raw).ok_or_else(|| {
            unsupported!(
                "{scope}: PCRE2 pattern {pattern:?} at byte {offset}: {}",
                error_message(error)
            )
        })?;
        let mut code = Code {
            raw,
            names: Vec::new(),
            capture_count: 0,
            jit: false,
        };
        let mut count: u32 = 0;
        let mut size: u32 = 0;
        let mut table: *const u8 = ptr::null();
        // SAFETY: the output types match these PCRE2_INFO requests, and the name
        // table stays valid for the lifetime of the compiled code.
        unsafe {
            pcre2_pattern_info_8(
                raw.as_ptr(),
                PCRE2_INFO_CAPTURECOUNT,
                (&mut code.capture_count as *mut u32).cast(),
            );
            pcre2_pattern_info_8(
                raw.as_ptr(),
                PCRE2_INFO_NAMECOUNT,
                (&mut count as *mut u32).cast(),
            );
            pcre2_pattern_info_8(
                raw.as_ptr(),
                PCRE2_INFO_NAMEENTRYSIZE,
                (&mut size as *mut u32).cast(),
            );
            pcre2_pattern_info_8(
                raw.as_ptr(),
                PCRE2_INFO_NAMETABLE,
                (&mut table as *mut *const u8).cast(),
            );
            for index in 0..count {
                let entry = table.add((index * size) as usize);
                let group = (usize::from(*entry) << 8) | usize::from(*entry.add(1));
                let name = CStr::from_ptr(entry.add(2).cast()).to_string_lossy().into_owned();
                code.names.push((name, group));
            }
        }
        // SAFETY: code is still exclusively owned. JIT is optional: unsupported
        // patterns or platforms keep using the interpreter.
        let mut jit_size: usize = 0;
        code.jit = unsafe {
            pcre2_jit_compile_8(raw.as_ptr(), PCRE2_JIT_COMPLETE | PCRE2_JIT_PARTIAL_HARD) == 0
                && pcre2_pattern_info_8(
                    raw.as_ptr(),
                    PCRE2_INFO_JITSIZE,
                    (&mut jit_size as *mut usize).cast(),
                ) == 0
                && jit_size > 0
        };
        Ok(Self(Arc::new(code)))
    }

    pub fn group_names(&self) -> impl Iterator<Item = &str> {
        self.0.names.iter().map(|(name, _)| name.as_str())
    }

    pub fn capture_count(&self) -> u32 {
        self.0.capture_count
    }

    /// Search from `offset`, retaining earlier text as assertion context.
    pub fn search(
        &self,
        text: &str,
        offset: usize,
        eof: bool,
        state: &mut MatchState,
    ) -> unified::Result<Resolution> {
        if !text.is_char_boundary(offset) {
            return Err(parsing_failed!(
                "PCRE2 search offset is not a UTF-8 boundary"
            ));
        }
        if state.data.is_none() || state.capture_count < self.0.capture_count {
            let capture_count = state.capture_count.max(self.0.capture_count);
            // SAFETY: null selects PCRE2's default allocator; include group zero.
            let data = NonNull::new(unsafe {
                pcre2_match_data_create_8(capture_count + 1, ptr::null_mut())
            })
            .ok_or_else(|| parsing_failed!("PCRE2 could not allocate match data"))?;
            // PCRE2 may cap the requested vector size; never expose more slots
            // than the allocation contains when reading captures below.
            if unsafe { pcre2_get_ovector_count_8(data.as_ptr()) } <= capture_count {
                // SAFETY: allocation has not yet been transferred into MatchData.
                unsafe { pcre2_match_data_free_8(data.as_ptr()) };
                return Err(parsing_failed!("PCRE2 capture vector is too small"));
            }
            if let Some(state) = &mut state.data {
                // SAFETY: replace the exclusively owned, undersized capture data.
                unsafe { pcre2_match_data_free_8(state.data.as_ptr()) };
                state.data = data;
            } else {
                // SAFETY: null selects the default allocator.
                let context =
                    NonNull::new(unsafe { pcre2_match_context_create_8(ptr::null_mut()) });
                let Some(context) = context else {
                    // SAFETY: allocation has not yet been transferred into MatchData.
                    unsafe { pcre2_match_data_free_8(data.as_ptr()) };
                    return Err(parsing_failed!("PCRE2 could not allocate match context"));
                };
                // SAFETY: context is live and exclusively owned.
                unsafe {
                    pcre2_set_match_limit_8(context.as_ptr(), 1_000_000);
                    pcre2_set_depth_limit_8(context.as_ptr(), 1_000);
                    pcre2_set_heap_limit_8(context.as_ptr(), 8 * 1024);
                }
                state.data = Some(MatchData { data, context });
            }
            state.capture_count = capture_count;
        }
        let state = state.data.as_mut().unwrap();
        // JIT's SIMD loads require a readable address even for length zero;
        // an empty Rust allocation may instead have a dangling pointer.
        let subject = if text.is_empty() {
            c"".as_ptr().cast()
        } else {
            text.as_ptr()
        };
        // SAFETY: pointers are live and uniquely owned by this search; subject
        // and offset are UTF-8, and match data was sized for this pattern. Both
        // matching modes were compiled when selecting the JIT entry point.
        let result = unsafe {
            let options = PCRE2_NO_UTF_CHECK | if eof { 0 } else { PCRE2_PARTIAL_HARD };
            let interpret = || {
                pcre2_match_8(
                    self.0.raw.as_ptr(),
                    subject,
                    text.len(),
                    offset,
                    options | PCRE2_NO_JIT,
                    state.data.as_ptr(),
                    state.context.as_ptr(),
                )
            };
            if self.0.jit {
                let result = pcre2_jit_match_8(
                    self.0.raw.as_ptr(),
                    subject,
                    text.len(),
                    offset,
                    options,
                    state.data.as_ptr(),
                    state.context.as_ptr(),
                );
                // JIT uses its bounded default stack. A pattern that needs more
                // can still run under the interpreter's resource limits.
                if result == PCRE2_ERROR_JIT_STACKLIMIT {
                    interpret()
                } else {
                    result
                }
            } else {
                interpret()
            }
        };
        if result == PCRE2_ERROR_NOMATCH {
            return Ok(Resolution::NoMatch);
        }
        if result < 0 && result != PCRE2_ERROR_PARTIAL {
            return Err(parsing_failed!(
                "PCRE2 matching failed: {}",
                error_message(result)
            ));
        }
        // SAFETY: a match or partial match initializes the first pair; complete
        // matches initialize every capture slot, using PCRE2_UNSET for absences.
        let spans = unsafe {
            std::slice::from_raw_parts(
                pcre2_get_ovector_pointer_8(state.data.as_ptr()),
                if result == PCRE2_ERROR_PARTIAL {
                    2
                } else {
                    2 * (self.0.capture_count as usize + 1)
                },
            )
        };
        if result == PCRE2_ERROR_PARTIAL {
            return Ok(Resolution::Pending {
                start: spans[0].max(offset),
            });
        }
        let mut captures = Captures::new();
        for (name, index) in &self.0.names {
            let start = spans[2 * index];
            if start != PCRE2_UNSET {
                let value = text
                    .get(start..spans[2 * index + 1])
                    .ok_or_else(|| parsing_failed!("PCRE2 capture is not a UTF-8 span"))?;
                captures.insert(name.clone(), Value::String(value.to_string()));
            }
        }
        let start = spans[0];
        let end = spans[1];
        if start < offset || start > end || text.get(start..end).is_none() {
            return Err(parsing_failed!("PCRE2 match is not a forward UTF-8 span"));
        }
        Ok(Resolution::Matched {
            start,
            len: end - start,
            captures,
        })
    }
}

fn error_message(code: i32) -> String {
    let mut buffer = [0; 256];
    // SAFETY: the output buffer is writable for the supplied length.
    let len = unsafe { pcre2_get_error_message_8(code, buffer.as_mut_ptr(), buffer.len()) };
    if len < 0 {
        return format!("error {code}");
    }
    String::from_utf8_lossy(&buffer[..len as usize]).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partial_matching_retains_lookaround_and_unicode_context() {
        let mut state = MatchState::default();
        let pattern = Pattern::new("test", r"(?<=天气)<call>(?P<name>\w+)(?=\{)").unwrap();
        assert!(matches!(
            pattern.search("天气<call>获取", 6, false, &mut state).unwrap(),
            Resolution::Pending { start: 6 }
        ));
        assert_eq!(
            pattern.search("天气<call>获取{", 6, false, &mut state).unwrap(),
            Resolution::Matched {
                start: 6,
                len: 12,
                captures: Captures::from_iter([("name".into(), "获取".into())])
            }
        );
        let negative = Pattern::new("test", r"<call>(?!bad)").unwrap();
        assert!(matches!(
            negative.search("<call>ba", 0, false, &mut state).unwrap(),
            Resolution::Pending { .. }
        ));
        assert_eq!(
            negative.search("<call>bad", 0, false, &mut state).unwrap(),
            Resolution::NoMatch
        );
    }

    #[test]
    fn greedy_tails_wait_and_matching_limits_are_errors() {
        let mut state = MatchState::default();
        let pattern = Pattern::new("test", r"<call>(?P<name>\w+)").unwrap();
        assert!(matches!(
            pattern.search("<call>get", 0, false, &mut state).unwrap(),
            Resolution::Pending { .. }
        ));
        assert!(matches!(
            pattern.search("<call>get", 0, true, &mut state).unwrap(),
            Resolution::Matched { len: 9, .. }
        ));
        let expensive = Pattern::new("test", r"(*NO_START_OPT)(*NO_AUTO_POSSESS)^(a+)+$").unwrap();
        assert!(expensive.search(&format!("{}!", "a".repeat(100)), 0, true, &mut state).is_err());
        let deep = Pattern::new("test", r"^(a|aa)+$").unwrap();
        let error = deep.search(&"a".repeat(10_000), 0, true, &mut state).unwrap_err();
        assert!(format!("{error}").contains("depth limit"));
    }

    #[test]
    fn reused_state_grows_captures_and_clears_absent_groups() {
        let mut state = MatchState::default();
        let literal = Pattern::new("test", "end").unwrap();
        let captures = Pattern::new("test", r"(?P<first>a)(?P<second>b)?").unwrap();
        literal.search("end", 0, true, &mut state).unwrap();
        assert_eq!(
            captures.search("ab", 0, true, &mut state).unwrap(),
            Resolution::Matched {
                start: 0,
                len: 2,
                captures: Captures::from_iter([
                    ("first".into(), "a".into()),
                    ("second".into(), "b".into()),
                ]),
            }
        );
        literal.search("end", 0, true, &mut state).unwrap();
        assert_eq!(
            captures.search("a", 0, true, &mut state).unwrap(),
            Resolution::Matched {
                start: 0,
                len: 1,
                captures: Captures::from_iter([("first".into(), "a".into())]),
            }
        );
    }

    #[test]
    fn empty_subject_is_safe_and_remains_empty() {
        let text = String::new();
        let mut state = MatchState::default();
        let literal = Pattern::new("test", "<call>").unwrap();
        for eof in [false, true] {
            assert_eq!(
                literal.search(&text, 0, eof, &mut state).unwrap(),
                Resolution::NoMatch
            );
        }
        let empty = Pattern::new("test", r"\A\z").unwrap();
        assert_eq!(
            empty.search(&text, 0, true, &mut state).unwrap(),
            Resolution::Matched {
                start: 0,
                len: 0,
                captures: Captures::new()
            }
        );
    }

    #[test]
    fn interpreter_fallback_preserves_partial_matches() {
        let pattern = Pattern::new("test", r"(*NO_JIT)(?<=天气)<call>(?P<name>\w+)(?=\{)").unwrap();
        assert!(!pattern.0.jit);
        let mut state = MatchState::default();
        assert_eq!(
            pattern.search("天气<call>获取", 6, false, &mut state).unwrap(),
            Resolution::Pending { start: 6 }
        );
        assert_eq!(
            pattern.search("天气<call>获取{", 6, false, &mut state).unwrap(),
            Resolution::Matched {
                start: 6,
                len: 12,
                captures: Captures::from_iter([("name".into(), "获取".into())]),
            }
        );
    }
}
