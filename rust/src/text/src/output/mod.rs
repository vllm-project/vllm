//! Output processing helpers shared by text and chat layers.

mod decoded;
mod incremental;

pub use decoded::{DecodedTextEvent, TextDecodeOptions, decoded_text_event_stream};
pub use incremental::IncrementalTextDecoder;
