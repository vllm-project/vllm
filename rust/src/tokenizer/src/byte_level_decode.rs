//! Fast GPT-2 byte-level detokenization that writes into a single `Vec<u8>`,
//! avoiding the `Vec<String>` / `String::join` assembly in fastokens' generic
//! `Decoder::decode` pipeline.

/// Reverse GPT-2 byte-to-unicode mapping: codepoint → original byte. The GPT-2
/// table only emits codepoints in U+0000..U+0143, so a flat array suffices.
const CHAR_TO_BYTE: [u8; 324] = build_char_to_byte();

const fn is_nice(b: u8) -> bool {
    (b >= b'!' && b <= b'~') || (b >= 0xA1 && b <= 0xAC) || b >= 0xAE
}

const fn build_char_to_byte() -> [u8; 324] {
    let mut table = [0u8; 324];
    let mut b: u16 = 0;
    while b < 256 {
        let cp = if is_nice(b as u8) {
            b as u32
        } else {
            256 + nice_offset(b as u8)
        };
        table[cp as usize] = b as u8;
        b += 1;
    }
    table
}

const fn nice_offset(b: u8) -> u32 {
    let mut i: u16 = 0;
    let mut n: u32 = 0;
    while i < b as u16 {
        if !is_nice(i as u8) {
            n += 1;
        }
        i += 1;
    }
    n
}

/// Decode byte-level encoded token strings into a single UTF-8 string,
/// matching `fastokens::decoders::ByteLevelDecoder`.
pub fn decode_byte_level<'a, I: IntoIterator<Item = &'a str>>(tokens: I) -> String {
    let iter = tokens.into_iter();
    let (lower, _) = iter.size_hint();
    let mut bytes: Vec<u8> = Vec::with_capacity(lower.saturating_mul(4));
    for token in iter {
        for c in token.chars() {
            let cp = c as usize;
            if cp < CHAR_TO_BYTE.len() {
                bytes.push(CHAR_TO_BYTE[cp]);
            } else {
                // Non-GPT2 codepoints (e.g. DeepSeek's U+FF5C, U+2581) pass through.
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            }
        }
    }
    String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_byte_to_char_ref() -> [char; 256] {
        let mut table = ['\0'; 256];
        let mut next: u32 = 256;
        for b in 0..=255u8 {
            let cp = if is_nice(b) {
                b as u32
            } else {
                let cp = next;
                next += 1;
                cp
            };
            table[b as usize] = char::from_u32(cp).unwrap();
        }
        table
    }

    #[test]
    fn char_to_byte_roundtrips_every_byte() {
        let byte_to_char = build_byte_to_char_ref();
        for b in 0..=255u8 {
            let cp = byte_to_char[b as usize] as usize;
            assert!(cp < CHAR_TO_BYTE.len());
            assert_eq!(CHAR_TO_BYTE[cp], b, "mismatch for byte {b:#x}");
        }
    }

    #[test]
    fn decode_ascii() {
        assert_eq!(decode_byte_level(["Hello"]), "Hello");
    }

    #[test]
    fn decode_space_marker() {
        // GPT-2 maps 0x20 → Ġ (U+0120).
        assert_eq!(
            decode_byte_level(["\u{120}Hello", "\u{120}world"]),
            " Hello world",
        );
    }

    #[test]
    fn decode_multibyte_euro() {
        // € → 0xE2 0x82 0xAC, each mapped to a specific GPT-2 char.
        let byte_to_char = build_byte_to_char_ref();
        let encoded: String =
            [0xE2u8, 0x82, 0xAC].iter().map(|&b| byte_to_char[b as usize]).collect();
        assert_eq!(decode_byte_level([encoded.as_str()]), "€");
    }

    #[test]
    fn decode_preserves_non_gpt2_chars() {
        let tok = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
        assert_eq!(decode_byte_level([tok]), "<｜begin▁of▁sentence｜>");
    }
}
