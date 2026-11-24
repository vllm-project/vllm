// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Language codes accepted by the speech-to-text APIs.

use tracing::warn;

use crate::error::ApiError;

/// ISO 639-1 codes and names of the languages OpenAI's speech-to-text
/// supports, from
/// <https://platform.openai.com/docs/guides/speech-to-text/supported-languages>.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/model_executor/models/whisper_utils.py#L6-L64>
pub(super) const ISO639_1_SUPPORTED_LANGS: &[(&str, &str)] = &[
    ("af", "Afrikaans"),
    ("ar", "Arabic"),
    ("hy", "Armenian"),
    ("az", "Azerbaijani"),
    ("be", "Belarusian"),
    ("bs", "Bosnian"),
    ("bg", "Bulgarian"),
    ("ca", "Catalan"),
    ("zh", "Chinese"),
    ("hr", "Croatian"),
    ("cs", "Czech"),
    ("da", "Danish"),
    ("nl", "Dutch"),
    ("en", "English"),
    ("et", "Estonian"),
    ("fi", "Finnish"),
    ("fr", "French"),
    ("gl", "Galician"),
    ("de", "German"),
    ("el", "Greek"),
    ("he", "Hebrew"),
    ("hi", "Hindi"),
    ("hu", "Hungarian"),
    ("is", "Icelandic"),
    ("id", "Indonesian"),
    ("it", "Italian"),
    ("ja", "Japanese"),
    ("kn", "Kannada"),
    ("kk", "Kazakh"),
    ("ko", "Korean"),
    ("lv", "Latvian"),
    ("lt", "Lithuanian"),
    ("mk", "Macedonian"),
    ("ms", "Malay"),
    ("mr", "Marathi"),
    ("mi", "Maori"),
    ("ne", "Nepali"),
    ("no", "Norwegian"),
    ("fa", "Persian"),
    ("pl", "Polish"),
    ("pt", "Portuguese"),
    ("ro", "Romanian"),
    ("ru", "Russian"),
    ("sr", "Serbian"),
    ("sk", "Slovak"),
    ("sl", "Slovenian"),
    ("es", "Spanish"),
    ("sw", "Swahili"),
    ("sv", "Swedish"),
    ("tl", "Tagalog"),
    ("ta", "Tamil"),
    ("th", "Thai"),
    ("tr", "Turkish"),
    ("uk", "Ukrainian"),
    ("ur", "Urdu"),
    ("vi", "Vietnamese"),
    ("cy", "Welsh"),
];

/// Every language code of Whisper's tokenizer, from `transformers`'
/// `models.whisper.tokenization_whisper.LANGUAGES`.
const WHISPER_LANGUAGE_CODES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue",
];

/// Ensure a requested language is a code the model supports natively, or at
/// least one Whisper knows, which passes with a warning.
///
/// Original Python definition:
/// <https://github.com/vllm-project/vllm/blob/dd8c157895/vllm/model_executor/models/interfaces.py#L1440-L1463>
pub(super) fn validate_language(
    language: &str,
    supported: &[(&str, &str)],
    model: &str,
) -> Result<(), ApiError> {
    if supported.iter().any(|&(code, _)| code == language) {
        return Ok(());
    }
    let supported_codes = supported.iter().map(|&(code, _)| code).collect::<Vec<_>>();
    if WHISPER_LANGUAGE_CODES.contains(&language) {
        warn!(
            language,
            model,
            ?supported_codes,
            "language is not natively supported by the model; results may be less accurate"
        );
        return Ok(());
    }
    Err(ApiError::invalid_request(
        format!(
            "Unsupported language: '{language}'.  Must be one of ['{}'].",
            supported_codes.join("', '")
        ),
        None,
    ))
}
