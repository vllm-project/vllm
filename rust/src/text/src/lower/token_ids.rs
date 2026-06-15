use thiserror::Error;

#[derive(Debug, Error)]
#[error(
    "token_id(s) {token_ids:?} in {parameter} contain out-of-vocab token ids. \
     Vocabulary size: {vocab_size}"
)]
pub struct OutOfVocabError {
    pub parameter: &'static str,
    pub token_ids: Vec<u32>,
    pub vocab_size: usize,
}

pub(crate) fn validate_vocab_range(
    parameter: &'static str,
    token_ids: impl IntoIterator<Item = u32>,
    vocab_size: usize,
) -> std::result::Result<(), OutOfVocabError> {
    let invalid_token_ids: Vec<_> = token_ids
        .into_iter()
        .filter(|&token_id| token_id as usize >= vocab_size)
        .collect();
    if invalid_token_ids.is_empty() {
        return Ok(());
    }

    Err(OutOfVocabError {
        parameter,
        token_ids: invalid_token_ids,
        vocab_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_vocab_range_rejects_out_of_vocab_ids() {
        let error =
            validate_vocab_range("logprob_token_ids", [5_u32, 1000, 1001], 1000).unwrap_err();

        assert_eq!(error.parameter, "logprob_token_ids");
        assert_eq!(error.token_ids, vec![1000, 1001]);
        assert_eq!(error.vocab_size, 1000);
    }
}
