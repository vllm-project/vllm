// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_engine_core_client::protocol::tensor::WireNdArray;

use crate::error::Result;

/// Rows of one token-aligned auxiliary output (Python's AuxOutput, such as
/// routed experts), kept as the engine sent them: one chunk per engine output,
/// with one row per token along axis 0.
///
/// The engine sends each output's new rows, so a consumer appends the rows of
/// every output it keeps and concatenates them once, when the request
/// finishes. Joining at the end keeps the cost linear in the request length,
/// and keeping chunks lets text decoding drop the rows of tokens cut off by a
/// stop string.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AuxOutputRows {
    chunks: Vec<WireNdArray>,
}

impl AuxOutputRows {
    /// Wrap the rows of one engine output.
    pub(crate) fn from_engine(rows: Option<Box<WireNdArray>>) -> Self {
        Self {
            chunks: rows.map(|rows| *rows).into_iter().collect(),
        }
    }

    /// Whether no rows were received.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Append the rows of a later output.
    pub fn append(&mut self, mut later: Self) {
        self.chunks.append(&mut later.chunks);
    }

    /// Drop up to `num_rows` rows from the end, like Python's
    /// `rows[:-num_rows]` for the tokens a stop string cut off. A chunk keeps
    /// its row shape even when no row is left in it.
    pub fn drop_trailing(&mut self, mut num_rows: usize) -> Result<()> {
        for chunk in self.chunks.iter_mut().rev() {
            if num_rows == 0 {
                break;
            }
            let chunk_rows = chunk.shape.first().copied().unwrap_or(0);
            let dropped = num_rows.min(chunk_rows);
            chunk.truncate_rows(chunk_rows - dropped)?;
            num_rows -= dropped;
        }
        Ok(())
    }

    /// Concatenate the rows along the token axis, like Python's
    /// `np.concatenate`; `None` when no output carried rows.
    pub fn concat(self) -> Result<Option<WireNdArray>> {
        Ok(WireNdArray::concat_rows(self.chunks)?)
    }
}

impl From<WireNdArray> for AuxOutputRows {
    fn from(rows: WireNdArray) -> Self {
        Self { chunks: vec![rows] }
    }
}

#[cfg(test)]
mod tests {
    use vllm_engine_core_client::protocol::dtype::TensorDtype;
    use vllm_engine_core_client::protocol::tensor::ArrayRowsError;

    use super::*;
    use crate::error::Error;

    /// One `[1, 2]` row per value.
    fn rows(values: &[u8]) -> WireNdArray {
        WireNdArray::from_raw(
            TensorDtype::U8,
            vec![values.len(), 1, 2],
            values.iter().flat_map(|&value| [value, 0]).collect(),
        )
    }

    fn collected(rows: AuxOutputRows) -> Option<WireNdArray> {
        rows.concat().unwrap()
    }

    #[test]
    fn concat_joins_chunks_along_the_token_axis() {
        let mut all = AuxOutputRows::default();
        assert_eq!(collected(all.clone()), None);
        all.append(AuxOutputRows::from(rows(&[1, 2])));
        all.append(AuxOutputRows::default());
        all.append(AuxOutputRows::from(rows(&[3])));
        assert_eq!(collected(all), Some(rows(&[1, 2, 3])));
    }

    #[test]
    fn drop_trailing_cuts_rows_across_chunks() {
        let mut all = AuxOutputRows::from(rows(&[1, 2]));
        all.append(AuxOutputRows::from(rows(&[3, 4])));
        all.drop_trailing(3).unwrap();
        assert_eq!(collected(all.clone()), Some(rows(&[1])));
        // Like `rows[:-n]` with `n` past the start, nothing is left.
        all.drop_trailing(5).unwrap();
        assert_eq!(collected(all), Some(rows(&[])));
    }

    #[test]
    fn concat_rejects_a_changed_row_shape() {
        let mut all = AuxOutputRows::from(rows(&[1]));
        all.append(AuxOutputRows::from(WireNdArray::from_raw(
            TensorDtype::U8,
            vec![1, 2, 1],
            vec![0, 0],
        )));
        assert!(matches!(
            all.concat(),
            Err(Error::AuxOutputRows(ArrayRowsError::Mismatch { .. }))
        ));
    }
}
