// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::fmt;
use std::marker::PhantomData;

use serde::de::{IgnoredAny, SeqAccess, Visitor, value::SeqAccessDeserializer};
use serde::{Deserialize, Deserializer};
use serde_with::DeserializeAs;

/// Decode an array-like struct using its derived field types and defaults,
/// skipping appended fields as Python's msgspec does. Opt in only for schemas
/// whose extensions preserve the known field order and semantics.
pub(super) struct AllowTrailingFields;

impl<'de, T: Deserialize<'de>> DeserializeAs<'de, T> for AllowTrailingFields {
    fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ArrayVisitor<T>(PhantomData<T>);

        impl<'de, T: Deserialize<'de>> Visitor<'de> for ArrayVisitor<T> {
            type Value = T;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("an array-like struct")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<T, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let value = T::deserialize(SeqAccessDeserializer::new(&mut seq))?;
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(value)
            }
        }

        deserializer.deserialize_seq(ArrayVisitor(PhantomData))
    }
}
