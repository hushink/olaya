//! Hugging Face tokenizer adapter.

use std::path::Path;

use olaya_core::{DecisionError, TokenEncoder};

use crate::ModelError;

/// A `tokenizers::Tokenizer` that implements [`TokenEncoder`].
///
/// Encoding never adds special tokens; families add `[CLS]` / `[SEP]` (or the
/// model's own markers) explicitly.
#[derive(Clone)]
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
}

impl HfTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelError> {
        let mut inner = tokenizers::Tokenizer::from_file(path.as_ref())
            .map_err(|error| ModelError::Tokenizer(error.to_string()))?;
        // Some checkpoints bake truncation/padding into tokenizer.json (GLiClass
        // does); families apply their own layout-specific truncation.
        inner
            .with_truncation(None)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))?;
        inner.with_padding(None);
        Ok(Self { inner })
    }

    /// Load `<dir>/tokenizer.json`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, ModelError> {
        Self::from_file(dir.as_ref().join("tokenizer.json"))
    }
}

impl HfTokenizer {
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Unicode NFC, as upstream's `unicodedata.normalize("NFC", text)`.
    pub fn nfc(&self, text: &str) -> String {
        tokenizers::NormalizedString::from(text)
            .nfc()
            .get()
            .to_owned()
    }

    /// The character (code point) offset where each encoded token ends.
    pub fn char_ends(&self, text: &str) -> Result<Vec<usize>, ModelError> {
        let encoding = self
            .inner
            .encode_char_offsets(text, false)
            .map_err(|error| ModelError::Tokenizer(error.to_string()))?;
        Ok(encoding.get_offsets().iter().map(|(_, end)| *end).collect())
    }
}

impl TokenEncoder for HfTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, DecisionError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|error| DecisionError::tokenizer(error.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn token_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }
}
