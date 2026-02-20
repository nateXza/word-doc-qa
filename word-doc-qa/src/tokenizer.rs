// tokenizer.rs — Simple word-level tokenizer with a fixed vocabulary.
//
// We build the vocabulary from the corpus during data loading and then
// convert strings to integer token-id sequences for the model.
//
// Special tokens:
//   0  [PAD]   — padding token used to align batches
//   1  [UNK]   — unknown word not present in vocabulary
//   2  [SEP]   — separator between question and context
//   3  [CLS]   — sequence start / classification token

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

pub const PAD_ID: u32 = 0;
pub const UNK_ID: u32 = 1;
pub const SEP_ID: u32 = 2;
pub const CLS_ID: u32 = 3;
const RESERVED: u32 = 4; // first non-special id

/// Vocabulary built from the document corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    pub word2id: HashMap<String, u32>,
    pub id2word: HashMap<u32, String>,
    pub size: usize,
}

impl Vocab {
    /// Construct vocabulary from raw text, capped at `max_size` tokens.
    pub fn build(texts: &[String], max_size: usize) -> Self {
        let mut freq: HashMap<String, usize> = HashMap::new();

        for text in texts {
            for token in tokenize_str(text) {
                *freq.entry(token).or_insert(0) += 1;
            }
        }

        // Sort by descending frequency, then alphabetically for determinism.
        let mut pairs: Vec<(String, usize)> = freq.into_iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

        let mut word2id: HashMap<String, u32> = HashMap::new();
        let mut id2word: HashMap<u32, String> = HashMap::new();

        // Insert special tokens at fixed IDs.
        for (id, name) in [(PAD_ID, "[PAD]"), (UNK_ID, "[UNK]"), (SEP_ID, "[SEP]"), (CLS_ID, "[CLS]")] {
            word2id.insert(name.to_owned(), id);
            id2word.insert(id, name.to_owned());
        }

        // Fill regular tokens up to max_size.
        let capacity = (max_size as u32).saturating_sub(RESERVED);
        for (i, (word, _)) in pairs.iter().take(capacity as usize).enumerate() {
            let id = RESERVED + i as u32;
            word2id.insert(word.clone(), id);
            id2word.insert(id, word.clone());
        }

        let size = word2id.len();
        Self { word2id, id2word, size }
    }

    /// Convert a string to a token-id sequence.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        tokenize_str(text)
            .into_iter()
            .map(|t| *self.word2id.get(&t).unwrap_or(&UNK_ID))
            .collect()
    }

    /// Encode question + context as a single sequence:
    ///   [CLS] <question tokens> [SEP] <context tokens> [PAD…]
    ///
    /// Returns the padded/truncated token-id vector and the offset at which the
    /// context starts (needed to map answer spans back to text).
    pub fn encode_pair(&self, question: &str, context: &str, max_len: usize) -> (Vec<u32>, usize) {
        let q_ids = self.encode(question);
        let c_ids = self.encode(context);

        // [CLS] q [SEP] c — then truncate/pad to max_len.
        let mut ids = Vec::with_capacity(max_len);
        ids.push(CLS_ID);
        ids.extend_from_slice(&q_ids[..q_ids.len().min(max_len.saturating_sub(2))]);
        ids.push(SEP_ID);

        let context_start = ids.len(); // offset into ids where context begins

        let remaining = max_len.saturating_sub(ids.len());
        ids.extend_from_slice(&c_ids[..c_ids.len().min(remaining)]);

        // Pad to max_len.
        ids.resize(max_len, PAD_ID);

        (ids, context_start)
    }

    /// Decode a slice of token ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter(|&&id| id != PAD_ID && id != CLS_ID && id != SEP_ID)
            .map(|id| self.id2word.get(id).map(|s| s.as_str()).unwrap_or("[UNK]"))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Lowercases text and splits on whitespace/punctuation, returning word tokens.
pub fn tokenize_str(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| c.is_whitespace() || "(),:;?!\"'".contains(c))
        .filter(|s| !s.is_empty())
        .map(|s| s.to_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocab_roundtrip() {
        let texts = vec!["hello world test".to_owned(), "hello again".to_owned()];
        let vocab = Vocab::build(&texts, 100);

        let ids = vocab.encode("hello world");
        assert_eq!(ids.len(), 2);
        assert_ne!(ids[0], UNK_ID);

        // Unknown word maps to UNK.
        let unk_ids = vocab.encode("xyzzy");
        assert_eq!(unk_ids[0], UNK_ID);
    }

    #[test]
    fn test_encode_pair_length() {
        let texts = vec!["quick brown fox".to_owned()];
        let vocab = Vocab::build(&texts, 100);
        let (ids, _) = vocab.encode_pair("quick", "brown fox", 10);
        assert_eq!(ids.len(), 10);
    }
}
