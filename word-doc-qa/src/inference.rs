// inference.rs — Answer questions using either the trained neural model
//               or the deterministic retrieval engine (used as fallback and
//               for demonstrating correct answers while the model trains).
//
// Two-tier approach:
//   1. First attempt: BM25-style retrieval + answer extraction over the
//      embedded calendar knowledge base.  This gives correct answers
//      immediately and validates the data pipeline.
//   2. Second attempt: Load the trained Burn model from the latest checkpoint
//      and run the span-extraction forward pass.  Falls back gracefully if no
//      checkpoint is present (e.g., before training).

use std::collections::HashMap;
use std::fs;

use crate::data::get_embedded_calendar;
use crate::tokenizer::Vocab;

// ──────────────────────────────────────────────────────────────────────────────
// Public entry point
// ──────────────────────────────────────────────────────────────────────────────

/// Answer a natural language question about the CPUT calendar documents.
pub fn answer_question(question: &str) -> String {
    // Always try the neural model first if a checkpoint exists.
    if let Some(neural_answer) = try_neural_answer(question) {
        return neural_answer;
    }

    // Fall back to retrieval-based answer extraction.
    retrieval_answer(question)
}

// ──────────────────────────────────────────────────────────────────────────────
// Retrieval-based answer extraction
// ──────────────────────────────────────────────────────────────────────────────

/// Build the full knowledge base from all three calendar years.
fn knowledge_base() -> Vec<(String, String)> {
    // Each entry: (source_tag, text_chunk)
    let mut kb = Vec::new();

    for year in &["calader_2026.docx", "calendar_2025.docx", "calendar_2024.docx"] {
        let text = get_embedded_calendar(year);
        // Split into paragraph-sized chunks for retrieval.
        for chunk in text.split('\n') {
            let chunk = chunk.trim();
            if chunk.len() > 10 {
                kb.push((year.to_string(), chunk.to_owned()));
            }
        }
    }
    kb
}

/// Tokenise a string into lowercase words (for BM25 scoring).
fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty() && s.len() > 1)
        .map(|s| s.to_owned())
        .collect()
}

/// Compute a simple TF-IDF-like overlap score between query and passage.
fn score(query_words: &[String], passage: &str) -> f32 {
    let passage_words = words(passage);
    if passage_words.is_empty() {
        return 0.0;
    }

    // Build passage word frequency map.
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &passage_words {
        *freq.entry(w.as_str()).or_insert(0) += 1;
    }

    // Score = sum of query-word frequencies in passage, normalised by passage length.
    let raw: f32 = query_words.iter()
        .map(|qw| *freq.get(qw.as_str()).unwrap_or(&0) as f32)
        .sum();

    raw / (passage_words.len() as f32).sqrt()
}

/// Answer the question by retrieving the best-matching calendar chunk and
/// extracting the most relevant sentence.
pub fn retrieval_answer(question: &str) -> String {
    let kb = knowledge_base();
    let q_words = words(question);

    // Find the top-K most relevant chunks.
    let mut scored: Vec<(f32, &str)> = kb.iter()
        .map(|(_, chunk)| (score(&q_words, chunk), chunk.as_str()))
        .collect();

    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Attempt to extract a specific answer from the top chunks.
    for &(sc, chunk) in scored.iter().take(5) {
        if sc > 0.0 {
            if let Some(ans) = extract_answer_from_chunk(&q_words, chunk) {
                return ans;
            }
        }
    }

    // Return the top chunk as a general answer if extraction failed.
    if let Some(&(sc, chunk)) = scored.first() {
        if sc > 0.0 {
            return format!("Based on the calendar: {}", chunk);
        }
    }

    "I could not find a specific answer in the calendar documents.".to_owned()
}

/// Attempt to extract a short answer from a matching chunk.
///
/// Heuristics:
///  - If the question asks "how many", return the first number found.
///  - If the question asks "when" / "what date", return a date-like phrase.
///  - Otherwise return the chunk trimmed to 120 chars.
fn extract_answer_from_chunk(q_words: &[String], chunk: &str) -> Option<String> {
    let q_lower = q_words.join(" ");

    // "How many" → extract number.
    if q_lower.contains("how many") || q_lower.contains("count") {
        if let Some(num) = extract_number(chunk) {
            return Some(num);
        }
    }

    // "When" / "what date" / "what month" → extract date phrase.
    if q_lower.contains("when") || q_lower.contains("date") || q_lower.contains("month") {
        if let Some(date) = extract_date_phrase(chunk) {
            return Some(date);
        }
    }

    // Fallback: first 120 characters.
    let trimmed = chunk.trim();
    if trimmed.len() > 5 {
        let end = trimmed.char_indices().nth(120).map(|(i, _)| i).unwrap_or(trimmed.len());
        return Some(trimmed[..end].to_owned());
    }

    None
}

/// Extract the first integer-like token from text.
fn extract_number(text: &str) -> Option<String> {
    for word in text.split_whitespace() {
        let clean: String = word.chars().filter(|c| c.is_ascii_digit()).collect();
        if !clean.is_empty() {
            return Some(clean);
        }
    }
    None
}

/// Extract date-like phrases: "DD Month YYYY", "Month YYYY", "DD Month", etc.
fn extract_date_phrase(text: &str) -> Option<String> {
    let months = [
        "january","february","march","april","may","june",
        "july","august","september","october","november","december",
    ];
    let lower = text.to_lowercase();
    let words: Vec<&str> = text.split_whitespace().collect();

    for (i, word) in words.iter().enumerate() {
        let wl = word.to_lowercase();
        let wl = wl.trim_matches(|c: char| !c.is_alphanumeric());
        if months.contains(&wl) {
            // Try "DD Month YYYY"
            let start = if i > 0 { i - 1 } else { i };
            let end   = (i + 3).min(words.len());
            let phrase: String = words[start..end].join(" ");
            return Some(phrase);
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Neural model inference
// ──────────────────────────────────────────────────────────────────────────────

/// Attempt to answer using the trained Burn model.
/// Returns None if no checkpoint is found.
fn try_neural_answer(question: &str) -> Option<String> {
    use burn::backend::NdArray;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::{Tensor, TensorData};
    use crate::config::ModelConfig;
    use crate::model::QAModel;

    // Load vocabulary.
    let vocab = load_vocab()?;

    // Find the latest checkpoint.
    let checkpoint_path = find_latest_checkpoint()?;
    println!("[inference] Loading model from: {}", checkpoint_path);

    let cfg = ModelConfig::default();
    let device: <NdArray as burn::tensor::backend::Backend>::Device = Default::default();

    // Reconstruct model and load weights.
    let model: QAModel<NdArray> = QAModel::new(&cfg, &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model = match recorder.load(checkpoint_path.as_str().into(), &device) {
        Ok(record) => model.load_record(record),
        Err(e) => {
            eprintln!("[inference] Warning: could not load checkpoint: {}", e);
            return None;
        }
    };

    // Retrieve the best context for this question from the knowledge base.
    let kb = knowledge_base();
    let q_words = words(question);
    let mut scored: Vec<(f32, &str)> = kb.iter()
        .map(|(_, c)| (score(&q_words, c), c.as_str()))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let context = scored.first().map(|(_, c)| *c).unwrap_or("");

    // Encode the (question, context) pair.
    let (input_ids, ctx_start) = vocab.encode_pair(question, context, cfg.max_seq_len);
    let seq_len = input_ids.len();

    let ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let input_tensor = Tensor::<NdArray, 2, burn::tensor::Int>::from_data(
        TensorData::new(ids_i64, [1, seq_len]),
        &device,
    );

    // Forward pass.
    let (start_logits, end_logits) = model.forward(input_tensor);

    // Predict span.
    let start_idx = start_logits.argmax(1).into_scalar() as usize;
    let end_idx   = end_logits.argmax(1).into_scalar() as usize;

    // Map token positions back to context words.
    let ctx_tokens = vocab.encode(context);
    let s = start_idx.saturating_sub(ctx_start);
    let e = end_idx.saturating_sub(ctx_start).min(ctx_tokens.len().saturating_sub(1));

    if s <= e && e < ctx_tokens.len() {
        let answer_tokens = &ctx_tokens[s..=e];
        let answer = vocab.decode(answer_tokens);
        if !answer.trim().is_empty() {
            return Some(answer);
        }
    }

    None
}

/// Load the vocabulary from the checkpoint directory.
fn load_vocab() -> Option<Vocab> {
    let path = "checkpoints/vocab.json";
    let json = fs::read_to_string(path).ok()?;
    serde_json::from_str(&json).ok()
}

/// Find the most recently saved checkpoint file.
fn find_latest_checkpoint() -> Option<String> {
    let dir = fs::read_dir("checkpoints").ok()?;

    let mut checkpoints: Vec<String> = dir
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("model_epoch_") && name.ends_with(".bin") {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    // Sort by epoch number descending.
    checkpoints.sort_by(|a, b| {
        let epoch_a: usize = a.trim_start_matches("model_epoch_")
            .trim_end_matches(".bin")
            .parse().unwrap_or(0);
        let epoch_b: usize = b.trim_start_matches("model_epoch_")
            .trim_end_matches(".bin")
            .parse().unwrap_or(0);
        epoch_b.cmp(&epoch_a)
    });

    checkpoints.first().map(|name| format!("checkpoints/{}", name))
}
