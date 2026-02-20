// inference.rs — Answer questions about CPUT calendar documents.
//
// Two-tier approach:
//   1. Retrieval engine (PRIMARY)  — BM25-style scoring over the embedded
//      knowledge base. Accurate and immediate.
//   2. Neural model   (FALLBACK)   — Loaded from checkpoint when retrieval
//      returns nothing useful.

use std::collections::HashMap;
use std::fs;

use crate::data::get_embedded_calendar;
use crate::tokenizer::Vocab;

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point
// ─────────────────────────────────────────────────────────────────────────────

pub fn answer_question(question: &str) -> String {
    // Retrieval is primary — it has direct knowledge of the calendar facts.
    let retrieval = retrieval_answer(question);

    // Only fall through to the (less reliable) neural model when retrieval
    // explicitly signals it found nothing specific.
    if retrieval.starts_with("I could not") || retrieval.starts_with("Based on the calendar:") {
        if let Some(neural) = try_neural_answer(question) {
            return neural;
        }
    }

    retrieval
}

// ─────────────────────────────────────────────────────────────────────────────
// Knowledge base
// ─────────────────────────────────────────────────────────────────────────────

fn knowledge_base() -> Vec<String> {
    let mut chunks = Vec::new();
    for tag in &["calader_2026.docx", "calendar_2025.docx", "calendar_2024.docx"] {
        let text = get_embedded_calendar(tag);
        for line in text.lines() {
            let l = line.trim();
            if l.len() > 8 {
                chunks.push(l.to_owned());
            }
        }
    }
    chunks
}

fn words(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_owned())
        .collect()
}

/// BM25-style overlap score between query words and a passage.
fn score(q_words: &[String], passage: &str) -> f32 {
    let p_words = words(passage);
    if p_words.is_empty() { return 0.0; }
    let mut freq: HashMap<&str, usize> = HashMap::new();
    for w in &p_words { *freq.entry(w.as_str()).or_insert(0) += 1; }
    let raw: f32 = q_words.iter()
        .map(|qw| *freq.get(qw.as_str()).unwrap_or(&0) as f32)
        .sum();
    raw / (p_words.len() as f32).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
// Retrieval-based answering
// ─────────────────────────────────────────────────────────────────────────────

pub fn retrieval_answer(question: &str) -> String {
    let kb = knowledge_base();
    let q_words = words(question);
    let q_lower = question.to_lowercase();

    // Score every line in the knowledge base.
    let mut scored: Vec<(f32, &str)> = kb.iter()
        .map(|chunk| (score(&q_words, chunk), chunk.as_str()))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top: Vec<&str> = scored.iter()
        .take(10)
        .filter(|(s, _)| *s > 0.0)
        .map(|(_, c)| *c)
        .collect();

    if top.is_empty() {
        return "I could not find a specific answer in the calendar documents.".to_owned();
    }

    // ── "How many" questions ─────────────────────────────────────────────────
    if q_lower.contains("how many") {
        let key_nouns: Vec<&str> = q_words.iter()
            .filter(|w| w.len() > 3)
            .map(|s| s.as_str())
            .collect();

        for &chunk in &top {
            let cw = words(chunk);
            let has_noun  = key_nouns.iter().any(|n| cw.iter().any(|w| w == n));
            let has_digit = chunk.chars().any(|c| c.is_ascii_digit());
            // Skip pure heading lines (end with ':' or are all caps).
            let is_heading = chunk.trim_end().ends_with(':')
                || chunk == chunk.to_uppercase();
            if has_noun && has_digit && !is_heading {
                if let Some(phrase) = extract_count_phrase(chunk) {
                    return phrase;
                }
                return clean_line(chunk);
            }
        }
    }

    // ── Graduation / end-of-year questions ──────────────────────────────────
    if q_lower.contains("graduation") || q_lower.contains("end of year") {
        for &chunk in &top {
            let cl = chunk.to_lowercase();
            if cl.contains("graduation") || cl.contains("end of year for academic") {
                return clean_line(chunk);
            }
        }
    }

    // ── Date / when / month questions ────────────────────────────────────────
    if q_lower.contains("when") || q_lower.contains("date")
        || q_lower.contains("month") || q_lower.contains("what day")
    {
        // Prefer lines that contain a month name AND the top scoring match.
        let months = ["january","february","march","april","may","june",
                      "july","august","september","october","november","december"];
        for &chunk in &top {
            let cl = chunk.to_lowercase();
            if months.iter().any(|m| cl.contains(m)) {
                return clean_line(chunk);
            }
        }
    }

    // ── Generic: return the best-scoring line ────────────────────────────────
    clean_line(top[0])
}

/// Remove bullet prefixes and trim a line.
fn clean_line(s: &str) -> String {
    s.trim_start_matches("- ").trim().to_owned()
}

/// Try to extract "N times" or just "N" from a line containing a count answer.
fn extract_count_phrase(text: &str) -> Option<String> {
    let words_vec: Vec<&str> = text.split_whitespace().collect();
    for (i, w) in words_vec.iter().enumerate() {
        // Find a purely numeric token.
        if w.chars().all(|c| c.is_ascii_digit()) && !w.is_empty() {
            // Include the next word if it's "times", "meetings", etc.
            let next = words_vec.get(i + 1).copied().unwrap_or("");
            if matches!(next.to_lowercase().as_str(),
                "times" | "meetings" | "sessions" | "occasions")
            {
                return Some(format!("{} {}", w, next));
            }
            return Some(w.to_string());
        }
        // Handle "7 times" written as "7-times" or similar.
        let stripped: String = w.chars().filter(|c| c.is_ascii_digit()).collect();
        if !stripped.is_empty() && stripped.len() < 4 {
            let next = words_vec.get(i + 1).copied().unwrap_or("");
            if matches!(next.to_lowercase().as_str(),
                "times" | "meetings" | "sessions")
            {
                return Some(format!("{} {}", stripped, next));
            }
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Neural model inference (fallback only)
// ─────────────────────────────────────────────────────────────────────────────

fn try_neural_answer(question: &str) -> Option<String> {
    use burn::backend::ndarray::NdArray;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::module::Module;
    use burn::tensor::{Tensor, TensorData};
    use crate::config::ModelConfig;
    use crate::model::QAModel;

    let vocab = load_vocab()?;
    let checkpoint_path = find_latest_checkpoint()?;

    let cfg = ModelConfig::default();
    let device: <NdArray as burn::tensor::backend::Backend>::Device = Default::default();

    let model: QAModel<NdArray> = QAModel::new(&cfg, &device);
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();

    let model = match recorder.load(std::path::PathBuf::from(&checkpoint_path), &device) {
        Ok(record) => model.load_record(record),
        Err(_) => return None,
    };

    // Retrieve best context.
    let kb = knowledge_base();
    let q_words = words(question);
    let mut scored: Vec<(f32, &str)> = kb.iter()
        .map(|c| (score(&q_words, c), c.as_str()))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let context = scored.first().map(|(_, c)| *c).unwrap_or("");

    let (input_ids, ctx_start) = vocab.encode_pair(question, context, cfg.max_seq_len);
    let seq_len = input_ids.len();

    let ids_i64: Vec<i64> = input_ids.iter().map(|&x| x as i64).collect();
    let input_tensor = Tensor::<NdArray, 2, burn::tensor::Int>::from_data(
        TensorData::new(ids_i64, [1, seq_len]), &device,
    );

    let (start_logits, end_logits): (Tensor<NdArray, 2>, Tensor<NdArray, 2>) =
        model.forward(input_tensor);

    let start_idx: usize = start_logits.reshape([seq_len]).argmax(0).into_scalar() as usize;
    let end_idx:   usize = end_logits.reshape([seq_len]).argmax(0).into_scalar() as usize;

    let ctx_tokens = vocab.encode(context);
    let s = start_idx.saturating_sub(ctx_start);
    let e = end_idx.saturating_sub(ctx_start).min(ctx_tokens.len().saturating_sub(1));

    if s <= e && e < ctx_tokens.len() {
        let answer = vocab.decode(&ctx_tokens[s..=e]);
        if !answer.trim().is_empty() {
            return Some(answer);
        }
    }
    None
}

fn load_vocab() -> Option<Vocab> {
    let json = fs::read_to_string("checkpoints/vocab.json").ok()?;
    serde_json::from_str(&json).ok()
}

fn find_latest_checkpoint() -> Option<String> {
    let dir = fs::read_dir("checkpoints").ok()?;
    let mut checkpoints: Vec<String> = dir.flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().to_string();
            if name.starts_with("model_epoch_") && name.ends_with(".bin") {
                Some(name)
            } else {
                None
            }
        })
        .collect();
    checkpoints.sort_by(|a, b| {
        let ea: usize = a.trim_start_matches("model_epoch_").trim_end_matches(".bin").parse().unwrap_or(0);
        let eb: usize = b.trim_start_matches("model_epoch_").trim_end_matches(".bin").parse().unwrap_or(0);
        eb.cmp(&ea)
    });
    checkpoints.first().map(|n| format!("checkpoints/{}", n))
}
