// config.rs — Hyperparameters and model/training configuration.
// Centralising all tunable values here makes comparison of configurations simple.

use serde::{Deserialize, Serialize};

/// Top-level training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of full passes over the training data.
    pub epochs: usize,
    /// Samples per gradient-update step.
    pub batch_size: usize,
    /// Adam learning rate.
    pub learning_rate: f64,
    /// Fraction of data held out for validation.
    pub val_split: f64,
    /// Save a checkpoint every N epochs.
    pub checkpoint_every: usize,
    /// Directory in which checkpoints are written.
    pub checkpoint_dir: String,
    /// Transformer model hyperparameters.
    pub model: ModelConfig,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 30,
            batch_size: 8,
            learning_rate: 1e-4,
            val_split: 0.15,
            checkpoint_every: 5,
            checkpoint_dir: "checkpoints".into(),
            model: ModelConfig::default(),
        }
    }
}

impl TrainingConfig {
    /// Alternative smaller configuration for fast experimentation / comparison.
    pub fn small() -> Self {
        Self {
            epochs: 15,
            batch_size: 4,
            learning_rate: 3e-4,
            model: ModelConfig::small(),
            ..Default::default()
        }
    }
}

/// Transformer architecture hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Vocabulary size (must match the tokenizer).
    pub vocab_size: usize,
    /// Maximum sequence length (tokens).
    pub max_seq_len: usize,
    /// Model / embedding dimensionality.
    pub d_model: usize,
    /// Number of multi-head attention heads.
    pub n_heads: usize,
    /// Number of stacked transformer encoder layers (≥ 6 required).
    pub n_layers: usize,
    /// Feed-forward hidden dimension inside each encoder layer.
    pub d_ff: usize,
    /// Dropout probability applied throughout the model.
    pub dropout: f64,
    /// Size of the answer-span projection head (start + end logits).
    pub output_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8000,
            max_seq_len: 512,
            d_model: 256,
            n_heads: 8,
            n_layers: 6,     // requirement: minimum 6 layers
            d_ff: 1024,
            dropout: 0.1,
            output_size: 2,  // start-position logit + end-position logit
        }
    }
}

impl ModelConfig {
    /// Smaller variant for architecture comparison experiments.
    pub fn small() -> Self {
        Self {
            d_model: 128,
            n_heads: 4,
            n_layers: 6,  // still minimum 6
            d_ff: 512,
            ..Default::default()
        }
    }
}
