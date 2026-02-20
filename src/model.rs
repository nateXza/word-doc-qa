// model.rs — Transformer-based Extractive Q&A model.
//
// Architecture overview:
//
//   input_ids  →  TokenEmbedding  ─┐
//                                  ├─ (sum) → TransformerEncoder × N layers → OutputProjection
//   positions  →  PositionalEmbed ─┘
//
// The output projection produces (start_logits, end_logits), one scalar per
// token position.  The predicted answer span is argmax(start_logits) ..
// argmax(end_logits).  This mirrors the extractive approach of classic BERT-QA.
//
// The model is generic over the Burn Backend trait so it can be compiled for
// CPU (NdArray), GPU (WGPU), or any other Burn backend without code changes.

use burn::{
    module::Module,
    nn::{
        Embedding, EmbeddingConfig,
        LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
        Dropout, DropoutConfig,
        attention::{MultiHeadAttention, MultiHeadAttentionConfig},
    },
    tensor::{Tensor, backend::Backend},
};

use crate::config::ModelConfig;

// ──────────────────────────────────────────────────────────────────────────────
// Token Embedding
// ──────────────────────────────────────────────────────────────────────────────

/// Learned token embeddings: vocabulary → d_model.
#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    embedding: Embedding<B>,
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(device: &B::Device, vocab_size: usize, d_model: usize) -> Self {
        let embedding = EmbeddingConfig::new(vocab_size, d_model).init(device);
        Self { embedding }
    }

    /// Forward: [batch, seq_len] → [batch, seq_len, d_model]
    pub fn forward(&self, input_ids: Tensor<B, 2, burn::tensor::Int>) -> Tensor<B, 3> {
        self.embedding.forward(input_ids)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Positional Embedding
// ──────────────────────────────────────────────────────────────────────────────

/// Learned positional embeddings: position → d_model.
/// Using learned rather than sinusoidal to keep the code idiomatic with Burn
/// and to allow the positions to be task-adapted during fine-tuning.
#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    embedding: Embedding<B>,
    max_len: usize,
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(device: &B::Device, max_len: usize, d_model: usize) -> Self {
        let embedding = EmbeddingConfig::new(max_len, d_model).init(device);
        Self { embedding, max_len }
    }

    /// Forward: [batch, seq_len, d_model] → [batch, seq_len, d_model]
    /// Adds position encodings to the token embeddings already supplied.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, seq, _] = x.dims();
        let device = x.device();

        // Build position indices [0, 1, …, seq-1] and broadcast to [batch, seq].
        let positions: Vec<i64> = (0..seq as i64).collect();
        let pos_tensor = Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(positions, [seq]),
            &device,
        )
        .unsqueeze::<2>()        // [1, seq]
        .expand([batch, seq]);   // [batch, seq]

        let pos_emb = self.embedding.forward(pos_tensor); // [batch, seq, d_model]
        x + pos_emb
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Feed-Forward Sub-layer
// ──────────────────────────────────────────────────────────────────────────────

/// Position-wise feed-forward network: two linear layers with GELU activation.
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(device: &B::Device, d_model: usize, d_ff: usize, dropout: f64) -> Self {
        let fc1 = LinearConfig::new(d_model, d_ff).with_bias(true).init(device);
        let fc2 = LinearConfig::new(d_ff, d_model).with_bias(true).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        Self { fc1, fc2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        self.fc2.forward(x)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Single Transformer Encoder Layer
// ──────────────────────────────────────────────────────────────────────────────

/// One encoder layer = Multi-Head Self-Attention + Add&Norm + FFN + Add&Norm.
#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    self_attn: MultiHeadAttention<B>,
    norm1: LayerNorm<B>,
    ff: FeedForward<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
}

impl<B: Backend> EncoderLayer<B> {
    pub fn new(device: &B::Device, d_model: usize, n_heads: usize, d_ff: usize, dropout: f64) -> Self {
        let self_attn = MultiHeadAttentionConfig::new(d_model, n_heads)
            .with_dropout(dropout)
            .init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let ff = FeedForward::new(device, d_model, d_ff, dropout);
        let dropout = DropoutConfig::new(dropout).init();
        Self { self_attn, norm1, ff, norm2, dropout }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Self-attention sub-layer with residual connection.
        let attn_input = x.clone();
        let mha_output = self.self_attn.forward(
            burn::nn::attention::MhaInput::self_attn(attn_input),
        );
        let x = self.norm1.forward(x + self.dropout.forward(mha_output.context));

        // Feed-forward sub-layer with residual connection.
        let ff_out = self.ff.forward(x.clone());
        self.norm2.forward(x + self.dropout.forward(ff_out))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Stacked Transformer Encoder (≥ 6 layers)
// ──────────────────────────────────────────────────────────────────────────────

/// Stack of N encoder layers.
#[derive(Module, Debug)]
pub struct TransformerEncoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
}

impl<B: Backend> TransformerEncoder<B> {
    pub fn new(
        device: &B::Device,
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
    ) -> Self {
        assert!(n_layers >= 6, "Assignment requires at least 6 encoder layers (got {})", n_layers);

        let layers = (0..n_layers)
            .map(|_| EncoderLayer::new(device, d_model, n_heads, d_ff, dropout))
            .collect();

        Self { layers }
    }

    /// Forward: [batch, seq, d_model] → [batch, seq, d_model]
    pub fn forward(&self, mut x: Tensor<B, 3>) -> Tensor<B, 3> {
        for layer in &self.layers {
            x = layer.forward(x);
        }
        x
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Output Projection Layer (span extraction head)
// ──────────────────────────────────────────────────────────────────────────────

/// Projects encoder output to (start_logits, end_logits) per token.
#[derive(Module, Debug)]
pub struct SpanHead<B: Backend> {
    proj: Linear<B>,
}

impl<B: Backend> SpanHead<B> {
    pub fn new(device: &B::Device, d_model: usize) -> Self {
        // d_model → 2: one score for start, one for end.
        let proj = LinearConfig::new(d_model, 2).with_bias(true).init(device);
        Self { proj }
    }

    /// Forward: [batch, seq, d_model] → (start_logits, end_logits) both [batch, seq]
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let logits = self.proj.forward(x); // [batch, seq, 2]
        let [batch, seq, _] = logits.dims();

        // Select index 0 along dim 2 → start logits [batch, seq]
        let start = logits.clone()
            .slice([0..batch, 0..seq, 0..1])
            .reshape([batch, seq]);

        // Select index 1 along dim 2 → end logits [batch, seq]
        let end = logits
            .slice([0..batch, 0..seq, 1..2])
            .reshape([batch, seq]);

        (start, end)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Complete Q&A Model
// ──────────────────────────────────────────────────────────────────────────────

/// Full transformer-based Q&A model, generic over the Burn Backend.
///
/// Input  : integer token-id tensor [batch, seq_len]
/// Output : (start_logits, end_logits), each [batch, seq_len]
#[derive(Module, Debug)]
pub struct QAModel<B: Backend> {
    token_emb:   TokenEmbedding<B>,
    pos_emb:     PositionalEmbedding<B>,
    encoder:     TransformerEncoder<B>,
    span_head:   SpanHead<B>,
    dropout:     Dropout,
}

impl<B: Backend> QAModel<B> {
    /// Construct the model from a `ModelConfig` and a Burn device.
    pub fn new(cfg: &ModelConfig, device: &B::Device) -> Self {
        let token_emb = TokenEmbedding::new(device, cfg.vocab_size, cfg.d_model);
        let pos_emb   = PositionalEmbedding::new(device, cfg.max_seq_len, cfg.d_model);
        let encoder   = TransformerEncoder::new(
            device, cfg.n_layers, cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout
        );
        let span_head = SpanHead::new(device, cfg.d_model);
        let dropout   = DropoutConfig::new(cfg.dropout).init();

        Self { token_emb, pos_emb, encoder, span_head, dropout }
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `input_ids` — Integer tensor of shape [batch, seq_len].
    ///
    /// # Returns
    /// `(start_logits, end_logits)` — Float tensors of shape [batch, seq_len].
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, burn::tensor::Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        // 1. Token + positional embeddings (sum), then dropout.
        let tok = self.token_emb.forward(input_ids);   // [B, S, D]
        let x   = self.pos_emb.forward(tok);           // [B, S, D]
        let x   = self.dropout.forward(x);

        // 2. N-layer transformer encoder.
        let encoded = self.encoder.forward(x);         // [B, S, D]

        // 3. Span-extraction head → logits.
        self.span_head.forward(encoded)
    }

    /// Compute cross-entropy loss for span extraction.
    ///
    /// Both start and end positions are trained with independent softmax CE loss,
    /// as in the original BERT-QA paper.
    pub fn loss(
        &self,
        start_logits: Tensor<B, 2>,
        end_logits:   Tensor<B, 2>,
        start_targets: Tensor<B, 1, burn::tensor::Int>,
        end_targets:   Tensor<B, 1, burn::tensor::Int>,
    ) -> Tensor<B, 1> {
        use burn::nn::loss::CrossEntropyLossConfig;

        let loss_fn = CrossEntropyLossConfig::new().init(&start_logits.device());

        let loss_start = loss_fn.forward(start_logits, start_targets);
        let loss_end   = loss_fn.forward(end_logits, end_targets);

        (loss_start + loss_end) / 2.0
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Model parameter summary helper
// ──────────────────────────────────────────────────────────────────────────────

/// Print a rough parameter count for the model.
pub fn count_parameters(cfg: &ModelConfig) -> usize {
    let embed_params = cfg.vocab_size * cfg.d_model       // token embedding
                     + cfg.max_seq_len * cfg.d_model;     // positional embedding

    // Per encoder layer:
    //   4 matrices for MHA (Q, K, V, out) each d_model × d_model + bias
    //   2 linear layers for FFN: d_model×d_ff and d_ff×d_model + biases
    //   2 LayerNorms: 2 × d_model
    let mha_params = 4 * (cfg.d_model * cfg.d_model + cfg.d_model);
    let ff_params  = 2 * cfg.d_model * cfg.d_ff
                   + cfg.d_ff + cfg.d_model;              // biases
    let ln_params  = 2 * 2 * cfg.d_model;
    let layer_params = mha_params + ff_params + ln_params;

    let encoder_params = cfg.n_layers * layer_params;
    let head_params    = cfg.d_model * 2 + 2;            // output projection

    embed_params + encoder_params + head_params
}
