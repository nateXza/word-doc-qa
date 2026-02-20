# word-doc-qa

**SEG 580S Assignment 1** — Q&A system for CPUT academic calendar documents,
built with Rust and the [Burn](https://burn.dev/) deep learning framework.

## Quick Start

```bash
# 1. Copy calendar documents into the data/ folder
cp /path/to/calader_2026.docx    data/
cp /path/to/calendar_2025.docx   data/
cp /path/to/calendar_2024.docx   data/

# 2. Train the model
cargo run --release -- train

# 3. Ask a question
cargo run --release -- ask "What is the month and date of the 2026 End of Year Graduation?"

# 4. Run demo questions
cargo run --release -- demo
```

## Project Structure

```
word-doc-qa/
├── Cargo.toml           # Dependencies (pinned as per assignment)
├── src/
│   ├── main.rs          # CLI entry point
│   ├── config.rs        # Hyperparameters & model config
│   ├── data.rs          # Document loading & Q&A pair generation
│   ├── tokenizer.rs     # Word-level vocabulary & encoding
│   ├── model.rs         # Transformer encoder Q&A model
│   ├── training.rs      # Training loop with metrics & checkpoints
│   └── inference.rs     # Retrieval + neural inference engine
├── data/                # Place .docx files here
├── docs/
│   └── REPORT.md        # Project report (Section 1–4)
└── checkpoints/         # Created during training
    ├── vocab.json
    ├── model_epoch_5.bin
    ├── ...
    └── metrics.json
```

## Architecture

Transformer-based extractive Q&A model (BERT-QA style):

- **6 encoder layers** (configurable, ≥ 6 as required)
- **8 attention heads**, d_model = 256
- **Span extraction head**: predicts start + end token positions
- Generic over the Burn `Backend` trait (CPU or GPU)

See `docs/REPORT.md` for full architecture diagram and results.

## Demo Answers

| Question | Answer |
|----------|--------|
| What is the month/date of the 2026 End of Year Graduation? | December 2026 |
| How many times did the HDC meet in 2024? | 7 times |
| When does Term 1 start in 2026? | 26 January |
| When is Good Friday in 2026? | 3 April 2026 |
| When does Term 4 end in 2025? | 12 December 2025 |
