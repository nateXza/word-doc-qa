// word-doc-qa: Q&A system over CPUT calendar documents
// Uses Burn deep learning framework with a transformer encoder architecture.
//
// Usage:
//   cargo run -- train                       # Train the model
//   cargo run -- ask "Your question here"   # Ask a question
//   cargo run -- demo                        # Run built-in demo questions

mod config;
mod data;
mod model;
mod training;
mod inference;
mod tokenizer;

use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "train" => {
            println!("=== Training Q&A Model ===");
            training::train_model();
        }
        "ask" => {
            if args.len() < 3 {
                eprintln!("Error: Please provide a question.");
                eprintln!("Usage: cargo run -- ask \"Your question here\"");
                std::process::exit(1);
            }
            let question = args[2..].join(" ");
            println!("=== Q&A Inference ===");
            println!("Question: {}", question);
            let answer = inference::answer_question(&question);
            println!("Answer: {}", answer);
        }
        "demo" => {
            println!("=== Demo Questions ===\n");
            run_demo();
        }
        "extract" => {
            println!("=== Extracting Document Data ===");
            let docs = data::load_documents(&["data/calader_2026.docx",
                                              "data/calendar_2025.docx",
                                              "data/calendar_2024.docx"]);
            for doc in &docs {
                println!("Loaded: {} ({} chars)", doc.filename, doc.content.len());
            }
        }
        _ => {
            eprintln!("Unknown command: {}", args[1]);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn print_usage() {
    println!("word-doc-qa: Transformer-based Q&A system for CPUT calendar documents");
    println!();
    println!("USAGE:");
    println!("  cargo run -- train              Train the model on loaded documents");
    println!("  cargo run -- ask \"<question>\"  Answer a question about the documents");
    println!("  cargo run -- demo               Run demonstration questions");
    println!("  cargo run -- extract            Extract and show document content");
    println!();
    println!("EXAMPLES:");
    println!("  cargo run -- ask \"What month will the 2026 End of Year Graduation be held?\"");
    println!("  cargo run -- ask \"How many times did the HDC hold their meetings in 2024?\"");
}

fn run_demo() {
    let questions = vec![
        "What is the month and date of the 2026 End of Year Graduation Ceremony?",
        "How many times did the HDC hold their meetings in 2024?",
        "When does Term 1 start in 2026?",
        "What public holidays occur in April 2026?",
        "When is the Annual Open Day in 2026?",
        "When does Term 4 end in 2025?",
        "How many Senate meetings are in the 2024 calendar?",
    ];

    for q in &questions {
        let answer = inference::answer_question(q);
        println!("Q: {}", q);
        println!("A: {}\n", answer);
    }
}
