Extractive Question Answering with Transformers — Project Report
1) Overview

This project builds an extractive question answering (QA) system: given a question and a context paragraph, the model returns the exact answer span found inside the context. We fine-tuned a Transformer model (DistilBERT) for span extraction on SQuAD-style data, evaluated with Exact Match (EM) and F1, and saved the final model for inference.

2) Data

Format: CSV with columns: context, question, answer, answer_start, answer_end.

Alignment rule: the gold answer must match the context slice, i.e. context[answer_start:answer_end] == answer.

If answer_end missing: computed as answer_start + len(answer).

Split: ~80% train, 10% validation, 10% test (random, shuffled, fixed seed for reproducibility).

3) Preprocessing (what happens before training)

Fast tokenization: We used a fast tokenizer (use_fast=True), which exposes offset mapping so we can connect character indices in the context to token indices used by the model.

Input packing: The model sees a single sequence:
[CLS] question tokens [SEP] context tokens [SEP]

Char → token alignment:
We convert the character-level (answer_start, answer_end) into token-level (start_positions, end_positions) using the tokenizer’s character-to-token mapping for the context portion.

Length management: Inputs are truncated to a maximum sequence length (e.g., 384 tokens) with truncation="only_second" so that only the context is shortened when necessary.

Out-of-window fallback: If an answer falls outside the current window (rare but possible when the context is long), the span label defaults to the [CLS] position. (For production setups, a sliding window / doc_stride is recommended to cover long contexts robustly.)

4) Model

Base encoder: DistilBERT (uncased) loaded from a public checkpoint.

QA head: A lightweight span-prediction head on top of the encoder produces two scores per token:
start logits (probability the token starts the answer) and end logits (probability the token ends the answer).

Initialization note: The QA head is newly initialized (the base encoder is pre-trained on language), so fine-tuning is necessary.

5) Training setup

Objective: Cross-entropy loss on start and end positions.

Optimizer & schedule: Handled by the Trainer API (AdamW under the hood).

Key hyperparameters:

Max sequence length: ~384 tokens

Batch size: 8 (per device)

Learning rate: 3e-5

Epochs: 3 (typical; early runs used 1 for a quick sanity check)

Validation during training: Evaluated each epoch on the validation split; best model kept (lowest validation loss).

Hardware: CUDA GPU (cuda:0) when available.

6) Inference (how the model answers)

The question and context are tokenized and packed together.

The encoder computes contextual representations for every token.

The QA head outputs start/end logits for each token.

The best (start, end) pair is selected (with basic constraints like start ≤ end and reasonable span length).

Token indices are mapped back to character indices via offsets; the answer substring is returned from the original context.

7) Evaluation metrics

Exact Match (EM): 1.0 if the predicted answer string exactly equals the gold answer string (after basic trimming); otherwise 0.0. Aggregated as a mean across examples.

F1: Word-overlap F1 between predicted and gold answers after simple normalization (lowercasing, removing punctuation/extra symbols, and supporting Arabic digits/letters). F1 rewards near-misses where the meaning matches but the string isn’t identical.

8) Results (test set)

Test EM: ~0.63

Test F1: ~0.82
These are solid baseline numbers for a compact encoder (DistilBERT) on SQuAD-style data without heavy tuning. F1 > EM indicates the model often captures most of the correct phrase even when not an exact string match.

9) Artifacts (what to save/share)

The fine-tuned model and tokenizer are saved in a folder (e.g., qa-final/). Important files include:

Model: model.safetensors (or pytorch_model.bin) and config.json

Tokenizer: tokenizer.json, vocab.txt, tokenizer_config.json, special_tokens_map.json (and added_tokens.json if present)

This folder can be loaded directly for inference in any environment that supports Hugging Face Transformers.

10) Known limitations & next steps

Long contexts: If many answers fall outside the 384-token window, enable sliding windows with a doc_stride (e.g., 128). This typically boosts EM/F1 on long paragraphs.

Model capacity: Switching to a stronger base (e.g., RoBERTa-base) usually improves accuracy at the cost of speed/size.

Training budget: More epochs (e.g., 4–5), careful learning-rate tuning, and early stopping can yield further gains.

Data quality: Ensure perfect alignment (context[answer_start:answer_end] == answer); misaligned labels harm learning disproportionately.

11) High-level reproduction steps

Prepare a SQuAD-style CSV with aligned character spans.

Split into train/validation/test (e.g., 80/10/10).

Use a fast tokenizer to build inputs and map character spans to token spans.

Fine-tune DistilBERT (or another base) for span extraction; validate each epoch and keep the best checkpoint.

Report EM and F1 on the held-out test set.

Save the final model + tokenizer for deployment.

12) Acknowledgements

Built with 🤗 Hugging Face Transformers and Datasets. DistilBERT by Hugging Face; SQuAD formulation by Stanford.
