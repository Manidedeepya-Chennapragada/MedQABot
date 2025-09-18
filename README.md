# Medical QA — LoRA Fine-Tuning (QLoRA)
##Overview
## Overview
This project fine-tunes **microsoft/Phi-3-mini-4k-instruct** using **LoRA (PEFT)** with **4-bit QLoRA** on a medical Q&A dataset. The goal is to apply a small, instruction-tuned base model to domain-specific QA while while remaining trainable on a consumer GPU.
##Approach
- **Parameter-Efficient**: LoRA updates a small set of low-rank adapters (r=16, α=32, dropout=0.1) while freezing most base weights.
- **Memory Efficient**: 4-bit quantization (NF4) with compute dtype float16 enables single-GPU training.
- **Stable Training**: Gradient checkpointing and `use_cache=False` minimize activation memory and avoid DynamicCache issues in training.
## Data & Preprocessing
- Input format: pairs of `question` → `answer`.
- Cleaning: lowercasing/whitespace normalization; optional de-duplication; rows with empty fields dropped.
- Tokenization: sequences truncated to **1024** tokens to match the selected context budget.
## Training Setup
- Base model:  **microsoft/Phi-3-mini-4k-instruct**
- LoRA: r=16, alpha=32, dropout=0.1; target modules: attention + MLP.
- Quantization: 4-bit (NF4); compute dtype: fp16 (bf16 preferred if stable).
- Optimizers: paged_adamw_8bit; Scheduler: cosine; Warmup: 100 steps.
(Chosen for stability under deadline; see “Trade-offs” below.)
- Other: gradient_checkpointing=True, remove_unused_columns=False, eval_strategy="steps", save_strategy="steps"
## Assumptions
- The medical Q&A dataset contains authoritative answers (no PHI).
- Questions and answers fit within 1024 tokens.
- Evaluation uses standard **language-modeling loss** (per-token cross-entropy) as a proxy for answer quality.
- Hardware: single **NVIDIA 4070 SUPER (~13 GB)** GPU; training feasible with QLoRA + LoRA.
## Results (observed)
- Final training loss: 
- Final eval loss: 
- Perplexity:

## Strengths
- Efficient adaptation to medical domain without full fine-tuning.
- Fits on consumer GPU with 4-bit quantization and LoRA.
- Checkpointing at intervals with `load_best_model_at_end=True` preserves the best eval loss reduces overfitting risk.
## Limitations
- LM loss is not equal to the factual correctness; medical QA needs grounded evaluation.
- No retrieval grounding → risk of hallucinations on rare/edge cases.
## Potential Improvements
1. **bf16 compute** (if stable): set `bnb_4bit_compute_dtype=torch.bfloat16`, `bf16=True, fp16=False`.
2. **Sequence packing**: Improve throughput by packing multiple short samples.
3. **Curriculum / length-aware sampling**: start with short QA, ramp to longer rationale questions.
4. **RAG hybridization**: ground answers with citations from vetted medical sources.
5. **LoRA search**: tune r/alpha (e.g., r∈{8,16,32}) and target modules for best quality/speed.
6. **Safety layers**: classifier or rule-based filters to prevent unsafe advice.
7. **Metrics**: add domain-specific scoring and error taxonomy for qualitative analysis.
8. **Mixed dataset**: include general instruction data (10–20%) to reduce overfitting.

## Repro Notes
- Transformers version parsed in notebook; use `eval_strategy="steps"` on **≥4.46**.
- Ensure `self.model.config.use_cache=False` to avoid DynamicCache errors during training.
- For 1024 tokens on a 4070 SUPER, effective batch **16–32** is usually sufficient; your run used a larger value by design for stability.
