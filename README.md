# Medical QA — LoRA Fine-Tuning (QLoRA)
## Overview
This project fine-tunes **microsoft/Phi-3-mini-4k-instruct** using **LoRA (PEFT)** with **4-bit QLoRA** on a medical Q&A dataset. The goal is to apply a small, instruction-tuned base model to domain-specific QA while remaining trainable on a consumer GPU.

## Approach
- **Parameter-Efficient**: LoRA updates a small set of low-rank adapters (r=16, α=32, dropout=0.1) while freezing most base weights.
- **Memory Efficient**: 4-bit quantization (NF4) with compute dtype float16 enables single-GPU training.
- **Stable Training**: Gradient checkpointing and `use_cache=False` minimize activation memory and avoid DynamicCache issues in training.
## Data & Preprocessing
- **Input format:** Original dataset stored as a CSV with two columns: `question` and `answer`.

- **Cleaning pipeline:**
  - **Missing values:** Any row with a null/NaN in `question` or `answer` was dropped to prevent incomplete samples.
  - **Duplicates:** Exact duplicates (where both `question` and `answer` were identical) were removed to avoid model bias toward repeated examples.
  - **Case normalization:** All text converted to lowercase, ensuring consistent tokenization and reducing vocabulary size.
  - **Whitespace normalization:** Leading/trailing spaces stripped; multiple spaces or tabs replaced with a single space.
  - **Regex-based cleanup:** Applied regex substitutions to eliminate unwanted characters (extra punctuation, special symbols) and normalize spacing.
  - **Length filter:** Removed rows with very short or empty questions/answers, since they often represent noise or ambiguous samples (e.g., “?” or one-word answers).
  - **Ambiguity handling:** Ambiguous or context-poor questions (e.g., “Why?” without subject) were excluded during filtering to improve training signal.

- **Splitting strategy:**
  - The cleaned dataset was partitioned into **training, validation, and test sets**.  
  - To prevent data leakage, near-duplicates or semantically identical Q&A pairs were kept within the same split.  
  - Typical proportions: **80% train, 10% validation, 10% test**. Validation was used for hyperparameter tuning and checkpoint selection, while the test set was held out for final evaluation.

- **Tokenization:**
  - Each question–answer pair was tokenized with the model’s tokenizer.
  - Maximum sequence length set to **1024 tokens**: long enough for most medical rationales, but safe for training on a 4070 SUPER (13 GB VRAM).
  - Longer samples were truncated; shorter samples padded with the model’s `pad_token`.
    
## Training Setup
- Base model:  **microsoft/Phi-3-mini-4k-instruct**
- LoRA: r=16, alpha=32, dropout=0.1; target modules: attention + MLP.
- Quantization: 4-bit (NF4); compute dtype: fp16 (bf16 preferred if stable).
- Optimizers: paged_adamw_8bit; Scheduler: cosine; Warmup: 100 steps.
(Chosen for stability under deadline; see “Trade-offs” below.)
- Other: gradient_checkpointing=True, remove_unused_columns=False, eval_strategy="steps", save_strategy="steps"
## System Architecture (Pipeline Flow)
The overall workflow for the medical QA system can be summarized as:  
**Raw Dataset → Preprocessing (cleaning, deduplication, splitting) → Tokenization → Fine-tuning (LoRA + QLoRA on Phi-3 Mini) → Evaluation (loss, perplexity, qualitative checks) → Inference (user query → model answer).**  

## Assumptions
- The medical Q&A dataset contains authoritative answers (no PHI).
- Questions and answers fit within 1024 tokens.
- Evaluation uses standard **language-modeling loss** (per-token cross-entropy) as a proxy for answer quality.
- Hardware: single **NVIDIA 4070 SUPER (~12 GB)** GPU; training feasible with QLoRA + LoRA.
## Results

### Training & Validation
- Best validation (@ step 400): loss 1.0165, perplexity 2.7636
- Final training (@ step 400): loss 0.8495
- Generalization gap (val loss – train loss) @ 400: 0.1670 (up from 0.1182 @ 100)
These numbers show stable learning with limited overfitting.

### Evaluation (100-sample subset)
- **ROUGE-1:** 0.3678  
- **ROUGE-2:** 0.1755  
- **ROUGE-L:** 0.2597  
- **BLEU:** 0.0724
For a sample size of 100 the ROUGE-1/L capture key terms, ROUGE-2 and BLEU are low due to phrasing variance. Loss/perplexity trends confirm learning.
### Takeaway
Model shows steady progress; further gains need longer training, better decoding, or semantic metrics

## Strengths
- Efficient adaptation to medical domain without full fine-tuning.
- Fits on consumer GPU with 4-bit quantization and LoRA.
- Checkpointing at intervals with `load_best_model_at_end=True` preserves the best eval loss reduces overfitting risk.

## Limitations
- LM loss is not equal to the factual correctness, a medical QA needs grounded evaluation.
- No retrieval grounding → risk of hallucinations on rare/edge cases.
  
## Potential Improvements
1. **bf16 compute** (if stable): set `bnb_4bit_compute_dtype=torch.bfloat16`, `bf16=True, fp16=False`.
2. **Sequence packing**: Improve throughput by packing multiple short samples.
3. **Curriculum / length-aware sampling**: start with short QA, ramp to longer rationale questions.
4. **RAG hybridization**: ground answers with citations from vetted medical sources.
5. **LoRA search**: tune r/alpha (e.g., r∈{8,16,32}) and target modules for best quality/speed.
6. **Safety layers**: classifier or rule-based filters to prevent unsafe advice.
7. **Metrics**: add domain-specific scoring for full dataset

## Future Work: Retrieval-Augmented Generation (RAG)  
While the current system relies purely on fine-tuning, incorporating **RAG** would enhance factual correctness by grounding responses in verified medical sources. A retrieval layer could fetch relevant passages from trusted datasets or medical knowledge bases, which the model would then use as context during generation. This approach would reduce hallucinations and improve reliability for edge-case queries.  

## Reproduction Notes
- Transformers version, use `eval_strategy="steps"` on **≥4.46**.
- Ensure `self.model.config.use_cache=False` to avoid DynamicCache errors during training.
- For 1024 tokens on a 4070 SUPER, effective batch **16–32** is usually sufficient; your run used a larger value by design for stability.

-----
## How to Run (sketch)
1. **Install dependencies**  
   ```bash
   pip install transformers peft bitsandbytes accelerate datasets torch
   ```
2. **Prepare raw dataset**  
   - Ensure you have the original CSV file: `mle_screening_dataset.csv`  
   - The file should contain two columns: `question` and `answer`.
3. **Run preprocessing notebook**  
   - Open and execute `Med_Data_Preprocess.ipynb`.  
   - This notebook:
     - Cleans the raw dataset (handles missing values, duplicates, normalization).  
     - Splits the data into **train, validation, and test** sets.  
   -Have 3 csv files processed Hugging Face–compatible datasets ready for training.
4. **Run fine-tuning notebook**  
   - Open and execute `Fine-tuning.ipynb`.  
   - This notebook:
     - Loads the preprocessed datasets (`train`, `validation`, `test`).  
     - Initializes the **MedicalQATrainer** with LoRA + QLoRA configuration.  
     - Trains the model and evaluates periodically.  
5. **Checkpoints and evaluation**  
   - Best checkpoint is restored automatically via `load_best_model_at_end=True`.  
   - Final model and tokenizer are saved in the specified `output_dir`. 
