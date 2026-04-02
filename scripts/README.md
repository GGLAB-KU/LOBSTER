# Scripts

This directory contains the scripts used in the LOBSTER paper for LLM-based bias detection.

## Overview

| Directory | Purpose | Input Data |
|:---|:---|:---|
| `llm_evaluation/` | Benchmark LLMs on the annotated gold-standard segments | LOBSTER annotations (`dataset/annotations.zip`) |
| `llm_predictions/` | Run bias/language/contribution detection on the full review corpus | Raw review corpus (see [Corpus Compilation](#corpus-compilation)) |

Both categories use **the same prompts** (from `prompts/`) and **the same LLM configuration**. They differ only in the input data:
- **LLM Evaluation** runs on the 529 consensus-labeled segments to measure model performance.
- **LLM Predictions** runs on the full corpus of 15,645 reviews across six venues to produce the large-scale analysis.

## Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your LLM credentials
cp .env.example .env
# Edit .env with your API keys

# 3. Extract the annotations
cd dataset && unzip -P lobster annotations.zip && cd ..
```

## Paper Parameters

The following exact parameters were used to produce all results in the paper:

| Parameter | Value |
|:---|:---|
| **LLM Model** | `gemini-3.1-pro-preview` (Gemini 3.1 Pro via Vertex AI) |
| **Temperature** | `0.0` (deterministic) |
| **Top-p** | `0.95` |
| **Seed** | `42` |
| **Prompt version** | `v23` |
| **Max retries** | `3` |
| **Retry delay** | `10s` |

## LLM Evaluation Scripts

These scripts evaluate LLM performance on the LOBSTER gold-standard annotations. They work directly with the provided dataset after extracting `annotations.zip`.

> **Note:** The bias detection script additionally requires the raw review datasets (NLPEERv2, ARR) to reconstruct full review text from the annotation CSV. See [Corpus Compilation](#corpus-compilation).

### Bias Detection

```bash
# Using paper defaults (v23 prompt, auto-detects annotation file)
python scripts/llm_evaluation/detect_language_bias.py

# With explicit parameters
python scripts/llm_evaluation/detect_language_bias.py \
    --prompt-version v23 \
    --csv dataset/annotations/language_bias_annotations.jsonl \
    --datasets-dir /path/to/language-bias-peer-review/datasets
```

### Contribution Type Detection

```bash
python scripts/llm_evaluation/detect_contribution_type.py \
    --prompt-version v23 \
    --csv dataset/annotations/contribution_type_annotations.jsonl
```

### Language-of-Study Detection

```bash
python scripts/llm_evaluation/detect_language_of_study.py \
    --prompt-version v23 \
    --csv dataset/annotations/language_of_study_annotations.jsonl
```

### Baseline Calculation

```bash
python scripts/llm_evaluation/calculate_baseline.py
```

## LLM Prediction Scripts

These scripts run inference on the full review corpus. **They require the raw review data** (see [Corpus Compilation](#corpus-compilation)) and the companion `language-bias-peer-review` repository.

```bash
# Clone the companion repo alongside LOBSTER
git clone https://github.com/GGLAB-KU/language-bias-peer-review.git

# Run predictions
python scripts/llm_predictions/run_bias_detection.py
python scripts/llm_predictions/run_contribution_type.py
python scripts/llm_predictions/run_language_detection.py
```

## Corpus Compilation

The raw review data cannot be redistributed with LOBSTER. To reproduce the full-corpus analysis:

### Step 1: Download Source Data

| Source | Venues Covered | Download URL |
|:---|:---|:---|
| **NLPEERv2** | EMNLP 2023, EMNLP 2024 | [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/items/d4a4061b-e4e3-4b1e-a90d-d48a3d69e3c0) |
| **ARR Data Collection Initiative** | ACL 2025, ARR 2024, COLING/NAACL 2025, EMNLP 2025 | [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/items/4266a71b-1d5c-40bf-8923-7beec1c5263e) |

### Step 2: Organize Files

Place the downloaded JSONL files into `datasets/` inside the LOBSTER repository:

```
LOBSTER/datasets/
├── NLPEERv2-EMNLP-2023/
│   └── emnlp2023.jsonl
├── ARR-EMNLP-2024-v1.1/
│   └── emnlp2024.jsonl
└── ARR-Data-Collection-Initiative-2025/
    ├── __dataset_v1.1_acl2025_feb.jsonl
    ├── dataset_v1.1.1_acl2025_dec_feb.jsonl
    ├── dataset_v1_coling2025_naacl2025.jsonl
    ├── dataset_v1.2_arr2024_apr_jun.jsonl
    └── dataset_v1.3_emnlp2025.jsonl
```

### Step 3: Run Scripts

Evaluation scripts use `datasets/` by default:

```bash
python scripts/llm_evaluation/detect_language_bias.py

# Or point to a custom location:
python scripts/llm_evaluation/detect_language_bias.py \
    --datasets-dir /path/to/datasets
```

For prediction scripts, ensure `language-bias-peer-review` is cloned alongside LOBSTER (they auto-detect it).
