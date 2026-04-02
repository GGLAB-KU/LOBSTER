# 🦞 LOBSTER — Language-Of-study Bias in ScienTific pEer Review

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.07119-b31b1b.svg)](https://arxiv.org/abs/2604.07119)

This repository accompanies the paper:

> **[Are Non-English Papers Reviewed Fairly? Language-of-Study Bias in NLP Peer Reviews](https://arxiv.org/abs/2604.07119)**

**LOBSTER** is the first human-annotated dataset for detecting **language-of-study (LoS) bias** in NLP peer reviews — the tendency for reviewers to evaluate a paper differently based on the language(s) it studies, rather than its scientific merit.

## Overview

Language-of-study bias manifests as:

- **Negative bias** — devaluing or dismissing research because of the language(s) studied (e.g., *"It could be better if the authors would experiment with English datasets to further demonstrate its effectiveness"*).
- **Positive bias** — overly praising the use of certain languages without engaging with methodology (e.g., *"More work in low-resource languages is always good"*).

We present the first systematic characterization of LoS bias, introduce LOBSTER with **534 expert-annotated review segments**, and benchmark six state-of-the-art LLMs for automatic detection — with the best model (**Gemini 3.1 Pro**) achieving **87.37 Macro F1**.

Our large-scale analysis of **15,645 reviews** across six NLP venues reveals that **non-English papers face bias rates roughly 40× higher** than English-only ones, with negative bias consistently outweighing positive bias.

## Dataset Statistics

### Annotation Layers

| Layer | Records | Description |
|:---|---:|:---|
| Language Bias | 534 | Bias labels for review segments (529 with annotator consensus, 5 where annotation was not possible due to demanding deeper topic expertise) |
| Contribution Type | 100 | Paper contribution categories |
| Language of Study | 100 | Languages studied by the paper |

### Corpus Coverage

| Venue | Papers | Reviews | Annotated Segments |
|:---|---:|---:|---:|
| EMNLP 2023 | 2,020 | 6,449 | 375 |
| EMNLP 2024 | 1,063 | 1,425 | 103 |
| ACL 2025 (Dec–Feb) | 2,187 | 3,756 | 56 |
| ARR 2024 (Apr–Jun) | 464 | 499 | — |
| COLING/NAACL 2025 | 410 | 498 | — |
| EMNLP 2025 (Jun–Aug) | 1,762 | 3,018 | — |
| **Total** | **7,906** | **15,645** | **534** |

Review sources: [NLPEERv2](https://tudatalib.ulb.tu-darmstadt.de/items/d4a4061b-e4e3-4b1e-a90d-d48a3d69e3c0) (EMNLP 2023/2024), [ARR Data Collection Initiative](https://tudatalib.ulb.tu-darmstadt.de/items/4266a71b-1d5c-40bf-8923-7beec1c5263e) (remaining venues).

### Bias Label Distribution (n=534)

| Label | Count |
|:---|---:|
| No Bias Detected | 439 |
| Negative Bias | 73 |
| Positive Bias | 17 |
| Unclear / Needs Context | 4 |
| No Majority | 1 |

> **Note:** The 5 segments labeled *Unclear / Needs Context* or *No Majority* could not be annotated due to demanding deeper topic expertise. They are included in the dataset for transparency but excluded from LLM evaluation. The `final_label` field for these records reflects the original unresolved label.

### LLM Benchmark (3-way classification, n=529)

| Model | Macro F1 | Weighted F1 |
|:---|---:|---:|
| **Gemini 3.1 Pro** | **87.37** | **93.60** |
| Grok 4.1 Fast | 79.75 | 90.96 |
| GPT 5.2 | 78.29 | 90.77 |
| Claude Opus 4.6 | 74.96 | 88.91 |
| DeepSeek V3.2 | 66.89 | 81.75 |
| Llama 4 Maverick 17B | 63.94 | 79.00 |
| Random baseline | 33.33 | 70.85 |
| Majority baseline | 30.23 | 75.27 |

## Repository Structure

```
LOBSTER/
├── README.md                          # This file
├── LICENSE                            # CC BY-NC 4.0
├── requirements.txt                   # Python dependencies
├── .env.example                       # Template for LLM credentials
├── annotation_guideline.md            # Complete annotation protocol
│
├── llm_providers/                     # LLM provider abstraction (Google Cloud, OpenRouter)
│
├── dataset/
│   ├── annotationSchema.md            # JSONL schema documentation
│   │
│   ├── annotations.zip                # Password-protected ZIP (pwd: lobster)
│   │   ├── language_bias_annotations.jsonl
│   │   ├── contribution_type_annotations.jsonl
│   │   └── language_of_study_annotations.jsonl
│   │
│   ├── llm_evaluation.zip             # Password-protected ZIP (pwd: lobster)
│   │   ├── language_bias/             # 6 LLMs + ablation runs
│   │   ├── contribution_type/         # Gemini 3.1 Pro
│   │   └── language_of_study/         # Gemini 3.1 Pro
│   │
│   └── llm_predictions.zip            # Password-protected ZIP (pwd: lobster)
│       ├── acl2025/
│       ├── arr2024_apr_jun/
│       ├── coling_naacl2025/
│       ├── emnlp2023/
│       ├── emnlp2024/
│       ├── emnlp2025/
│       └── negative_bias_subcategories/
│
├── scripts/
│   ├── README.md                      # Reproduction instructions & parameters
│   ├── llm_evaluation/                # Prompt evaluation scripts
│   │   ├── detect_language_bias.py
│   │   ├── detect_language_of_study.py
│   │   ├── detect_contribution_type.py
│   │   └── calculate_baseline.py
│   └── llm_predictions/               # Full-corpus inference scripts
│       ├── run_bias_detection.py
│       ├── run_language_detection.py
│       └── run_contribution_type.py
│
└── prompts/                           # LLM prompt templates
    ├── review_biases_toward_language.md
    ├── languages_of_study.md
    ├── contribution_type.md
    └── negative_bias_subcategory.md
```

## Quick Start

> **Important**: To protect reviewer privacy and prevent the reviews from appearing in GitHub search results, the `.jsonl` data files are compressed into password-protected ZIP archives. You must provide the password **`lobster`** to interact with them.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure LLM credentials (only needed for running scripts)
cp .env.example .env
# Edit .env with your API keys

# 3. Extract the annotations
cd dataset && unzip -P lobster annotations.zip && cd ..
```

```python
import json

# Load the bias annotations
with open("dataset/annotations/language_bias_annotations.jsonl") as f:
    bias_data = [json.loads(line) for line in f]

print(f"Loaded {len(bias_data)} annotated review segments")

# Get the gold label for each segment
for record in bias_data:
    label = record["final_label"]
    # label ∈ {"Negative Bias", "Positive Bias", "No Bias Detected", ...}

# Filter for biased segments
biased = [r for r in bias_data if r["final_label"] in ("Negative Bias", "Positive Bias")]
print(f"Found {len(biased)} biased segments")

# Check individual annotator votes
if biased:
    sample_votes = biased[0]["votes"] 
    # e.g., ["neg_bias", "neg_bias", "not_bias"]
    print(f"Sample annotator votes: {sample_votes}")
```

## Corpus Compilation

The raw review corpus used for the large-scale analysis (**15,645 reviews**) cannot be redistributed with LOBSTER. To reproduce the full-corpus predictions, you must download the source datasets from their original hosts:

| Source | Venues Covered | Download |
|:---|:---|:---|
| **NLPEERv2** | EMNLP 2023, EMNLP 2024 | [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/items/d4a4061b-e4e3-4b1e-a90d-d48a3d69e3c0) |
| **ARR Data Collection Initiative** | ACL 2025, ARR 2024, COLING/NAACL 2025, EMNLP 2025 | [TUdatalib](https://tudatalib.ulb.tu-darmstadt.de/items/4266a71b-1d5c-40bf-8923-7beec1c5263e) |

After downloading, place the JSONL files following the directory structure described in [`scripts/README.md`](scripts/README.md#corpus-compilation). Once in place, you can run the prediction scripts to reproduce the full-corpus bias detection seamlessly.

## Tasks

LOBSTER supports three classification tasks:

1. **Bias Classification** (main task) — Classify a review segment as *Negative Bias*, *Positive Bias*, or *No Bias Detected*, given the paper title, abstract, and the review segment. Multi-class, evaluated with Macro F1.

2. **Contribution Type Classification** — Categorize each paper's contribution focus (e.g., Modeling, NLPApplications, DataAndBenchmarking). Multi-label, evaluated on 100 annotated papers.

3. **Language-of-Study Detection** — Determine the linguistic scope of each paper using a six-category taxonomy (single-language, multilingual-specified, etc.). Multi-label, evaluated on 100 annotated papers.

## Data Format

### Annotations (Gold-Standard)

Human-annotated data is in `dataset/annotations/` (after extracting `annotations.zip`) as **JSONL** (one JSON object per line). See [`dataset/annotationSchema.md`](dataset/annotationSchema.md) for detailed schema documentation.

### LLM Evaluation

LLM outputs on the annotation set (used for prompt evaluation and model benchmarking) are in `dataset/llm_evaluation/` (after extracting `llm_evaluation.zip`), organized by task.

### LLM Predictions

Full-corpus model predictions are in `dataset/llm_predictions/` (after extracting `llm_predictions.zip`), organized by venue. Each venue directory contains:
- `bias_results_*.jsonl` — Bias detection predictions
- `contrib_results_*.jsonl` — Contribution type predictions
- `lang_results_*.jsonl` — Language of study predictions

Subcategory predictions for negatively biased reviews are in `llm_predictions/negative_bias_subcategories/`.

## Scripts

See [`scripts/README.md`](scripts/README.md) for detailed reproduction instructions, including:
- Exact LLM parameters used in the paper (model, temperature, top-p, seed)
- How to run evaluation and prediction scripts
- Corpus compilation guide

## Annotation Guideline

See [`annotation_guideline.md`](annotation_guideline.md) for the complete annotation protocol, including definitions of bias categories, decision rules for borderline cases, and illustrative examples.

## Key Findings

- Non-English papers face **bias rates ~40× higher** than English-only papers.
- **Negative bias consistently outweighs positive bias** across all venues.
- **Positive bias is concentrated in multilingual papers**: ~39% of biased reviews for specified multilingual papers are positively biased, compared to 31% for single non-English and just 4% for English. A recurring pattern of positive bias is observed for low-resource languages (e.g., Marathi, Vietnamese, Indonesian, Swahili), suggesting reviewers may treat diverse language coverage as a merit in itself — decoupled from the quality of the methodology.
- Four subcategories of negative bias identified, with **unjustified cross-lingual generalization demands** being the most dominant form.
- Bias patterns are structural, not isolated — they persist across all six venues examined.

## Ethical Considerations

This study examines bias in peer review, a topic that inherently involves sensitive judgments about reviewer behavior. All analyses are based on publicly available review data from OpenReview. We do not identify individual reviewers and use password-protected archives to reduce the risk of review content appearing in web search results.

## Acknowledgments

We thank Dr. Ali Hürriyetoğlu and Tamta Kapanadze for their help with the annotation effort, including participation in adjudication discussions and guideline refinement. Part of this work was initiated by Dagstuhl Seminar 25301 "Linguistics and language models: What can they learn from each other?". Marie-Catherine de Marneffe is a research associate of the Fonds de la Recherche Scientifique – FNRS. Finally, this research is with support from Google.org and the Google Cloud Research Credits program for the Gemini Academic Program.

## License

This project (code and data) is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

## Citation

If you use LOBSTER in your research, please cite:

```bibtex
@article{barkhordar2026lobster,
  title     = {Are Non-English Papers Reviewed Fairly? {Language-of-Study} Bias in {NLP} Peer Reviews},
  author    = {Barkhordar, Ehsan and Safa, Abdulfattah and Blaschke, Verena and Lombart, Erika and de Marneffe, Marie-Catherine and {\c{S}}ahin, G{\"o}zde G{\"u}l},
  journal   = {arXiv preprint arXiv:2604.07119},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.07119}
}
```
