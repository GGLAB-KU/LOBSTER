# Annotation Schema

The LOBSTER gold-standard human annotations are stored as clean JSONL files with only final, adjudicated labels.

> [!IMPORTANT]
> To protect reviewer privacy and prevent these files from appearing in web searches, the `.jsonl` files have been compressed into the password-protected archive `annotations.zip`.
> **Password**: `lobster`
> 
> Please extract the archive (e.g., `unzip -P lobster annotations.zip`) before running your analyses or experiments.

---

## JSONL Schema

### `language_bias_annotations.jsonl`

Each line is one annotated review segment (534 total; 529 with annotator consensus, 5 where annotation was not possible due to demanding deeper topic expertise).

| Field | Type | Description |
|:---|:---|:---|
| `venue` | string | Conference venue (e.g., `EMNLP-2023-Main`) |
| `paper_id` | string | Paper identifier |
| `review_id` | string | Review identifier |
| `openreview_link` | string | URL to the review on OpenReview |
| `title` | string | Paper title |
| `abstract` | string | Paper abstract |
| `review_text` | string | Full review text |
| `bias_quoted_text` | string | The specific segment being annotated |
| `final_label` | string | Final decided label (combining adjudicated decision and majority vote) |
| `annotator_count` | int | Number of annotators who labeled this segment |
| `votes` | list[string] | Array of individual annotator votes cast for this segment |

**Bias labels:** `Negative Bias`, `Positive Bias`, `No Bias Detected`, `Unclear / Needs Context`, `No Majority`

---

### `contribution_type_annotations.jsonl`

Each line is one paper with its contribution type (100 total).

| Field | Type | Description |
|:---|:---|:---|
| `venue` | string | Conference venue |
| `paper_id` | string | Paper identifier |
| `title` | string | Paper title |
| `abstract` | string | Paper abstract |
| `contribution_type` | string or list | One or more contribution categories |

**Contribution types:** `Modeling`, `NLPApplications`, `DataAndBenchmarking`, `EmpiricalAnalysis`, `LinguisticAnalysis`, `DomainAdaptation`, `SurveyOrPosition`

---

### `language_of_study_annotations.jsonl`

Each line is one paper with its language-of-study classification (100 total).

| Field | Type | Description |
|:---|:---|:---|
| `venue` | string | Conference venue |
| `paper_id` | string | Paper identifier |
| `title` | string | Paper title |
| `abstract` | string | Paper abstract |
| `language_of_study` | object | Structured language classification |

**`language_of_study` object:**

| Sub-field | Type | Description |
|:---|:---|:---|
| `language_scope` | string | `single-language`, `multilingual-specified`, `multilingual-partial`, `multilingual-count-only`, `multilingual-unspecified`, `language-agnostic` |
| `languages` | list[string] | Specific languages (e.g., `["Chinese", "English"]`) |
| `languages_count` | int | Number of languages |
| `evidence_type` | string | `explicit_list`, `dataset_implied`, `count_only`, `claim_only` |

---

See the [`README.md`](../README.md) for the annotation guideline and full project overview.
