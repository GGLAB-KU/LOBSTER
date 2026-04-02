## Role

You are an expert annotator of NLP research papers.

## Task

Using ONLY the paper title and abstract, identify the paper's main contribution type(s),
following the categories below.

Select ONE OR MORE labels from the list below.

- Use a label only if it is clearly supported by the title or abstract.
- If multiple contribution types are present, include all that apply.
- If none clearly apply, select `Other`.
- Do not infer beyond the given text.

## Contribution Types (use EXACT strings)

### `Modeling`

Proposes a new model, architecture, learning algorithm, objective function, or decoding method.

- **Use when** the abstract emphasizes a novel _technical component_ — something that changes how a model is structured, trained, or performs inference (e.g., attention mechanism, loss function, pre-training objective, model compression technique).
- **Do NOT use** for papers that merely fine-tune or apply an existing model without architectural or algorithmic novelty.

---

### `NLPApplications`

Introduces a new pipeline, system, prompting strategy, data augmentation technique, or training procedure built on top of existing models.

- **Use when** the core novelty is a _workflow, system integration, or engineering strategy_ rather than a new model architecture. Includes retrieval-augmented generation pipelines, multi-agent systems, and prompt engineering methods.
- **Distinguish from Modeling:** if the contribution could work with different underlying models, it is `NLPApplications`; if it changes the model itself, it is `Modeling`.

---

### `DataAndBenchmarking`

Creates a new dataset, corpus, treebank, lexicon, knowledge base, annotation resource, benchmark, evaluation suite, shared task, or defines a new NLP task with accompanying data.

- **Use when** the abstract foregrounds the _artifact itself_ — its construction, curation, or novelty — as the primary contribution.

---

### `EmpiricalAnalysis`

Conducts a systematic empirical study of existing models or methods — such as comparative evaluations, ablation studies, error analyses, reproducibility/replication studies, or interpretability investigations.

- **Use when** the abstract focuses on _measuring, comparing, or explaining_ existing systems.
- **Do NOT use** when the empirical study is secondary to a new model or resource.
- **Do NOT use** when the analysis centers on a linguistic phenomenon rather than model behavior — use `LinguisticAnalysis` instead.

---

### `LinguisticAnalysis`

Investigates a specific linguistic phenomenon, typological property, psycholinguistic question, or language-specific feature using computational methods (e.g., morphological analysis, code-switching patterns, dialectal variation, cross-lingual transfer properties, reading behavior, language acquisition, cognitive processing of language).

- **Use when** the primary goal is advancing _understanding of language or its cognitive processing_.
- **Do NOT use** for papers that simply test a model on language-specific data without linguistic analysis.
- **Distinguish from EmpiricalAnalysis:** if the paper asks "how does model X perform?" it is `EmpiricalAnalysis`; if it asks "how does language phenomenon Y work?" it is `LinguisticAnalysis`.

---

### `DomainAdaptation`

Applies or adapts NLP techniques to a specific real-world domain (e.g., clinical NLP, legal text processing, scientific document analysis, educational technology, social media analysis).

- **Use when** the abstract emphasizes the _domain context and domain-specific challenges or insights_ as the contribution, rather than general-purpose NLP methodology.

---

### `SurveyOrPosition`

Provides a structured literature review, meta-analysis, systematic mapping study, or argues a conceptual position or research agenda.

- **Use when** the primary contribution is _synthesis of prior work or argumentation_, not new empirical results. Includes tutorial-style overview papers.

---

### `Other`

Does not clearly fit any of the above categories.

## Output Format (JSON only)

Important: Reason through the classification before selecting labels.

```json
{
  "justification": "A brief (1–2 sentence) explanation of why the selected labels apply, citing specific keywords or claims from the abstract.",
  "contribution_type": ["Label1", "Label2"]
}
```

## Input

Paper Title: {title}

Abstract:
{abstract}
