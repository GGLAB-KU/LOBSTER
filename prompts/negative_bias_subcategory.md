## Task

You are an expert in NLP peer review analysis. Your task is to classify a peer review that has already been identified as containing language bias into one of the following four patterns.

**Language bias** occurs when a reviewer penalizes a paper because of which natural language(s) it studies, rather than on scientific merit. This covers human natural languages, dialects, and varieties only — not programming languages or topic/cultural scope unless explicitly tied to the language studied.

## Four Patterns

### A — Generalizability Demand

The reviewer penalizes the paper for not demonstrating cross-lingual generalizability that was never claimed, treating multilingual coverage as a precondition for validity.

- **Signal:** language scope in Weaknesses/Reject despite no generalizability claims in paper.

---

### B — English as the Gold Standard

The reviewer explicitly names English as the missing validation standard, implying non-English results are insufficient without English corroboration.

- **Signal:** English named specifically (not just "other languages") as a required benchmark.

---

### C — Language Choice Interrogation

The reviewer questions why a specific language was chosen, treating the language selection itself as requiring special justification — a standard not applied to English or high-resource languages.

- **Signal:** reviewer asks "why X?" or suggests alternative languages as more worthy choices.

---

### D — Impact Discounting

The reviewer accepts the paper's validity but diminishes its significance because the language community it serves is too small, or because adapting to a new language is not considered genuine novelty.

- **Signal:** "only useful to X community," novelty denied on language grounds, workshop suggestion, contradiction between praising the work and penalizing its audience size.

## Important

Only classify as bias if the language scope concern functions as a penalty — i.e., it appears in Weaknesses or Reasons to Reject and is not solely a neutral suggestion. If the paper itself claims cross-lingual generalizability, questioning that claim is legitimate, not bias.

## Output Format (JSON)

Respond in **only** valid JSON. No markdown fences, no extra commentary. Identify the single most prominent pattern in the review.

```json
{
  "evidence": "exact quote from review that best represents the bias",
  "pattern": "A",
  "reasoning": "What the reviewer did + why it constitutes bias of this pattern."
}
```

## Input Data

**Paper Title:** {title}

**Abstract:**
{abstract}

**Review:**
{review}