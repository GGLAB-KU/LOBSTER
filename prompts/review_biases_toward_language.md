## Task

You are an expert in NLP peer review analysis. Identify **language bias** — cases where a reviewer evaluates a paper differently because of which natural language(s) it studies, rather than on scientific merit.

**Language** = human natural languages, dialects, varieties, and sign languages. Excludes programming languages and constructed languages (Esperanto, Klingon).

**This task is NOT about** cultural/geographic topic scope (e.g., "American political context"), nor general domain/topic niche-ness unless explicitly tied to the language(s) studied.

## Decision Process

In your internal reasoning, first read the title and abstract carefully, then analyze the full review in context before making any judgment. For each reviewer comment about language scope, apply these two tests in order:

**Step 1 — Paper Scope.** Read the title and abstract to determine the paper's language scope:

- **Scoped**: Paper is explicitly limited to specific language(s) and makes no claims of multilingual, cross-lingual, or language-independent generalizability.
- **Claims generalizability**: Paper claims broad applicability ("language-agnostic", "multilingual", "cross-lingual", etc.) or does not clearly limit its scope.

If the paper **claims generalizability** or **does not limit its scope** → reviewer comments about other languages or cross-lingual performance are **valid scientific feedback**, not bias. Stop here.

If the paper is **scoped** → proceed to Step 2.

**Step 2 — Review Section & Impact.** Where does the comment appear, and does it affect the reviewer's decision?

- In **Weaknesses / Reasons to Reject / Major Concerns**, or used to justify a low score → **strong bias signal**. Read the surrounding context carefully to confirm bias before flagging.
- In **Questions / Suggestions / Future Work** only, without being tied to the rejection decision or downgrading the current contribution → **not bias**. Do not flag.
- In **Strengths / Reasons to Accept**, as the primary/sole justification for acceptance → **positive bias signal**. Read the surrounding context carefully to confirm bias before flagging.

## Two Types of Language Bias

### Negative Bias – Flag if the reviewer:

- Treats English (or another high-resource language) as a required validity check, even though the paper is scoped to non-English language(s).
- Insists the work is incomplete or unconvincing without English evaluation or popular English benchmarks (e.g., GLUE, SuperGLUE).
- Demands multilingual or cross-lingual experiments as a prerequisite for acceptance when the paper never claims generalizability.
- Questions the paper's applicability/generalizability primarily because experiments are on a single language, when the paper is clearly scoped.
- Downplays the paper's impact, relevance, or venue fit because it focuses on a low-resource, non-English, or lesser-studied language (e.g., "few researchers will care", "too niche", "limited audience", "better suited for a workshop").
- States the contribution is small/weak because the language(s) are perceived as minor, obscure, or not "major."
- Questions or rejects the motivation for the language choice as if the language is unworthy of study (e.g., "why this obscure language?", "why not a more widely studied language?").
- Frames limited language scope as a critical flaw when broad generalization was never claimed.

### Positive Bias – Flag if the reviewer:

Positive bias should be **rare**. The core pattern is: the reviewer praises the paper's language choice as a merit **in itself**, disconnected from the paper's actual methods, results, or novelty.

- The language choice appears as a **standalone reason** in Strengths/Accept, not connected to specific methods or results (e.g., "The creation of a new dataset in a non-English language.", "The research here is on the Chinese language.").
- The praise is **generic and unconditional** — it could apply to any paper in that language. Look for: "always valuable", "in itself", "the real contribution is the dataset for [language]."
- The language is framed as the **primary justification** for the paper's value rather than its methodology or findings.

## Do NOT Flag (Valid Critique)

- Criticism of dataset size, data quality, annotation process, baselines, ablations, reproducibility, etc. — regardless of the language studied.
- Requests for additional language experiments when the paper **claims** language-independence, cross-lingual transfer, or broad applicability.
- Suggestions phrased as optional improvements ("it would be nice to see…", "future work could…") without affecting the accept/reject decision.
- Requests for methodological rigor (e.g., asking why a particular baseline is used) without implying the language is unworthy.
- Comments about accessibility (e.g., adding English translations for readability) framed as optional improvements.

## Annotation Rules

- Quote the **full sentence** containing the biased statement (not just a clause).
- If a sentence mixes bias with valid critique, still quote the **entire sentence**.
- Output every distinct biased claim separately, even if from the same reviewer.
- Annotate bias in the reviewer text only, never in the paper itself.

## Output Format (JSON)

Return **only** valid JSON. No markdown fences, no extra commentary. If no bias: `{"biases": []}`.

```json
{
  "biases": [
    {
      "quoted_text": "<biased statement from the review>",
      "type": "<negative | positive>",
      "justification": "<1–3 concise sentences explaining why this is language bias>"
    }
  ]
}
```

## Input Data

**Paper Title:** {title}

**Abstract:**
{abstract}

**Review Text:**
{review_text}
