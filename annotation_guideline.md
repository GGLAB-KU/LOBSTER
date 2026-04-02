# **🧭 Peer Review Bias Annotation Guideline**

**Version:** 1.3 — November 2025  
**Maintainer:** Ehsan Barkhordar

---

## **1\. Definitions**

### **1.1 Definition of Bias**

**Bias** is a *systematic and unfair deviation from impartial judgment*, often caused by irrelevant preferences or assumptions.  
 *(Oxford Dictionary of Psychology; Evans 2005\)*

---

### **1.2 Definition of Bias Toward/Against the Language(s) Studied in Peer Reviews**

This annotation task identifies **bias** in NLP peer reviews **toward/against the language(s) studied in the paper reviewed**. Annotators flag reviewer statements where judgments are influenced *because of the language(s) the paper studies*, not by scientific merit.

We annotate four categories:

---

#### **Negative Bias**

A reviewer *devalues or dismisses* the research because of the language(s) studied, or assumes a particular language (often English) that is not central to the study, is the **default, superior, or required standard** for demonstrating validity or importance.

**Example:**

*“The proposed approach was solely evaluated on three Chinese dialogue datasets… It could be better if the authors would experiment with English dialogue datasets to further demonstrate its effectiveness.”*

*For more info please visit [review url](https://openreview.net/forum?id=OVt2dIwxR1&noteId=gxoxnqRYHP)*  
This implies that **English evaluation is necessary** for meaningful results.

---

#### **Positive Bias**

A reviewer *overly praises* the use of certain languages — high-resource, low-resource, or multilingual — **without engaging with methodology, analysis, or contribution**.

**Example:**

*“More work in low-resource languages is always good.”*

*For more info please visit [review url](https://openreview.net/forum?id=L8Cxea5krb&noteId=RyNUhzLMo9)*  
 Praise is based solely on language choice, not scientific merit.

---

#### **No Bias Detected (Valid Critique)**

Scientifically grounded comments aligned with the paper’s scope.

Examples:

* *“Limited experimentation with non-English languages.”* *(valid if the paper claims multilingual generalization)*

* *“How does Bangla word analogy compare with English?”* *(valid if relevant to the paper’s goals)*  
* *“Prompts contain ungrammatical English that may or may not affect experimental results.”*  
* *“This paper explored a task on cross-cultural social norm discovery between English and Chinese, two main languages around the world.” (it's just stating facts)*  
* *“This paper provides a genre analysis of abstracts of research articles in the NLP field and some pieces of advice to make their usage of English closer to native speakers' one.”*

These are **not** biased toward/against the language(s) studied.

This category also includes criticisms that are unrelated to the language(s) studied, like reviewers complaining about the quality of the writing style / asking the authors to let a native English speaker proofread their text.

---

#### **Unclear / Needs Context**

Used only when the reviewer’s intent **cannot be judged** from the abstract \+ review alone.

Examples:

* *“Testing on a limited number of languages or training sets is not sufficient to support the claims”* *(claims are not clear here)*

---

## **2\. Goal of the Project**

Build a **high-quality, human-verified dataset** for evaluating and refining LLM-based detection of **bias toward the language(s) studied** in NLP peer reviews.

---

## **3 What Information Is Provided in the Spreadsheet**

Each row in the annotation spreadsheet corresponds to **one candidate fragment** extracted from a review. To support consistent human judgments, the spreadsheet includes:

### **3.1 Paper and venue metadata**

* **venue**: The conference/track (e.g., EMNLP-2023-Main).

* **title**: The paper title.

* **abstract**: The paper abstract (included because many bias judgments depend on whether the reviewer’s language-related expectation is **inside** or **outside** the paper’s stated scope).

### **3.2 Review access and context**

* **review\_link**: A link to the full OpenReview page for the submission (useful if more surrounding context is needed).

* **review\_text**: The full review text (provided to check tone/section and verify whether the quoted fragment is being interpreted fairly).

### **3.3 Candidate fragment to label**

* **bias\_quoted\_text**: The exact fragment that annotators should label (this is the primary text to annotate).

### **3.4 Model outputs (assistive only; do not follow blindly)**

These columns are outputs from an automatic bias-detection model and are included only to help triage and organize candidates. They can be wrong and should not override human judgment.

* **model\_bias\_inference**: The model’s predicted label for the fragment (e.g., negative / positive / none / unclear).  
   **Important:** This is a *suggestion*, not ground truth. Annotators must label based on the reviewer’s wording and the guideline definitions.

* **bias\_justification**: The model’s brief explanation for why it produced its prediction.  
   **Important:** This can be misleading or irrelevant. Use it only if it helps interpret a very short fragment; do not rely on it as evidence.

### **3.5 What Human Annotators Can Use**

Annotators may consult:

* **Full paper PDF**

* **Author rebuttals**

* **Metadata** (languages studied, track, contribution type)

These sources help resolve ambiguous cases but **should not be used to reinterpret the review**.

---

## **4\. Annotation Options**

Annotators choose **one** label per fragment:

| Label | Meaning |
| :---- | :---- |
| **Negative Bias** | Unfair dismissal due to language(s) studied |
| **Positive Bias** | Uncritical praise focused on language choice |
| **No Bias Detected** | Neutral or valid critique aligned with scope |
| **Unclear / Needs Context** | Ambiguous; requires full paper/rebuttal (use rarely\!) |

**Most fragments should fall into the first three categories.**

---

## **5\. Note**

* The LLM’s `bias_justification` is generally not very useful, but may help clarify intent in short or ambiguous cases. Labels must be based on **the reviewer’s own wording** and **independent human judgment**, not model reasoning.

* Always consider the **section and tone** of the review: neutral phrases may indicate **negative bias** in rejection sections, and unexplained praise may indicate **positive bias** in strength sections.  
* Focus on bias within the **review text**, not the paper.

* A single review may contain multiple biases.

* When unsure:

  * Prefer **No Bias Detected** if the critique is clearly methodological.

  * Use **Unclear / Needs Context** if judgment truly requires full paper or rebuttal.

* Additional spreadsheet instructions (with screenshots) will be added separately.

---

## **6\. What to Annotate**

For each row, read the text in the **bias\_quoted\_text** column and, based on the **Definitions** section, select the appropriate label from the **Annotation Options** in **Section 4**.

If necessary, add a short explanation in the **annotation notes** column (there is a dedicated column for this purpose). Notes should briefly clarify why you selected that label (e.g., what in the wording makes it biased, or why it is a valid critique).

Notes are optional, but recommended for:

* borderline or ambiguous cases  
* disagreements with the majority decision  
* fragments that may require later adjudication

Model columns (e.g., **model\_bias\_inference**, **bias\_justification**) are provided for reference only and must not determine the final annotation.


