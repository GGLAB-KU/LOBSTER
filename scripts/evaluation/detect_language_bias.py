"""
Script to find language bias in reviews from CSV annotations file.

This script processes a CSV file containing review annotations and runs bias detection
inference on each review. It uses the paper_id and review_id columns in the CSV to
look up paper/review data from the datasets directory:

  - EMNLP-2023 (Main/Findings): datasets/NLPEERv2-EMNLP-2023/emnlp2023.jsonl
  - EMNLP-ARR-2024:             datasets/ARR-EMNLP-2024-v1.1/emnlp2024.jsonl
  - ACL-ARR-2025:               datasets/ARR-Data-Collection-Initiative-2025/*.jsonl

For each review, it extracts paper title, abstract, and review text, then runs
bias detection inference via an LLM.
Results are saved to a JSONL file.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# ─── CONSTANTS ───────────────────────────
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent.parent  # LOBSTER repo root
ENV_PATH = PROJECT_ROOT / ".env"

# Import llm_providers from project root
sys.path.insert(0, str(PROJECT_ROOT))
from llm_providers import LLMProvider, create_llm_client, BaseLLMClient

MAX_WORKERS = 20
BATCH_RETRY_ROUNDS = 2

# LLM decoding defaults (tuned for ACL-style annotation: deterministic + JSON stability)
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95


# Dataset JSONL files
DATASETS_DIR = PROJECT_ROOT / "datasets"
EMNLP2023_JSONL = DATASETS_DIR / "NLPEERv2-EMNLP-2023" / "emnlp2023.jsonl"
EMNLP2024_JSONL = DATASETS_DIR / "ARR-EMNLP-2024-v1.1" / "emnlp2024.jsonl"
ARR2025_DIR = DATASETS_DIR / "ARR-Data-Collection-Initiative-2025"
ARR2025_JSONL_FILES = [
    ARR2025_DIR / "__dataset_v1.1_acl2025_feb.jsonl",
    ARR2025_DIR / "dataset_v1.1.1_acl2025_dec_feb.jsonl",
]

# Venue mappings (exact names from CSV)
EMNLP2023_VENUES = {"EMNLP-2023-Findings", "EMNLP-2023-Main", "EMNLP2023-workshop-calcs"}
EMNLP_ARR_2024_VENUES = {"EMNLP-ARR-2024"}
ACL_ARR_2025_VENUES = {"ACL-ARR-2025"}


# ─── LOGGER SETUP ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Quiet very chatty third-party loggers so progress bars stay readable.
for _noisy_logger in ("google", "google.api_core", "google.auth", "httpx", "urllib3"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)


# ─── PROGRESS BAR ────────────────────────
def _should_show_progress() -> bool:
    """
    Control whether to show a progress bar.

    - SHOW_PROGRESS=1/true/yes forces it on
    - SHOW_PROGRESS=0/false/no forces it off
    - otherwise, only show if stderr is a TTY
    """
    raw = os.getenv("SHOW_PROGRESS", "").strip().lower()
    if raw in ("1", "true", "yes", "y", "on"):
        return True
    if raw in ("0", "false", "no", "n", "off"):
        return False
    return sys.stderr.isatty()


def _get_tqdm():
    """Return tqdm callable if available, else None (keeps script runnable without tqdm)."""
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm
    except Exception:
        return None


# ─── IO HELPERS ──────────────────────────
def load_jsonl(filename: Path) -> list[dict[str, Any]]:
    """Load JSONL file and return list of records."""
    if not filename.exists():
        logger.warning(f"File not found: {filename}")
        return []
    return [json.loads(line) for line in filename.open(encoding="utf-8") if line.strip()]


def append_jsonl(record: dict[str, Any], filename: Path, lock: threading.Lock | None = None) -> None:
    """Thread-safe append to a JSONL file."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    if lock:
        with lock:
            with filename.open("a", encoding="utf-8") as f:
                f.write(line)
    else:
        with filename.open("a", encoding="utf-8") as f:
            f.write(line)


def load_processed_ids(output_file: Path) -> set[str]:
    """Return a set of already processed review IDs from output_file."""
    if not output_file.exists():
        return set()

    processed_ids = set()
    for line in output_file.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            review_id = record.get("review_id") or record.get("review_link", "")
            venue = record.get("venue", "")
            paper_id = record.get("paper_id", "")
            if review_id:
                processed_ids.add(f"{venue}::{paper_id}::{review_id}")
        except json.JSONDecodeError:
            continue
    return processed_ids


# ─── DATA LOADERS ────────────────────────
def load_emnlp2023_index() -> dict[str, dict[str, Any]]:
    """
    Load EMNLP-2023 data indexed by (paper_id, review_rid).

    Returns dict keyed by paper_id -> {meta, reviews keyed by rid}.
    """
    records = load_jsonl(EMNLP2023_JSONL)
    index: dict[str, dict[str, Any]] = {}

    for record in records:
        pid = str(record.get("paper_id", ""))
        meta = record.get("meta", {})
        reviews_by_rid: dict[str, dict] = {}
        for r in record.get("reviews", []):
            rid = r.get("rid", "")
            if rid:
                reviews_by_rid[rid] = r
        index[pid] = {"meta": meta, "reviews": reviews_by_rid}

    return index


def load_arr_index(jsonl_files: list[Path]) -> dict[str, dict[str, Any]]:
    """
    Load ARR / EMNLP-2024 data indexed by (submission_id, note_id).

    Loads from multiple JSONL files; later files override earlier ones on conflict.
    Returns dict keyed by submission_id -> {submission_meta, reviews keyed by note_id}.
    """
    index: dict[str, dict[str, Any]] = {}

    for fpath in jsonl_files:
        records = load_jsonl(fpath)
        for record in records:
            sid = record.get("submission_id", "")
            if not sid:
                continue
            meta = record.get("submission_meta", {})
            reviews_by_nid: dict[str, dict] = {}
            for r in record.get("reviews", []):
                nid = r.get("note_id", "")
                if nid:
                    reviews_by_nid[nid] = r
            index[sid] = {"submission_meta": meta, "reviews": reviews_by_nid}

    return index


# ─── REVIEW TEXT EXTRACTORS ──────────────
def get_review_text_emnlp2023(review: dict[str, Any], *, strip_summary: bool = False) -> str:
    """Extract full review text from NLPEERv2 EMNLP-2023 format."""
    report = review.get("report", {})
    parts: list[str] = []

    keys = [
        "paper_topic_and_main_contributions",
        "reasons_to_accept",
        "reasons_to_reject",
        "questions_for_the_authors",
        "missing_references",
        "typos_grammar_style_and_presentation_improvements",
        "ethical_concerns",
    ]

    # When strip_summary=True, exclude the paper summary field so the LLM
    # cannot infer paper scope from reviewer-written summaries.
    if strip_summary:
        keys = [k for k in keys if k != "paper_topic_and_main_contributions"]

    for key in keys:
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value.strip()}")

    return "\n\n".join(parts)


def get_review_text_arr(review: dict[str, Any], *, strip_summary: bool = False) -> str:
    """Extract full review text from ARR / EMNLP-2024 format."""
    report = review.get("report", {})
    parts: list[str] = []

    keys = [
        "paper_summary",
        "summary_of_strengths",
        "summary_of_weaknesses",
        "comments_suggestions_and_typos",
        "ethical_concerns",
    ]

    # When strip_summary=True, exclude the paper summary field so the LLM
    # cannot infer paper scope from reviewer-written summaries.
    if strip_summary:
        keys = [k for k in keys if k != "paper_summary"]

    for key in keys:
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value.strip()}")

    return "\n\n".join(parts)


# ─── PROVIDER CONFIG ─────────────────────
def get_llm_provider() -> LLMProvider | None:
    """
    Get the LLM provider from environment or return None for auto-detection.
    
    Set LLM_PROVIDER to "openrouter" or "google_cloud" to force a provider.
    If not set, the provider will be auto-detected based on available credentials.
    """
    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider_name:
        return None  # Auto-detect
    
    provider_map = {
        "openrouter": LLMProvider.OPENROUTER,
        "google_cloud": LLMProvider.GOOGLE_CLOUD,
        "google": LLMProvider.GOOGLE_CLOUD,
        "vertex": LLMProvider.GOOGLE_CLOUD,
        "vertexai": LLMProvider.GOOGLE_CLOUD,
    }
    
    if provider_name in provider_map:
        return provider_map[provider_name]
    
    logger.warning(f"Unknown LLM_PROVIDER={provider_name!r}, using auto-detection")
    return None


# ─── PROMPT BUILDING ─────────────────────
def load_prompt_template(prompt_version: str, prompt_file: str = "review_biases_toward_language.md") -> str:
    """Load the prompt template for a given version."""
    prompt_path = PROJECT_ROOT / "prompts" / prompt_file
    return prompt_path.read_text(encoding="utf-8")


def build_review_biases_prompt(
    template: str,
    paper_title: str,
    paper_abstract: str,
    review_text: str,
) -> str:
    """Construct the LLM prompt for a single review."""
    return (
        template
        .replace("{title}", paper_title)
        .replace("{abstract}", paper_abstract)
        .replace("{review_text}", review_text)
    )


# ─── LLM CALL ────────────────────────────
def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default


def call_llm(
    client: BaseLLMClient,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
    retry_delay: int = 10,
    seed: int = 42,
    timeout: int = 60,
) -> str:
    """
    Call the LLM with retries and return the response text.
    
    Uses the provider-agnostic LLM client interface.
    """
    # Decoding / sampling params (env-overridable)
    temperature = _get_env_float("LLM_TEMPERATURE", DEFAULT_TEMPERATURE)
    top_p = _get_env_float("LLM_TOP_P", DEFAULT_TOP_P)

    return client.call(
        prompt=prompt,
        model_name=model_name,
        max_retries=max_retries,
        retry_delay=retry_delay,
        seed=seed,
        timeout=timeout,
        temperature=temperature,
        top_p=top_p,
    )


# ─── RESPONSE PARSING ────────────────────
def parse_review_biases_response(response_text: str) -> list[dict[str, str]]:
    """
    Parse the LLM JSON response and return a list of all detected biases.
    
    Returns:
        List of bias dictionaries, each containing:
        - quoted_text: The quoted biased statement
        - type: "negative" or "positive"
        - justification: Explanation of why this is bias
        
        Returns empty list if no biases detected.
    """
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        data = json.loads(response_text[json_start:json_end])

        biases = data.get("biases", [])
        if not isinstance(biases, list):
            biases = []

        allowed_types = {"negative", "positive"}
        parsed_biases: list[dict[str, str]] = []

        for item in biases:
            if not isinstance(item, dict):
                continue

            quoted = (item.get("quoted_text") or "").strip()
            bias_type = (item.get("type") or "").strip().lower()
            justification = (item.get("justification") or "").strip()

            if quoted and bias_type in allowed_types and justification:
                parsed_biases.append({
                    "quoted_text": quoted,
                    "type": bias_type,
                    "justification": justification,
                })

        return parsed_biases
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return []


# ─── REVIEW PROCESSING ───────────────────
class ReviewProcessor:
    """Handles processing of individual reviews from CSV rows."""

    def __init__(
        self,
        client: BaseLLMClient,
        model_name: str,
        prompt_template: str,
        prompt_version: str,
        model_tag: str,
        emnlp2023_index: dict[str, dict[str, Any]],
        emnlp2024_index: dict[str, dict[str, Any]],
        arr2025_index: dict[str, dict[str, Any]],
        strip_summary: bool = False,
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.prompt_version = prompt_version
        self.model_tag = model_tag
        self.emnlp2023_index = emnlp2023_index
        self.emnlp2024_index = emnlp2024_index
        self.arr2025_index = arr2025_index
        self.strip_summary = strip_summary

    def _extract_emnlp2023(
        self,
        paper_id: str,
        review_id: str,
    ) -> tuple[str, str, str, str | None]:
        """Extract data for EMNLP-2023 venues. Returns (title, abstract, review_text, error)."""
        paper_data = self.emnlp2023_index.get(paper_id)
        if not paper_data:
            return "", "", "", f"Paper not found in EMNLP-2023 index: {paper_id}"

        meta = paper_data["meta"]
        review = paper_data["reviews"].get(review_id)
        if not review:
            return "", "", "", f"Review {review_id} not found for paper {paper_id}"

        return (
            meta.get("title", ""),
            meta.get("abstract", ""),
            get_review_text_emnlp2023(review, strip_summary=self.strip_summary),
            None,
        )

    def _extract_arr(
        self,
        paper_id: str,
        review_id: str,
        index: dict[str, dict[str, Any]],
        venue_label: str,
    ) -> tuple[str, str, str, str | None]:
        """Extract data for ARR / EMNLP-2024 venues. Returns (title, abstract, review_text, error)."""
        paper_data = index.get(paper_id)
        if not paper_data:
            return "", "", "", f"Paper not found in {venue_label} index: {paper_id}"

        meta = paper_data["submission_meta"]
        review = paper_data["reviews"].get(review_id)
        if not review:
            return "", "", "", f"Review {review_id} not found for paper in {venue_label}"

        paper_abstract = meta.get("abstract", "")
        if not paper_abstract:
            acl_abstract = meta.get("acl_anthology_abstract", "")
            if isinstance(acl_abstract, dict):
                paper_abstract = acl_abstract.get("#text", "")
            elif isinstance(acl_abstract, str):
                paper_abstract = acl_abstract

        return (
            meta.get("title", ""),
            paper_abstract,
            get_review_text_arr(review, strip_summary=self.strip_summary),
            None,
        )

    def process_row(
        self,
        row: dict[str, str],
        output_file: Path,
        lock: threading.Lock,
    ) -> tuple[str, bool, str]:
        """Process a single CSV row: extract data, build prompt, call LLM, save result."""
        venue = row.get("venue", "").strip()
        paper_id = row.get("paper_id", "").strip()
        review_id = row.get("review_id", "").strip()

        if not paper_id or not review_id:
            return review_id, False, "Missing paper_id or review_id"

        # Extract data based on venue
        if venue in EMNLP2023_VENUES:
            title, abstract, review_text, error = self._extract_emnlp2023(paper_id, review_id)
        elif venue in EMNLP_ARR_2024_VENUES:
            title, abstract, review_text, error = self._extract_arr(
                paper_id, review_id, self.emnlp2024_index, "EMNLP-ARR-2024"
            )
        elif venue in ACL_ARR_2025_VENUES:
            title, abstract, review_text, error = self._extract_arr(
                paper_id, review_id, self.arr2025_index, "ACL-ARR-2025"
            )
        else:
            return review_id, False, f"Unsupported venue: {venue}"

        if error:
            return review_id, False, error
        if not review_text:
            return review_id, False, "Empty review text"

        try:
            prompt = build_review_biases_prompt(self.prompt_template, title, abstract, review_text)
            response_text = call_llm(self.client, self.model_name, prompt)
            llm_biases = parse_review_biases_response(response_text)

            record = {
                "venue": venue,
                "paper_id": paper_id,
                "review_id": review_id,
                "llm_biases": llm_biases,
                "prompt_version": self.prompt_version,
                "model": self.model_tag,
            }

            append_jsonl(record, output_file, lock)
            return review_id, True, ""

        except Exception as e:
            # Don't spam logs for each failure; we summarize at the end.
            return review_id, False, f"{type(e).__name__}: {e}"


# ─── BATCH PROCESSING ────────────────────
def run_batch(
    rows: list[dict[str, str]],
    processor: ReviewProcessor,
    output_file: Path,
    already_processed: set[str],
    lock: threading.Lock,
) -> list[dict[str, str]]:
    """Process a batch of rows concurrently, returning failures."""
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures: dict[Any, dict[str, str]] = {}

        for row in rows:
            venue = row.get("venue", "").strip()
            paper_id = row.get("paper_id", "").strip()
            review_id = row.get("review_id", "").strip()
            unique_id = f"{venue}::{paper_id}::{review_id}"

            if unique_id in already_processed:
                continue

            future = executor.submit(processor.process_row, row, output_file, lock)
            futures[future] = {"paper_id": paper_id, "review_id": review_id, "venue": venue}

        total = len(futures)
        if total == 0:
            return failures

        tqdm = _get_tqdm()
        show_bar = _should_show_progress()
        pbar = None
        if tqdm and show_bar:
            pbar = tqdm(total=total, desc="Bias detection", unit="review", dynamic_ncols=True)

        ok = 0
        fail = 0

        for future in as_completed(futures):
            meta = futures[future]
            review_id = meta["review_id"]
            venue = meta["venue"]
            _, success, err = future.result()

            if success:
                ok += 1
            else:
                fail += 1
                failures.append({
                    "venue": venue,
                    "paper_id": meta["paper_id"],
                    "review_id": review_id,
                    "error": err,
                })

            if pbar:
                pbar.update(1)
                pbar.set_postfix_str(f"ok={ok} fail={fail}")

        if pbar:
            pbar.close()

    return failures


# ─── MAIN ────────────────────────────────
def main():
    """Main entry point for the bias detection pipeline."""
    load_dotenv(ENV_PATH)

    # Interactive configuration
    prompt_version = input("Enter prompt version: ").strip()
    if not prompt_version:
        raise ValueError("Prompt version is required")

    prompt_file = input("Enter prompt file [review_biases_toward_language.md]: ").strip()
    if not prompt_file:
        prompt_file = "review_biases_toward_language.md"

    csv_filename = input("Enter CSV filename: ").strip()
    if not csv_filename:
        raise ValueError("CSV filename is required")

    csv_file = FILE_DIR / csv_filename
    if not csv_file.exists():
        # Also try under dataset/annotations/
        csv_file = PROJECT_ROOT / "dataset" / "annotations" / csv_filename
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_filename}")

    # Option to strip paper summaries from review text (for no-context experiments)
    strip_summary_choice = input("Strip paper summaries from review text? (y/n) [n]: ").strip().lower()
    strip_summary = strip_summary_choice in ("y", "yes")
    if strip_summary:
        logger.info("Paper summaries WILL BE STRIPPED from review text")

    # Check if resuming from existing JSONL
    resume_choice = input("Resume from existing JSONL file? (y/n): ").strip().lower()
    existing_jsonl_file: Path | None = None
    
    if resume_choice in ("y", "yes"):
        existing_jsonl_path = input("Enter existing JSONL filename: ").strip()
        if not existing_jsonl_path:
            raise ValueError("JSONL filename is required when resuming")
        existing_jsonl_file = FILE_DIR / existing_jsonl_path
        if not existing_jsonl_file.exists():
            raise FileNotFoundError(f"JSONL file not found: {existing_jsonl_file}")

    # Setup provider and model
    provider = get_llm_provider()
    client = create_llm_client(provider)
    logger.info(f"Using LLM provider: {client.provider_name}")
    
    # Get model name based on the active provider
    model_name = client.get_model_name()
    if not model_name:
        env_var = "GOOGLE_CLOUD_MODEL" if "Google" in client.provider_name else "OPENROUTER_MODEL"
        raise ValueError(f"Model name not set. Set {env_var} in your .env file for {client.provider_name}")

    model_tag = model_name.rsplit("/", 1)[-1]
    
    # Use existing file or create new one
    if existing_jsonl_file:
        output_file = existing_jsonl_file
        logger.info(f"Resuming from existing file: {output_file}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        output_file = PROJECT_ROOT / "dataset" / "evaluation" / "language_bias" / f"bias_{model_tag}_{prompt_version}_{timestamp}.jsonl"

    # Load prompt template
    prompt_template = load_prompt_template(prompt_version, prompt_file)

    # Load CSV
    logger.info("Loading CSV file...")
    with csv_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info(f"Loaded {len(rows)} rows from CSV")

    # Load data indexes from datasets
    logger.info("Loading EMNLP-2023 index...")
    emnlp2023_index = load_emnlp2023_index()
    logger.info(f"Loaded {len(emnlp2023_index)} EMNLP-2023 papers")

    logger.info("Loading EMNLP-ARR-2024 index...")
    emnlp2024_index = load_arr_index([EMNLP2024_JSONL])
    logger.info(f"Loaded {len(emnlp2024_index)} EMNLP-ARR-2024 papers")

    logger.info("Loading ACL-ARR-2025 index...")
    arr2025_index = load_arr_index(ARR2025_JSONL_FILES)
    logger.info(f"Loaded {len(arr2025_index)} ACL-ARR-2025 papers")

    # Initialize processor
    processor = ReviewProcessor(
        client=client,
        model_name=model_name,
        prompt_template=prompt_template,
        prompt_version=prompt_version,
        model_tag=model_tag,
        emnlp2023_index=emnlp2023_index,
        emnlp2024_index=emnlp2024_index,
        arr2025_index=arr2025_index,
        strip_summary=strip_summary,
    )

    # Check for already processed rows
    processed_ids = load_processed_ids(output_file)
    logger.info(f"Found {len(processed_ids)} already processed rows")

    logger.info("=== Pipeline Setup ===")
    logger.info(f"Provider: {client.provider_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Rows to process: {len(rows):,}")
    logger.info(f"Output: {output_file}")
    logger.info("======================")

    lock = threading.Lock()

    # Initial pass
    failures = run_batch(rows, processor, output_file, processed_ids, lock)

    # Retry rounds
    for round_idx in range(1, BATCH_RETRY_ROUNDS + 1):
        if not failures:
            break
        failed_ids = {f"{f['venue']}::{f['paper_id']}::{f['review_id']}" for f in failures}
        logger.info(f"Retry round {round_idx}: attempting {len(failed_ids)} failed rows")
        processed_ids = load_processed_ids(output_file)
        retry_rows = [
            r for r in rows
            if f"{r.get('venue', '')}::{r.get('paper_id', '')}::{r.get('review_id', '')}" in failed_ids
        ]
        failures = run_batch(retry_rows, processor, output_file, processed_ids, lock)

    if failures:
        logger.error(f"{len(failures)} rows ultimately failed:")
        print("\n=== Failed rows ===")
        for f in failures:
            venue = f.get("venue", "")
            pid = f.get("paper_id", "")
            rid = f.get("review_id", "")
            err = f.get("error", "")
            print(f"- {venue}::{pid}::{rid} -> {err}")
        print("\n--- JSON ---")
        print(json.dumps(failures, ensure_ascii=False, indent=2))
    else:
        logger.info("All rows processed successfully.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
