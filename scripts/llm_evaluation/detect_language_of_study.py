"""
Detect languages studied in NLP papers from a JSONL annotations file.

This script processes a JSONL file containing paper metadata (venue, paper_id,
openreview_link, title, abstract) and uses an LLM to classify the natural languages studied in
each paper according to the v23 prompt categories.

It handles multiple venue types:
  - OpenReview URLs (EMNLP-2023 venues)
  - TUdatalib review IDs (ACL-ARR-2025)
  - EMNLP-ARR-2024 review tokens

For each paper, it extracts title, abstract, and ALL reviews for that paper,
then runs language detection inference using the languages_of_study.md prompt.

Valid v23 language_scope values:
    single-language, multilingual-specified, multilingual-partial,
    multilingual-count-only, multilingual-unspecified, language-agnostic

Output: JSONL file in results/language_of_study/v23/ with fields:
    venue, paper_id, title, abstract, language_scope, languages,
    languages_count, evidence_type, justification, prompt_version, model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

# Import llm_providers from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from llm_providers import LLMProvider, create_llm_client, BaseLLMClient


# ─── CONSTANTS ───────────────────────────
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent.parent  # LOBSTER repo root
ENV_PATH = PROJECT_ROOT / ".env"

MAX_WORKERS = 20  # concurrent API requests; balances throughput vs rate limits
BATCH_RETRY_ROUNDS = 2
TOKEN_LENGTH = 35  # max chars for review token column in TUdatalib data
MAX_ID_LENGTH = 32  # max chars for paper/review ID column
MAX_REVIEW_LINK_LENGTH = 100  # max chars for OpenReview URL column

VALID_LANGUAGE_SCOPES = {
    "single-language",
    "multilingual-specified",
    "multilingual-partial",
    "multilingual-count-only",
    "multilingual-unspecified",
    "language-agnostic",
}

VALID_EVIDENCE_TYPES = {
    "explicit_list",
    "dataset_implied",
    "count_only",
    "claim_only",
}

DEFAULT_PROMPT_VERSION = "v23"
DEFAULT_CSV = PROJECT_ROOT / "dataset" / "annotations" / "language_of_study_annotations.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "llm_evaluation" / "language_of_study"

# Data directories
OPENREVIEW_DIR = PROJECT_ROOT / "openreview"
EMNLP2023_FINDINGS_DIR = OPENREVIEW_DIR / "emnlp2023_findings"
EMNLP2023_MAIN_DIR = OPENREVIEW_DIR / "emnlp2023_main"
EMNLP2023_WORKSHOP_DIR = OPENREVIEW_DIR / "emnlp2023_workshop_calcs"
TUDATALIB_DIR = PROJECT_ROOT / "ACL-ARR-2025"
TUDATALIB_REVIEWS_FILE = TUDATALIB_DIR / "metadata_reviews.jsonl"
ARR_EMNLP_2024_DIR = PROJECT_ROOT / "EMNLP-ARR-2024"
ARR_EMNLP_2024_PAPERS_REVIEWS_FILE = ARR_EMNLP_2024_DIR / "papers_and_reviews_final.jsonl"

# Venue mappings
EMNLP2023_VENUES = {"EMNLP2023 findings", "EMNLP2023 main", "EMNLP2023 workshop calcs", "EMNLP-2023-Main", "EMNLP-2023-Findings"}
ACL_ARR_2025_VENUES = {"ACL ARR 2025", "ACL-ARR-2025"}


# ─── LOGGER SETUP ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


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
            paper_id = record.get("paper_id") or ""
            title = record.get("title") or ""
            if not paper_id and title:
                paper_id = "missing_" + hashlib.md5((title + record.get("abstract", "")).encode()).hexdigest()[:15]
            venue = record.get("venue", "")
            if paper_id:
                processed_ids.add(f"{venue}::{paper_id}")
        except json.JSONDecodeError:
            continue
    return processed_ids


def truncate_id(value: str, max_length: int = MAX_ID_LENGTH) -> str:
    """Truncate a string ID to max_length if needed."""
    return value[:max_length] if len(value) > max_length else value


# ─── URL PARSING ─────────────────────────
def parse_openreview_url(url: str) -> tuple[str, str] | None:
    """
    Parse OpenReview URL to extract paper_id and note_id.
    
    Returns (paper_id, note_id) or None if parsing fails.
    """
    try:
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        paper_id = query_params.get("id", [None])[0]
        note_id = query_params.get("noteId", [None])[0]

        if paper_id:
            return paper_id, note_id or ""
    except Exception as e:
        logger.debug(f"Failed to parse URL {url}: {e}")
    return None


def normalize_review_token(review_link: str) -> str:
    """
    Normalize a review_link into a short review token (typically 35 chars).
    """
    token = review_link.strip()
    if not token:
        return token

    if token.startswith("http"):
        parsed = parse_openreview_url(token)
        if parsed:
            paper_id, note_id = parsed
            if note_id:
                token = note_id
            else:
                token = paper_id

    return token[:TOKEN_LENGTH] if len(token) > TOKEN_LENGTH else token


# ─── VENUE HELPERS ───────────────────────
def get_venue_dir(venue_name: str) -> Path | None:
    """Get the directory path for a venue."""
    venue_mapping = {
        "EMNLP2023 findings": EMNLP2023_FINDINGS_DIR,
        "EMNLP2023 main": EMNLP2023_MAIN_DIR,
        "EMNLP2023 workshop": EMNLP2023_WORKSHOP_DIR,
        "EMNLP2023 workshop_calcs": EMNLP2023_WORKSHOP_DIR,
        "EMNLP2023 workshop calcs": EMNLP2023_WORKSHOP_DIR,
        "EMNLP-2023-Main": EMNLP2023_MAIN_DIR,
        "EMNLP-2023-Findings": EMNLP2023_FINDINGS_DIR,
    }

    # Try exact match
    if venue_name in venue_mapping:
        return venue_mapping[venue_name]

    # Try case-insensitive match
    for key, path in venue_mapping.items():
        if key.lower() == venue_name.lower():
            return path

    # Try partial match
    venue_lower = venue_name.lower()
    if "findings" in venue_lower:
        return EMNLP2023_FINDINGS_DIR
    if "workshop" in venue_lower or "calcs" in venue_lower:
        return EMNLP2023_WORKSHOP_DIR
    if "main" in venue_lower:
        return EMNLP2023_MAIN_DIR

    return None


# ─── DATA LOADERS ────────────────────────
def load_venue_data(venue_dir: Path) -> tuple[dict[str, dict], dict[str, dict], dict[str, list[dict]]]:
    """Load papers and reviews for a venue directory.
    
    Returns:
        Tuple of (paper_map, review_map, paper_to_reviews) where:
        - paper_map: paper_id -> paper dict
        - review_map: review_id -> review dict
        - paper_to_reviews: paper_id -> list of all reviews for that paper
    """
    reviews = load_jsonl(venue_dir / "official_reviews.jsonl")
    papers = load_jsonl(venue_dir / "papers.jsonl")

    review_map = {r["id"]: r for r in reviews if r.get("id")}
    paper_map = {p["id"]: p for p in papers if p.get("id")}
    
    # Build paper_id -> list of reviews mapping
    paper_to_reviews: dict[str, list[dict]] = {}
    for r in reviews:
        paper_id = r.get("paper_id") or r.get("forum")
        if paper_id:
            paper_to_reviews.setdefault(paper_id, []).append(r)

    return paper_map, review_map, paper_to_reviews


def load_tudatalib_data() -> dict[str, list[dict[str, Any]]]:
    """
    Load TUdatalib reviews indexed by IDs and 35-char prefixes.
    """
    reviews = load_jsonl(TUDATALIB_REVIEWS_FILE)
    index: dict[str, list[dict[str, Any]]] = {}

    def add_key(key: str, record: dict[str, Any]) -> None:
        key = (key or "").strip()
        if key:
            index.setdefault(key, []).append(record)
            if len(key) >= TOKEN_LENGTH:
                index.setdefault(key[:TOKEN_LENGTH], []).append(record)

    for record in reviews:
        submission_id = (record.get("submission_id") or "").strip()
        if submission_id:
            add_key(submission_id, record)

        for review in record.get("reviews", []):
            if not isinstance(review, dict):
                continue
            for key in (review.get("note_id"), review.get("rid")):
                if key:
                    add_key(key.strip(), record)

    return index


def load_arr_emnlp_2024_index() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Load EMNLP-ARR-2024 data indexed by review token."""
    records = load_jsonl(ARR_EMNLP_2024_PAPERS_REVIEWS_FILE)
    index: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    for paper_record in records:
        for review in paper_record.get("reviews", []):
            if not isinstance(review, dict):
                continue
            token = (review.get("note_id") or review.get("rid") or "").strip()
            if token:
                index[token] = (paper_record, review)

    return index


def find_review_in_tudatalib(
    review_id: str,
    tudatalib_index: dict[str, list[dict]],
) -> tuple[dict, dict] | None:
    """
    Find review and paper in TUdatalib data by review_id (usually a 35-char prefix).
    
    Returns (paper_record, review_dict) or None if not found.
    """
    records = tudatalib_index.get(review_id)
    if not records:
        return None

    record = records[0]

    for review in record.get("reviews", []):
        if not isinstance(review, dict):
            continue
        note_id = (review.get("note_id") or "").strip()
        rid = (review.get("rid") or "").strip()
        if note_id.startswith(review_id) or rid.startswith(review_id):
            return record, review

    # Fallback: return first review if nothing matched exactly
    if record.get("reviews"):
        return record, record["reviews"][0]

    return None


# ─── REVIEW TEXT EXTRACTORS ──────────────
def get_review_text_openreview(review: dict[str, Any]) -> str:
    """Extract full review text from OpenReview format."""
    content = review.get("content") or {}
    parts: list[str] = []

    for key, value in content.items():
        if isinstance(value, dict) and "value" in value:
            val = value.get("value")
        else:
            val = value
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}: {val.strip()}")

    return "\n\n".join(parts)


def get_review_text_tudatalib(review: dict[str, Any]) -> str:
    """Extract full review text from TUdatalib format."""
    report = review.get("report", {})
    parts: list[str] = []

    keys = [
        "paper_summary",
        "summary_of_strengths",
        "summary_of_weaknesses",
        "comments_suggestions_and_typos",
        "ethical_concerns",
    ]

    for key in keys:
        value = report.get(key)
        if isinstance(value, str) and value.strip():
            label = key.replace("_", " ").title()
            parts.append(f"{label}: {value.strip()}")

    return "\n\n".join(parts)


# ─── PROMPT BUILDING ─────────────────────
def load_prompt_template(prompt_version: str) -> str:
    """Load the prompt template for a given version."""
    prompt_file = PROJECT_ROOT / "prompts" / "languages_of_study.md"
    return prompt_file.read_text(encoding="utf-8")


def build_languages_prompt(
    template: str,
    paper_title: str,
    paper_abstract: str,
    reviews_text: str,
) -> str:
    """Construct the LLM prompt for language detection."""
    return (
        template
        .replace("{title}", paper_title)
        .replace("{abstract}", paper_abstract)
        .replace("{reviews_text}", reviews_text)
    )


# ─── LLM CALL ────────────────────────────
def call_llm(
    client: BaseLLMClient,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
    retry_delay: int = 10,
) -> str:
    """
    Call the LLM with retries and return the response text.
    
    Uses the provider-agnostic LLM client interface.
    """
    return client.call(
        prompt=prompt,
        model_name=model_name,
        max_retries=max_retries,
        retry_delay=retry_delay,
        seed=42,
        timeout=60,
        temperature=0.0,
        top_p=0.95,
    )


# ─── RESPONSE PARSING ────────────────────
def parse_languages_response(response_text: str) -> tuple[str, list[str], int, str, str]:
    """
    Parse the LLM JSON response and return languages info.
    
    Returns:
        Tuple of (language_scope, languages, languages_count, evidence_type, justification)
    """
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        data = json.loads(response_text[json_start:json_end])

        language_scope = (data.get("language_scope") or "").strip()
        languages = data.get("languages", [])
        if not isinstance(languages, list):
            languages = []
        languages = [str(lang).strip() for lang in languages if lang]
        languages_count = int(data.get("languages_count", len(languages)))
        evidence_type = (data.get("evidence_type") or "").strip()
        justification = (data.get("justification") or "").strip()

        # Validate language_scope
        if language_scope and language_scope not in VALID_LANGUAGE_SCOPES:
            logger.warning(f"Invalid language_scope '{language_scope}', leaving as-is")

        # Validate evidence_type
        if evidence_type and evidence_type not in VALID_EVIDENCE_TYPES:
            logger.warning(f"Invalid evidence_type '{evidence_type}', leaving as-is")

        return language_scope, languages, languages_count, evidence_type, justification
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return "", [], 0, "", ""


# ─── PAPER PROCESSING ───────────────────
class PaperProcessor:
    """Handles processing of individual papers from annotation rows."""

    def __init__(
        self,
        client: BaseLLMClient,
        model_name: str,
        prompt_template: str,
        prompt_version: str,
        model_tag: str,
        venue_papers: dict[str, dict[str, dict]],
        venue_reviews: dict[str, dict[str, dict]],
        venue_paper_reviews: dict[str, dict[str, list[dict]]],
        tudatalib_index: dict[str, list[dict]],
        arr_emnlp_index: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.prompt_version = prompt_version
        self.model_tag = model_tag
        self.venue_papers = venue_papers
        self.venue_reviews = venue_reviews
        self.venue_paper_reviews = venue_paper_reviews
        self.tudatalib_index = tudatalib_index
        self.arr_emnlp_index = arr_emnlp_index

    def _load_venue_if_needed(self, venue_dir: Path) -> tuple[dict, dict, dict]:
        """Load venue data if not already cached."""
        venue_key = venue_dir.name
        if venue_key not in self.venue_papers:
            paper_map, review_map, paper_to_reviews = load_venue_data(venue_dir)
            self.venue_papers[venue_key] = paper_map
            self.venue_reviews[venue_key] = review_map
            self.venue_paper_reviews[venue_key] = paper_to_reviews
        return self.venue_papers[venue_key], self.venue_reviews[venue_key], self.venue_paper_reviews[venue_key]

    def _extract_emnlp2023(
        self,
        review_link: str,
        venue: str,
        csv_title: str,
        csv_abstract: str,
    ) -> tuple[str, str, str, str, str | None]:
        """Extract data for EMNLP2023 venues. Returns ALL reviews for the paper."""
        parsed = parse_openreview_url(review_link)
        if not parsed:
            return "", "", "", "", f"Failed to parse OpenReview URL: {review_link}"

        paper_id, note_id = parsed
        venue_dir = get_venue_dir(venue)
        if not venue_dir:
            return "", "", "", "", f"Unknown EMNLP2023 venue: {venue}"

        paper_map, review_map, paper_to_reviews = self._load_venue_if_needed(venue_dir)
        paper = paper_map.get(paper_id)
        
        # Get ALL reviews for this paper
        reviews_text_parts = []
        all_reviews = paper_to_reviews.get(paper_id, [])
        for i, review in enumerate(all_reviews, 1):
            review_text = get_review_text_openreview(review)
            if review_text:
                reviews_text_parts.append(f"=== Review {i} ===\n{review_text}")
        
        reviews_text = "\n\n".join(reviews_text_parts)

        # Use CSV data if available, otherwise fall back to loaded data
        paper_title = csv_title if csv_title else (paper.get("title", "") if paper else "")
        paper_abstract = csv_abstract if csv_abstract else (paper.get("abstract", "") if paper else "")

        if not paper_title and not paper:
            return "", "", "", "", f"Paper not found: {paper_id}"

        return (
            paper_title,
            paper_abstract,
            reviews_text,
            paper_id,
            None,
        )

    def _extract_emnlp_arr_2024(
        self,
        review_link: str,
        csv_title: str,
        csv_abstract: str,
    ) -> tuple[str, str, str, str, str | None]:
        """Extract data for EMNLP-ARR-2024 venue. Returns ALL reviews for the paper."""
        token = normalize_review_token(review_link)
        match = self.arr_emnlp_index.get(token)

        if not match:
            # If no match but we have CSV data, use it
            if csv_title:
                return csv_title, csv_abstract, "", token, None
            return "", "", "", "", f"Review token not found in EMNLP-ARR-2024: {token}"

        paper_record, _ = match
        paper_id = (paper_record.get("paper_id") or "").strip()
        meta = paper_record.get("meta") or {}

        # Use CSV data if available
        paper_title = csv_title if csv_title else meta.get("title", "")
        paper_abstract = csv_abstract if csv_abstract else meta.get("abstract", "")

        # Get ALL reviews for this paper
        reviews_text_parts = []
        all_reviews = paper_record.get("reviews", [])
        for i, review in enumerate(all_reviews, 1):
            if isinstance(review, dict):
                review_text = get_review_text_tudatalib(review)
                if review_text:
                    reviews_text_parts.append(f"=== Review {i} ===\n{review_text}")
        
        reviews_text = "\n\n".join(reviews_text_parts)

        return (
            paper_title,
            paper_abstract,
            reviews_text,
            paper_id,
            None,
        )

    def _extract_acl_arr_2025(
        self,
        review_link: str,
        csv_title: str,
        csv_abstract: str,
    ) -> tuple[str, str, str, str, str | None]:
        """Extract data for ACL ARR 2025 venue. Returns ALL reviews for the paper."""
        token = normalize_review_token(review_link)
        result = find_review_in_tudatalib(token, self.tudatalib_index)

        if not result:
            # If no match but we have CSV data, use it
            if csv_title:
                return csv_title, csv_abstract, "", token, None
            return "", "", "", "", f"Review not found in TUdatalib: {token}"

        paper_record, _ = result
        paper_id = (paper_record.get("submission_id") or "")[:MAX_ID_LENGTH]
        submission_meta = paper_record.get("submission_meta") or {}

        # Use CSV data if available
        paper_title = csv_title if csv_title else submission_meta.get("title", "")
        paper_abstract = csv_abstract if csv_abstract else submission_meta.get("abstract", "")

        if not paper_abstract:
            acl_abstract = submission_meta.get("acl_anthology_abstract", "")
            if isinstance(acl_abstract, dict):
                paper_abstract = acl_abstract.get("#text", "")
            elif isinstance(acl_abstract, str):
                paper_abstract = acl_abstract

        # Get ALL reviews for this paper
        reviews_text_parts = []
        all_reviews = paper_record.get("reviews", [])
        for i, review in enumerate(all_reviews, 1):
            if isinstance(review, dict):
                review_text = get_review_text_tudatalib(review)
                if review_text:
                    reviews_text_parts.append(f"=== Review {i} ===\n{review_text}")
        
        reviews_text = "\n\n".join(reviews_text_parts)

        return (
            paper_title,
            paper_abstract,
            reviews_text,
            paper_id,
            None,
        )

    def process_row(
        self,
        row: dict[str, str],
        output_file: Path,
        lock: threading.Lock,
    ) -> tuple[str, bool, str]:
        """Process a single annotation row: extract data, build prompt, call LLM, save result."""
        venue = row.get("venue", "").strip()
        paper_id = row.get("paper_id", "").strip()
        openreview_link = row.get("openreview_link", "").strip()
        csv_title = row.get("title", "").strip()
        csv_abstract = row.get("abstract", "").strip()

        if not paper_id and csv_title:
            paper_id = "missing_" + hashlib.md5((csv_title + csv_abstract).encode()).hexdigest()[:15]

        if not paper_id and not openreview_link:
            return "", False, "Missing paper_id and openreview_link"

        # Extract data based on venue
        # For EMNLP-2023 venues, use the openreview_link for data lookup
        # For ARR venues, use the paper_id (submission_id) prefix
        if venue in EMNLP2023_VENUES or "EMNLP-2023" in venue or "EMNLP2023" in venue:
            if openreview_link:
                title, abstract, reviews_text, extracted_id, error = self._extract_emnlp2023(
                    openreview_link, venue, csv_title, csv_abstract
                )
            elif csv_title:
                # Fallback: use CSV data directly when no openreview_link
                title, abstract, reviews_text, extracted_id, error = csv_title, csv_abstract, "", paper_id, None
                logger.info(f"No openreview_link for {venue}::{paper_id}, using title/abstract only")
            else:
                title, abstract, reviews_text, extracted_id, error = "", "", "", "", "Missing openreview_link and title"
        elif venue == "EMNLP-ARR-2024":
            # Use paper_id prefix as the review token for data lookup
            title, abstract, reviews_text, extracted_id, error = self._extract_emnlp_arr_2024(
                paper_id[:35] if paper_id else "", csv_title, csv_abstract
            )
        elif venue in ACL_ARR_2025_VENUES:
            title, abstract, reviews_text, extracted_id, error = self._extract_acl_arr_2025(
                paper_id[:35] if paper_id else "", csv_title, csv_abstract
            )
        else:
            # Default: try to use CSV data directly
            title = csv_title
            abstract = csv_abstract
            reviews_text = ""
            extracted_id = ""
            error = None

        if error:
            return paper_id, False, error
        if not title and not abstract:
            return paper_id, False, "No title or abstract available"

        try:
            prompt = build_languages_prompt(self.prompt_template, title, abstract, reviews_text)
            response_text = call_llm(self.client, self.model_name, prompt)
            language_scope, languages, languages_count, evidence_type, justification = parse_languages_response(response_text)

            record = {
                "venue": venue,
                "paper_id": paper_id,
                "openreview_link": openreview_link,
                "title": title,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "language_scope": language_scope,
                "languages": languages,
                "languages_count": languages_count,
                "evidence_type": evidence_type,
                "justification": justification,
                "prompt_version": self.prompt_version,
                "model": self.model_tag,
            }

            append_jsonl(record, output_file, lock)
            languages_str = ", ".join(languages) if languages else "none"
            logger.info(f"✔ Processed {venue}::{paper_id[:50]} - {language_scope}: {languages_str} ({languages_count})")
            return paper_id, True, ""

        except Exception as e:
            logger.error(f"❌ Failed to process row {paper_id}: {e}")
            return paper_id, False, str(e)


# ─── BATCH PROCESSING ────────────────────
def run_batch(
    rows: list[dict[str, str]],
    processor: PaperProcessor,
    output_file: Path,
    already_processed: set[str],
    lock: threading.Lock,
) -> list[dict[str, str]]:
    """Process a batch of rows concurrently, returning failures."""
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}

        for row in rows:
            venue = row.get("venue", "").strip()
            paper_id = row.get("paper_id", "").strip()
            
            if not paper_id and row.get("title", "").strip():
                paper_id = "missing_" + hashlib.md5((row.get("title", "").strip() + row.get("abstract", "").strip()).encode()).hexdigest()[:15]
                
            unique_id = f"{venue}::{paper_id}"

            if unique_id in already_processed:
                continue

            future = executor.submit(processor.process_row, row, output_file, lock)
            futures[future] = paper_id

        for future in as_completed(futures):
            paper_id = futures[future]
            _, success, err = future.result()
            if not success:
                logger.warning(f"Row {paper_id} failed: {err}")
                failures.append({"paper_id": paper_id, "error": err})

    return failures


# ─── CLI ARGUMENT PARSING ────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect languages of study in papers using LLM inference.",
    )
    parser.add_argument(
        "--prompt-version",
        default=DEFAULT_PROMPT_VERSION,
        help=f"Prompt version to use (default: {DEFAULT_PROMPT_VERSION})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Input JSONL file (default: {DEFAULT_CSV.relative_to(FILE_DIR)})",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from an existing JSONL output file",
    )
    return parser.parse_args()


# ─── MAIN ────────────────────────────────
def main():
    """Main entry point for the language detection pipeline."""
    load_dotenv(ENV_PATH)
    args = parse_args()

    prompt_version = args.prompt_version
    csv_file = args.csv

    if not csv_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {csv_file}")

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

    # Determine output file
    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(f"Resume file not found: {args.resume}")
        output_file = args.resume
        logger.info(f"Resuming from existing file: {output_file}")
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
        output_file = DEFAULT_OUTPUT_DIR / f"language_of_study_{model_tag}_{prompt_version}_{timestamp}.jsonl"

    # Load prompt template
    prompt_template = load_prompt_template(prompt_version)

    # Load annotations (JSONL)
    logger.info("Loading annotation file...")
    with csv_file.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(rows)} rows")

    # Load data indexes
    logger.info("Loading TUdatalib data...")
    tudatalib_index = load_tudatalib_data()
    logger.info(f"Loaded {len(tudatalib_index)} TUdatalib index entries")

    logger.info("Loading EMNLP-ARR-2024 data...")
    arr_emnlp_index = load_arr_emnlp_2024_index()
    logger.info(f"Loaded {len(arr_emnlp_index)} EMNLP-ARR-2024 review entries")

    # Initialize processor
    processor = PaperProcessor(
        client=client,
        model_name=model_name,
        prompt_template=prompt_template,
        prompt_version=prompt_version,
        model_tag=model_tag,
        venue_papers={},
        venue_reviews={},
        venue_paper_reviews={},
        tudatalib_index=tudatalib_index,
        arr_emnlp_index=arr_emnlp_index,
    )

    # Check for already processed rows
    processed_ids = load_processed_ids(output_file)
    logger.info(f"Found {len(processed_ids)} already processed rows")

    logger.info("=== Pipeline Setup ===")
    logger.info(f"Provider: {client.provider_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt version: {prompt_version}")
    logger.info(f"Valid scopes: {sorted(VALID_LANGUAGE_SCOPES)}")
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
        failed_links = {f["paper_id"] for f in failures}
        logger.info(f"Retry round {round_idx}: attempting {len(failed_links)} failed rows")
        processed_ids = load_processed_ids(output_file)
        retry_rows = [r for r in rows if r.get("paper_id", "").strip() in failed_links]
        failures = run_batch(retry_rows, processor, output_file, processed_ids, lock)

    if failures:
        logger.error(f"{len(failures)} rows ultimately failed:")
        print(json.dumps(failures, ensure_ascii=False, indent=2))
    else:
        logger.info("All rows processed successfully.")

    logger.info("Done.")


if __name__ == "__main__":
    main()
