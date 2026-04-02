"""
Detect contribution focus of NLP papers from a JSONL annotations file.

This script processes a JSONL file containing paper metadata (venue, paper_id,
title, abstract) and uses an LLM to classify each paper's contribution type(s)
according to the v23 prompt categories.

Valid v23 labels:
    Modeling, NLPApplications, DataAndBenchmarking, EmpiricalAnalysis,
    LinguisticAnalysis, DomainAdaptation, SurveyOrPosition, Other

Output: JSONL file in bias-detection-results/v23/ with fields:
    venue, paper_id, title, abstract, contribution_type, justification,
    prompt_version, model
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
MAX_REVIEW_LINK_LENGTH = 100

VALID_LABELS = {
    "Modeling",
    "NLPApplications",
    "DataAndBenchmarking",
    "EmpiricalAnalysis",
    "LinguisticAnalysis",
    "DomainAdaptation",
    "SurveyOrPosition",
    "Other",
}

DEFAULT_PROMPT_VERSION = "v23"
DEFAULT_CSV = PROJECT_ROOT / "dataset" / "annotations" / "contribution_type_annotations.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "llm_evaluation" / "contribution_type"


# ─── LOGGER SETUP ────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ─── CLI ARGUMENT PARSING ────────────────
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect contribution focus of papers using LLM inference.",
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
            paper_id = record.get("paper_id", "").strip()
            title = record.get("title", "").strip()
            
            if not paper_id and title:
                paper_id = "missing_" + hashlib.md5((title + record.get("abstract", "")).encode()).hexdigest()[:15]
                
            venue = record.get("venue", "")
            if paper_id:
                processed_ids.add(f"{venue}::{paper_id}")
        except json.JSONDecodeError:
            continue
    return processed_ids


def truncate_id(value: str, max_length: int = MAX_REVIEW_LINK_LENGTH) -> str:
    """Truncate a string ID to max_length if needed."""
    return value[:max_length] if len(value) > max_length else value


# ─── PROMPT BUILDING ─────────────────────
def load_prompt_template(prompt_version: str) -> str:
    """Load the contribution_type prompt template for a given version."""
    prompt_file = PROJECT_ROOT / "prompts" / "contribution_type.md"
    return prompt_file.read_text(encoding="utf-8")


def build_contribution_prompt(
    template: str,
    paper_title: str,
    paper_abstract: str,
) -> str:
    """Construct the LLM prompt for contribution focus detection."""
    return (
        template
        .replace("{title}", paper_title)
        .replace("{abstract}", paper_abstract)
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
def parse_contribution_response(response_text: str) -> tuple[list[str], str]:
    """
    Parse the LLM JSON response and return contribution focus info.

    Validates that all labels are in the v23 label set.

    Returns:
        Tuple of (contribution_type, justification)
    """
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        data = json.loads(response_text[json_start:json_end])

        contribution_type = data.get("contribution_type", [])
        if not isinstance(contribution_type, list):
            contribution_type = [contribution_type] if contribution_type else []
        contribution_type = [str(c).strip() for c in contribution_type if c]

        # Validate labels against v23 categories
        invalid = [l for l in contribution_type if l not in VALID_LABELS]
        if invalid:
            logger.warning(f"Invalid labels {invalid} in response, filtering out")
            contribution_type = [l for l in contribution_type if l in VALID_LABELS]

        justification = (data.get("justification") or "").strip()

        return contribution_type, justification
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return [], ""


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
    ):
        self.client = client
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.prompt_version = prompt_version
        self.model_tag = model_tag

    def process_row(
        self,
        row: dict[str, str],
        output_file: Path,
        lock: threading.Lock,
    ) -> tuple[str, bool, str]:
        """Process a single annotation row: build prompt, call LLM, save result."""
        venue = row.get("venue", "").strip()
        paper_id = row.get("paper_id", "").strip()
        title = row.get("title", "").strip()
        abstract = row.get("abstract", "").strip()

        if not paper_id:
            paper_id = "missing_" + hashlib.md5((title + abstract).encode()).hexdigest()[:15]

        if not title and not abstract:
            return paper_id, False, "No title or abstract available"

        try:
            prompt = build_contribution_prompt(self.prompt_template, title, abstract)
            response_text = call_llm(self.client, self.model_name, prompt)
            contribution_type, justification = parse_contribution_response(response_text)

            record = {
                "venue": venue,
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "contribution_type": contribution_type,
                "justification": justification,
                "prompt_version": self.prompt_version,
                "model": self.model_tag,
            }

            append_jsonl(record, output_file, lock)
            focus_str = ", ".join(contribution_type) if contribution_type else "none"
            logger.info(f"✔ Processed {venue}::{paper_id[:50]} - {focus_str}")
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
            title = row.get("title", "").strip()
            abstract = row.get("abstract", "").strip()
            
            if not paper_id:
                paper_id = "missing_" + hashlib.md5((title + abstract).encode()).hexdigest()[:15]
                
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


# ─── MAIN ────────────────────────────────
def main():
    """Main entry point for the contribution focus detection pipeline."""
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
        output_file = DEFAULT_OUTPUT_DIR / f"contribution_type_{model_tag}_{prompt_version}_{timestamp}.jsonl"

    # Load prompt template
    prompt_template = load_prompt_template(prompt_version)

    # Load annotations (JSONL)
    logger.info("Loading annotation file...")
    with csv_file.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    logger.info(f"Loaded {len(rows)} rows")

    # Initialize processor
    processor = PaperProcessor(
        client=client,
        model_name=model_name,
        prompt_template=prompt_template,
        prompt_version=prompt_version,
        model_tag=model_tag,
    )

    # Check for already processed rows
    processed_ids = load_processed_ids(output_file)
    logger.info(f"Found {len(processed_ids)} already processed rows")

    logger.info("=== Pipeline Setup ===")
    logger.info(f"Provider: {client.provider_name}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Prompt version: {prompt_version}")
    logger.info(f"Valid labels: {sorted(VALID_LABELS)}")
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
