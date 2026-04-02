"""
Shared base module for review analysis scripts.

Provides common infrastructure for run_bias_detection.py,
run_contribution_type.py, and run_language_detection.py:
  - Venue definitions and data loading
  - JSONL I/O helpers
  - LLM client setup and calling
  - User input helpers
  - Batch processing with ThreadPoolExecutor
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


from dotenv import load_dotenv

# ─── PATHS & CONFIG ───────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

from llm_providers import LLMProvider, create_llm_client, BaseLLMClient

# Batch processing defaults
MAX_WORKERS = 40
BATCH_RETRY_ROUNDS = 2

# LLM decoding defaults
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95

# Logger
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("base_runner")


# ─── VENUE DEFINITIONS ─────────────────────
@dataclass
class VenueConfig:
    """Configuration for a data venue."""
    name: str
    short_name: str
    data_file: Path
    data_format: str  # "arr" or "nlpeerv2"


DATASETS_DIR = PROJECT_ROOT / "datasets"
PROMPTS_DIR = PROJECT_ROOT / "prompts"

# Prompts are unversioned in this repo; this is the label the paper uses and
# what gets written into output filenames and result records.
PROMPT_VERSION = "v23"

VENUES = {
    "1": VenueConfig(
        name="EMNLP 2023 (NLPEERv2)",
        short_name="emnlp2023",
        data_file=DATASETS_DIR / "NLPEERv2-EMNLP-2023" / "emnlp2023.jsonl",
        data_format="nlpeerv2",
    ),
    "2": VenueConfig(
        name="EMNLP 2024 (ARR)",
        short_name="emnlp2024",
        data_file=DATASETS_DIR / "ARR-EMNLP-2024-v1.1" / "emnlp2024.jsonl",
        data_format="arr",
    ),
    "3": VenueConfig(
        name="COLING/NAACL 2025",
        short_name="coling_naacl2025",
        data_file=DATASETS_DIR / "ARR-Data-Collection-Initiative-2025" / "dataset_v1_coling2025_naacl2025.jsonl",
        data_format="arr",
    ),
    "4": VenueConfig(
        name="ACL 2025",
        short_name="acl2025",
        data_file=DATASETS_DIR / "ARR-Data-Collection-Initiative-2025" / "dataset_v1.1.1_acl2025_dec_feb.jsonl",
        data_format="arr",
    ),
    "5": VenueConfig(
        name="ARR 2024 Apr/Jun",
        short_name="arr2024_apr_jun",
        data_file=DATASETS_DIR / "ARR-Data-Collection-Initiative-2025" / "dataset_v1.2_arr2024_apr_jun.jsonl",
        data_format="arr",
    ),
    "6": VenueConfig(
        name="EMNLP 2025",
        short_name="emnlp2025",
        data_file=DATASETS_DIR / "ARR-Data-Collection-Initiative-2025" / "dataset_v1.3_emnlp2025.jsonl",
        data_format="arr",
    ),
}


# ─── IO HELPERS ───────────────────────────
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


def load_processed_ids(output_file: Path, id_key: str = "review_id") -> set[str]:
    """Return a set of already processed IDs from output_file."""
    if not output_file.exists():
        return set()

    processed_ids = set()
    for line in output_file.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            rid = record.get(id_key)
            if rid:
                processed_ids.add(rid)
        except json.JSONDecodeError:
            continue
    return processed_ids


# ─── TEXT EXTRACTORS ──────────────────────
def get_review_text(review: dict[str, Any]) -> str:
    """Extract full review text from any format (ARR or NLPEERv2).

    Works generically by iterating all keys in review.report and review.scores.
    """
    report = review.get("report") or {}
    parts: list[str] = []

    if isinstance(report, dict):
        for key, value in report.items():
            if not value:
                continue
            if isinstance(value, str):
                text = value.strip()
            elif isinstance(value, list):
                text = "\n".join(
                    item.strip() for item in value if isinstance(item, str) and item.strip()
                )
            else:
                text = str(value)
            if text:
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {text}")

    scores = review.get("scores")
    if isinstance(scores, dict) and scores:
        score_text = ", ".join(f"{k}: {v}" for k, v in scores.items())
        parts.append(f"Scores: {score_text}")

    return "\n\n".join(parts)


# ─── DATA LOADERS ─────────────────────────
@dataclass
class PaperRecord:
    """Unified paper record from any data source."""
    paper_id: str
    title: str
    abstract: str
    reviews: list[dict[str, Any]]


def _extract_paper_record(record: dict[str, Any]) -> PaperRecord | None:
    """Extract a PaperRecord from a JSONL record (any format)."""
    # ARR Initiative 2025 format
    if "submission_id" in record:
        paper_id = record["submission_id"]
        meta = record.get("submission_meta") or {}
    # ARR EMNLP 2024 / NLPEERv2 format
    elif "paper_id" in record:
        paper_id = record["paper_id"]
        meta = record.get("meta") or {}
    else:
        return None

    title = meta.get("title", "")
    abstract = meta.get("abstract", "")

    if not title:
        return None

    reviews = record.get("reviews") or []

    return PaperRecord(
        paper_id=str(paper_id),
        title=title,
        abstract=abstract,
        reviews=reviews,
    )


def load_paper_records(venue: VenueConfig) -> list[PaperRecord]:
    """Load all paper records from a venue's JSONL file."""
    raw = load_jsonl(venue.data_file)
    records = []
    for item in raw:
        pr = _extract_paper_record(item)
        if pr:
            records.append(pr)
    return records


# ─── LLM HELPERS ──────────────────────────
def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default


def get_llm_provider() -> LLMProvider | None:
    """Get the LLM provider from environment or return None for auto-detection."""
    provider_name = os.getenv("LLM_PROVIDER", "").strip().lower()
    if not provider_name:
        return None

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


def call_llm(
    client: BaseLLMClient,
    model_name: str,
    prompt: str,
    max_retries: int = 3,
    retry_delay: int = 10,
    seed: int = 42,
    timeout: int = 600,
) -> str:
    """Call the LLM with retries and return the response text."""
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


# ─── USER INPUT HELPERS ───────────────────
def get_prompt_version(prompt_filename: str) -> str:
    """Return the version label recorded in output filenames and records.

    Prompts ship unversioned in prompts/, so there is nothing to choose
    between; the label is what the paper refers to as the prompt version.
    """
    if not (PROMPTS_DIR / prompt_filename).exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPTS_DIR / prompt_filename}")
    return PROMPT_VERSION


def load_prompt_template(prompt_version: str, prompt_filename: str) -> str:
    """Load a prompt template. prompt_version is a label only; see get_prompt_version."""
    prompt_file = PROMPTS_DIR / prompt_filename
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")


def get_venue_selection() -> list[VenueConfig]:
    """Prompt user to select venues."""
    print("\n📋 Available venues:")
    for key, venue in VENUES.items():
        print(f"   {key}. {venue.name}")
    print("   a. All venues")

    while True:
        selection = input("\nEnter venue numbers (comma-separated) or 'a' for all: ").strip().lower()
        if selection == "a":
            return list(VENUES.values())

        selected_keys = [s.strip() for s in selection.split(",")]
        selected_venues = []
        valid = True

        for key in selected_keys:
            if key in VENUES:
                selected_venues.append(VENUES[key])
            else:
                print(f"❌ Invalid option: {key}")
                valid = False
                break

        if valid and selected_venues:
            return selected_venues
        if not selected_venues:
            print("❌ No venues selected. Please try again.")


def get_max_workers(env_key: str = "MAX_WORKERS") -> int:
    """Prompt user for max workers."""
    default = int(os.getenv(env_key, str(MAX_WORKERS)))
    while True:
        try:
            user_input = input(f"\nEnter max workers (default: {default}): ").strip()
            if not user_input:
                return default
            workers = int(user_input)
            if workers > 0:
                return workers
            print("❌ Must be a positive integer.")
        except ValueError:
            print("❌ Invalid number.")


def get_item_limit() -> int | None:
    """Parse --limit N from command-line arguments.

    Returns None if no limit is set, otherwise the limit value.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit items per venue (for testing)")
    args, _ = parser.parse_known_args()
    if args.limit:
        print(f"⚠️  Item limit set: {args.limit} per venue")
    return args.limit


def get_resume_file() -> Path | None:
    """Ask user if they want to resume from an existing JSONL file."""
    print("\n📂 Resume from existing file?")
    resume = input("Enter path to existing JSONL file (or press Enter to start fresh): ").strip()

    if not resume:
        return None

    resume_path = Path(resume)
    if not resume_path.is_absolute():
        resume_path = PROJECT_ROOT / resume

    if not resume_path.exists():
        print(f"❌ File not found: {resume_path}")
        retry = input("Try again? (y/n): ").strip().lower()
        if retry in ("y", "yes"):
            return get_resume_file()
        return None

    if not resume_path.suffix == ".jsonl":
        print("⚠️  Warning: File doesn't have .jsonl extension")

    return resume_path


def make_output_path(prefix: str, venue: VenueConfig, model_tag: str, prompt_version: str) -> Path:
    """Generate a timestamped output file path in a venue-specific subdirectory.

    Output structure: results/<venue_short_name>/<prefix>_<model_tag>_<prompt_version>_<timestamp>.jsonl
    """
    venue_dir = PROJECT_ROOT / "results" / venue.short_name
    venue_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return venue_dir / f"{prefix}_{model_tag}_{prompt_version}_{timestamp}.jsonl"


def confirm_run(
    task_label: str,
    prompt_version: str,
    venues: list[VenueConfig],
    total_items: int,
    already_processed: int,
    model_name: str,
    provider_name: str,
    max_workers: int,
    output_file: Path,
    is_resume: bool,
) -> bool:
    """Display summary and ask for confirmation."""
    print("\n" + "=" * 60)
    print(f"📊 {task_label} — RUN CONFIGURATION")
    print("=" * 60)
    print(f"  Provider:       {provider_name}")
    print(f"  Model:          {model_name}")
    print(f"  Prompt version: {prompt_version}")
    print(f"  Max workers:    {max_workers}")
    print(f"  Output file:    {output_file}")
    if is_resume:
        print(f"  Mode:           RESUME (continuing from existing file)")
    else:
        print(f"  Mode:           NEW")
    print()
    print("  Selected venues:")
    for venue in venues:
        print(f"    - {venue.name}")
    print()
    if already_processed > 0:
        print(f"  Already processed: {already_processed:,}")
    print(f"  Remaining to process: {total_items:,}")
    print("=" * 60)

    while True:
        confirm = input("\nProceed with this configuration? (y/n): ").strip().lower()
        if confirm in ("y", "yes"):
            return True
        if confirm in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


# ─── BATCH RUNNER ─────────────────────────
def run_batch(
    items: list[Any],
    process_fn,
    id_key: str,
    processed_ids: set[str],
    max_workers: int,
) -> list[dict[str, str]]:
    """Process a batch of items concurrently using process_fn.

    process_fn(item) -> (item_id, success, error_msg)
    """
    failures: list[dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}

        for item in items:
            item_id = getattr(item, id_key, None)
            if item_id and item_id in processed_ids:
                continue

            future = executor.submit(process_fn, item)
            futures[future] = item_id

        for future in as_completed(futures):
            item_id = futures[future]
            _, success, err = future.result()
            if not success:
                failures.append({"id": item_id, "error": err})

    return failures


def run_with_retries(
    items: list[Any],
    process_fn,
    id_key: str,
    output_file: Path,
    id_field_in_output: str,
    max_workers: int,
) -> list[dict[str, str]]:
    """Run batch processing with retry rounds."""
    processed_ids = load_processed_ids(output_file, id_key=id_field_in_output)

    failures = run_batch(items, process_fn, id_key, processed_ids, max_workers)

    for round_idx in range(1, BATCH_RETRY_ROUNDS + 1):
        if not failures:
            break

        failed_ids = {f["id"] for f in failures}
        logger.info(f"Retry round {round_idx}: attempting {len(failed_ids)} failed items")

        processed_ids = load_processed_ids(output_file, id_key=id_field_in_output)
        retry_items = [item for item in items if getattr(item, id_key) in failed_ids]

        failures = run_batch(retry_items, process_fn, id_key, processed_ids, max_workers)

    return failures
