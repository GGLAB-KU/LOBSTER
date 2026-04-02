#!/usr/bin/env python3
"""
Interactive script to detect languages studied in papers.

Uses base_runner for shared infrastructure (located in the
language-bias-peer-review project root).

Usage:
    python run_language_detection.py
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

# Add the language-bias-peer-review project root to sys.path for base_runner/llm_providers
_LOBSTER_ROOT = Path(__file__).resolve().parent.parent.parent
_PEER_REVIEW_ROOT = _LOBSTER_ROOT.parent / "language-bias-peer-review"
if _PEER_REVIEW_ROOT.exists():
    sys.path.insert(0, str(_PEER_REVIEW_ROOT))
else:
    sys.exit(
        f"ERROR: Companion repo not found at {_PEER_REVIEW_ROOT}\n"
        "This script requires the language-bias-peer-review repository.\n"
        "Clone it alongside LOBSTER:\n"
        "  git clone https://github.com/GGLAB-KU/language-bias-peer-review.git\n\n"
        "Expected layout:\n"
        "  parent_dir/\n"
        "    LOBSTER/\n"
        "    language-bias-peer-review/\n"
    )

from base_runner import (
    PROJECT_ROOT,
    VenueConfig,
    append_jsonl,
    call_llm,
    confirm_run,
    get_item_limit,
    get_llm_provider,
    get_max_workers,
    get_prompt_version,
    get_resume_file,
    get_review_text,
    get_venue_selection,
    load_paper_records,
    load_processed_ids,
    load_prompt_template,
    make_output_path,
    run_with_retries,
)
from llm_providers import create_llm_client

logger = logging.getLogger(__name__)

PROMPT_FILENAME = "languages_of_study.md"


# ─── DATA STRUCTURES ──────────────────────
@dataclass
class PaperPayload:
    """Payload for a paper to be processed."""
    venue: str
    paper_id: str
    title: str
    abstract: str
    reviews_text: str


# ─── DATA LOADING ─────────────────────────
def load_papers(venues: list[VenueConfig]) -> list[PaperPayload]:
    """Load all papers with their review texts from selected venues."""
    payloads: list[PaperPayload] = []
    for venue in venues:
        logger.info(f"Loading {venue.name}...")
        records = load_paper_records(venue)
        count = 0
        for paper in records:
            if not paper.title or not paper.abstract:
                continue

            # Combine all review texts for this paper
            review_parts = []
            for i, review in enumerate(paper.reviews, 1):
                text = get_review_text(review)
                if text:
                    review_parts.append(f"--- Review {i} ---\n{text}")

            reviews_text = "\n\n".join(review_parts)

            payloads.append(PaperPayload(
                venue=venue.short_name,
                paper_id=paper.paper_id,
                title=paper.title,
                abstract=paper.abstract,
                reviews_text=reviews_text,
            ))
            count += 1
        logger.info(f"  Loaded {count:,} papers")
    return payloads


# ─── PROMPT BUILDING ─────────────────────
def build_prompt(template: str, title: str, abstract: str, reviews_text: str) -> str:
    """Build the prompt by substituting placeholders."""
    return (
        template
        .replace("{title}", title)
        .replace("{abstract}", abstract)
        .replace("{reviews_text}", reviews_text)
    )


# ─── RESPONSE PARSING ─────────────────────
def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM JSON response into language detection result.

    Expects v23 prompt output format:
    {
        "language_scope": "single-language",
        "languages": ["English"],
        "languages_count": 1,
        "evidence_type": "dataset_implied",
        "justification": "..."
    }
    """
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        data = json.loads(response_text[json_start:json_end])

        # Normalize languages to a flat list of strings
        languages = data.get("languages", [])
        if not isinstance(languages, list):
            languages = [languages] if languages else []
        normalized = []
        for lang in languages:
            if isinstance(lang, str):
                normalized.append(lang.strip())
            elif isinstance(lang, dict):
                name = (lang.get("language_name") or lang.get("language") or lang.get("name") or "").strip()
                if name:
                    normalized.append(name)
        # Deduplicate while preserving order
        seen = set()
        unique_languages = []
        for lang in normalized:
            if lang not in seen:
                seen.add(lang)
                unique_languages.append(lang)

        return {
            "language_scope": data.get("language_scope", ""),
            "languages": unique_languages,
            "languages_count": data.get("languages_count", len(unique_languages)),
            "evidence_type": data.get("evidence_type", ""),
            "justification": data.get("justification", ""),
        }

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return {
            "language_scope": "",
            "languages": [],
            "languages_count": 0,
            "evidence_type": "",
            "justification": f"Parse error: {e}",
            "parse_error": True,
        }


# ─── PAPER PROCESSING ────────────────────
def process_paper(
    payload: PaperPayload,
    client,
    model_name: str,
    prompt_template: str,
    prompt_version: str,
    output_file: Path,
    lock: threading.Lock,
) -> tuple[str, bool, str]:
    """Process a single paper and save the result."""
    try:
        prompt = build_prompt(
            prompt_template,
            payload.title,
            payload.abstract,
            payload.reviews_text,
        )
        response_text = call_llm(client, model_name, prompt)
        result = parse_response(response_text)

        record = {
            "venue": payload.venue,
            "paper_id": payload.paper_id,
            "title": payload.title,
            **result,
            "prompt_version": prompt_version,
            "model": model_name,
        }
        append_jsonl(record, output_file, lock)

        lang_names = result.get("languages", [])
        lang_str = ", ".join(lang_names) if lang_names else "None"
        logger.info(f"✔ Processed {payload.venue}::{payload.paper_id[:20]}... - {lang_str}")
        return payload.paper_id, True, ""

    except Exception as e:
        logger.error(f"❌ Failed {payload.venue}::{payload.paper_id}: {e}")
        return payload.paper_id, False, str(e)


# ─── MAIN ─────────────────────────────────
def main():
    """Main entry point."""
    print("=" * 60)
    print("🌍 LANGUAGE OF STUDY DETECTION IN NLP PAPERS")
    print("=" * 60)

    # Configuration
    prompt_version = get_prompt_version(PROMPT_FILENAME)
    venues = get_venue_selection()
    max_workers = get_max_workers("LANG_MAX_WORKERS")
    item_limit = get_item_limit()
    prompt_template = load_prompt_template(prompt_version, PROMPT_FILENAME)

    # LLM setup
    provider = get_llm_provider()
    client = create_llm_client(provider)
    model_name = client.get_model_name()
    if not model_name:
        raise ValueError("Model name not set. Check your .env file.")

    model_tag = model_name.rsplit("/", 1)[-1].replace("/", "-")

    # Resume file (applies globally, not per-venue)
    resume_file = get_resume_file()

    # Process each venue separately
    total_processed = 0
    total_failures: list[dict[str, str]] = []

    for venue in venues:
        print(f"\n{'─' * 60}")
        print(f"📂 Processing venue: {venue.name}")
        print(f"{'─' * 60}")

        # Output file (per-venue subfolder)
        output_file = resume_file or make_output_path("lang_results", venue, model_tag, prompt_version)

        # Load data for this venue
        print("📥 Loading papers...")
        venue_payloads = load_papers([venue])
        if not venue_payloads:
            logger.warning(f"No papers found for {venue.name}. Skipping.")
            continue
        if item_limit:
            venue_payloads = venue_payloads[:item_limit]

        processed_ids = load_processed_ids(output_file, id_key="paper_id")
        pending = len([p for p in venue_payloads if p.paper_id not in processed_ids])

        # Confirm
        if not confirm_run(
            task_label="LANGUAGE OF STUDY DETECTION",
            prompt_version=prompt_version,
            venues=[venue],
            total_items=pending,
            already_processed=len(processed_ids),
            model_name=model_name,
            provider_name=client.provider_name,
            max_workers=max_workers,
            output_file=output_file,
            is_resume=resume_file is not None,
        ):
            print(f"⏭️  Skipping {venue.name}.")
            continue

        # Process
        print("🚀 Starting processing...")
        lock = threading.Lock()
        process_fn = partial(
            process_paper,
            client=client,
            model_name=model_name,
            prompt_template=prompt_template,
            prompt_version=prompt_version,
            output_file=output_file,
            lock=lock,
        )

        failures = run_with_retries(
            items=venue_payloads,
            process_fn=process_fn,
            id_key="paper_id",
            output_file=output_file,
            id_field_in_output="paper_id",
            max_workers=max_workers,
        )

        # Per-venue summary
        venue_processed = len(load_processed_ids(output_file, id_key="paper_id"))
        total_processed += venue_processed
        total_failures.extend(failures)
        print(f"  ✅ {venue.name}: {venue_processed:,} processed, {len(failures)} failed")
        print(f"  📁 Output: {output_file}")

    # Overall summary
    print("\n" + "=" * 60)
    print("📊 ALL VENUES COMPLETE")
    print("=" * 60)
    print(f"  Total processed: {total_processed:,}")
    print(f"  Total failed:    {len(total_failures)}")

    if total_failures:
        logger.error(f"{len(total_failures)} papers ultimately failed:")
        print(json.dumps(total_failures[:10], ensure_ascii=False, indent=2))
    else:
        logger.info("✅ All papers processed successfully!")

    print("\nDone.")


if __name__ == "__main__":
    main()
