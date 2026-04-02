#!/usr/bin/env python3
"""
Interactive script to run language bias detection on peer reviews.

Uses base_runner for shared infrastructure (located in the
language-bias-peer-review project root).

Usage:
    python run_bias_detection.py
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

PROMPT_FILENAME = "review_biases_toward_language.md"


# ─── DATA STRUCTURES ──────────────────────
@dataclass
class ReviewPayload:
    """Payload for a review to be processed."""
    venue: str
    paper_id: str
    review_id: str
    title: str
    abstract: str
    review_text: str


# ─── DATA LOADING ─────────────────────────
def load_reviews(venues: list[VenueConfig]) -> list[ReviewPayload]:
    """Load all reviews from selected venues."""
    payloads: list[ReviewPayload] = []
    for venue in venues:
        logger.info(f"Loading {venue.name}...")
        records = load_paper_records(venue)
        count = 0
        for paper in records:
            for review in paper.reviews:
                review_id = review.get("note_id") or review.get("rid")
                if not review_id:
                    continue
                text = get_review_text(review)
                if not text:
                    continue
                payloads.append(ReviewPayload(
                    venue=venue.short_name,
                    paper_id=paper.paper_id,
                    review_id=review_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    review_text=text,
                ))
                count += 1
        logger.info(f"  Loaded {count:,} reviews")
    return payloads


# ─── PROMPT BUILDING ─────────────────────
def build_prompt(template: str, title: str, abstract: str, review_text: str) -> str:
    """Build the prompt by substituting placeholders."""
    return (
        template
        .replace("{title}", title)
        .replace("{abstract}", abstract)
        .replace("{review_text}", review_text)
    )


# ─── RESPONSE PARSING ─────────────────────
def parse_response(response_text: str) -> list[dict[str, str]]:
    """Parse the LLM JSON response into a list of biases."""
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        data = json.loads(response_text[json_start:json_end])

        biases = data.get("biases", [])
        if not isinstance(biases, list):
            biases = []

        allowed_types = {"negative", "positive"}
        parsed: list[dict[str, str]] = []

        for item in biases:
            if not isinstance(item, dict):
                continue

            quoted = (item.get("quoted_text") or "").strip()
            bias_type = (item.get("type") or "").strip().lower()
            justification = (item.get("justification") or "").strip()

            if quoted and bias_type in allowed_types and justification:
                parsed.append({
                    "quoted_text": quoted,
                    "type": bias_type,
                    "justification": justification,
                })

        return parsed
    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return []


# ─── REVIEW PROCESSING ────────────────────
def process_review(
    payload: ReviewPayload,
    client,
    model_name: str,
    prompt_template: str,
    prompt_version: str,
    output_file: Path,
    lock: threading.Lock,
) -> tuple[str, bool, str]:
    """Process a single review and save the result."""
    try:
        prompt = build_prompt(
            prompt_template,
            payload.title,
            payload.abstract,
            payload.review_text,
        )
        response_text = call_llm(client, model_name, prompt)
        biases = parse_response(response_text)

        record = {
            "venue": payload.venue,
            "paper_id": payload.paper_id,
            "review_id": payload.review_id,
            "biases": biases,
            "prompt_version": prompt_version,
            "model": model_name,
        }
        append_jsonl(record, output_file, lock)

        bias_count = len(biases)
        logger.info(f"✔ Processed {payload.venue}::{payload.review_id[:20]}... - {bias_count} bias(es)")
        return payload.review_id, True, ""

    except Exception as e:
        logger.error(f"❌ Failed {payload.venue}::{payload.review_id}: {e}")
        return payload.review_id, False, str(e)


# ─── MAIN ─────────────────────────────────
def main():
    """Main entry point."""
    print("=" * 60)
    print("🔍 LANGUAGE BIAS DETECTION IN PEER REVIEWS")
    print("=" * 60)

    # Configuration
    prompt_version = get_prompt_version(PROMPT_FILENAME)
    venues = get_venue_selection()
    max_workers = get_max_workers("BIAS_MAX_WORKERS")
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
        output_file = resume_file or make_output_path("bias_results", venue, model_tag, prompt_version)

        # Load data for this venue
        print("📥 Loading reviews...")
        venue_payloads = load_reviews([venue])
        if not venue_payloads:
            logger.warning(f"No reviews found for {venue.name}. Skipping.")
            continue
        if item_limit:
            venue_payloads = venue_payloads[:item_limit]

        processed_ids = load_processed_ids(output_file, id_key="review_id")
        pending = len([p for p in venue_payloads if p.review_id not in processed_ids])

        # Confirm
        if not confirm_run(
            task_label="LANGUAGE BIAS DETECTION",
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
            process_review,
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
            id_key="review_id",
            output_file=output_file,
            id_field_in_output="review_id",
            max_workers=max_workers,
        )

        # Per-venue summary
        venue_processed = len(load_processed_ids(output_file, id_key="review_id"))
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
        logger.error(f"{len(total_failures)} reviews ultimately failed:")
        print(json.dumps(total_failures[:10], ensure_ascii=False, indent=2))
    else:
        logger.info("✅ All reviews processed successfully!")

    print("\nDone.")


if __name__ == "__main__":
    main()
