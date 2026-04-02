#!/usr/bin/env python3
"""
Interactive script to detect contribution focus types of papers.

Uses base_runner for shared infrastructure (located in the
language-bias-peer-review project root).

Usage:
    python run_contribution_type.py
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
    get_venue_selection,
    load_paper_records,
    load_processed_ids,
    load_prompt_template,
    make_output_path,
    run_with_retries,
)
from llm_providers import create_llm_client

logger = logging.getLogger(__name__)

PROMPT_FILENAME = "contribution_type.md"


# ─── DATA STRUCTURES ──────────────────────
@dataclass
class PaperPayload:
    """Payload for a paper to be processed."""
    venue: str
    paper_id: str
    title: str
    abstract: str


# ─── DATA LOADING ─────────────────────────
def load_papers(venues: list[VenueConfig]) -> list[PaperPayload]:
    """Load all papers from selected venues."""
    payloads: list[PaperPayload] = []
    for venue in venues:
        logger.info(f"Loading {venue.name}...")
        records = load_paper_records(venue)
        count = 0
        for paper in records:
            if not paper.title or not paper.abstract:
                continue
            payloads.append(PaperPayload(
                venue=venue.short_name,
                paper_id=paper.paper_id,
                title=paper.title,
                abstract=paper.abstract,
            ))
            count += 1
        logger.info(f"  Loaded {count:,} papers")
    return payloads


# ─── PROMPT BUILDING ─────────────────────
def build_prompt(template: str, title: str, abstract: str) -> str:
    """Build the prompt by substituting placeholders."""
    return (
        template
        .replace("{title}", title)
        .replace("{abstract}", abstract)
    )


# ─── RESPONSE PARSING ─────────────────────
VALID_CONTRIBUTION_TYPES = {
    "Modeling",
    "NLPApplications",
    "DataAndBenchmarking",
    "EmpiricalAnalysis",
    "LinguisticAnalysis",
    "DomainAdaptation",
    "SurveyOrPosition",
    "Other",
}


def parse_response(response_text: str) -> dict[str, Any]:
    """Parse the LLM JSON response into contribution focus result."""
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        data = json.loads(response_text[json_start:json_end])

        contribution_type = data.get("contribution_type", [])
        if not isinstance(contribution_type, list):
            contribution_type = [contribution_type] if contribution_type else []

        validated_focus = []
        for focus in contribution_type:
            focus_str = str(focus).strip()
            if focus_str in VALID_CONTRIBUTION_TYPES:
                validated_focus.append(focus_str)
            else:
                logger.warning(f"Invalid contribution type: {focus_str}")

        if not validated_focus:
            validated_focus = ["Other"]

        justification = data.get("justification", "").strip()

        return {
            "contribution_type": validated_focus,
            "justification": justification,
        }

    except Exception as e:
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return {
            "contribution_type": ["Other"],
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
        )
        response_text = call_llm(client, model_name, prompt)
        result = parse_response(response_text)

        record = {
            "venue": payload.venue,
            "paper_id": payload.paper_id,
            **result,
            "prompt_version": prompt_version,
            "model": model_name,
        }
        append_jsonl(record, output_file, lock)

        focus = result.get("contribution_type", [])
        focus_str = ", ".join(focus) if focus else "Unknown"
        logger.info(f"✔ Processed {payload.venue}::{payload.paper_id[:20]}... - {focus_str}")
        return payload.paper_id, True, ""

    except Exception as e:
        logger.error(f"❌ Failed {payload.venue}::{payload.paper_id}: {e}")
        return payload.paper_id, False, str(e)


# ─── MAIN ─────────────────────────────────
def main():
    """Main entry point."""
    print("=" * 60)
    print("📑 CONTRIBUTION FOCUS DETECTION IN NLP PAPERS")
    print("=" * 60)

    # Configuration
    prompt_version = get_prompt_version(PROMPT_FILENAME)
    venues = get_venue_selection()
    max_workers = get_max_workers("CONTRIB_MAX_WORKERS")
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
        output_file = resume_file or make_output_path("contrib_results", venue, model_tag, prompt_version)

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
            task_label="CONTRIBUTION FOCUS DETECTION",
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
