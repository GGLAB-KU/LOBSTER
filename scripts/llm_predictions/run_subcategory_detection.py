#!/usr/bin/env python3
"""
Classify negatively biased reviews into the four negative-bias patterns.

Takes the reviews that bias detection already flagged as negative
(dataset/llm_predictions/<venue>/bias_results_*.jsonl), pairs each with its
paper title/abstract and review text from the raw corpus, and asks the LLM to
assign one of the four patterns defined in prompts/negative_bias_subcategory.md:

    A  Generalizability Demand
    B  English as the Gold Standard
    C  Language Choice Interrogation
    D  Impact Discounting

Output: dataset/llm_predictions/negative_bias_subcategories/<venue>/predictions.json
(one JSON array per venue).

Note: the released predictions.json files were produced before this script
existed, so re-running will not reproduce them byte-for-byte.

Usage:
    python run_subcategory_detection.py
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# base_runner and llm_providers both live at the LOBSTER repo root
_LOBSTER_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_LOBSTER_ROOT))

from base_runner import (
    PROJECT_ROOT,
    VenueConfig,
    call_llm,
    confirm_run,
    get_item_limit,
    get_llm_provider,
    get_max_workers,
    get_prompt_version,
    get_review_text,
    get_venue_selection,
    load_jsonl,
    load_paper_records,
    load_prompt_template,
)
from llm_providers import create_llm_client

logger = logging.getLogger(__name__)

PROMPT_FILENAME = "negative_bias_subcategory.md"
PREDICTIONS_DIR = PROJECT_ROOT / "dataset" / "llm_predictions"
OUTPUT_ROOT = PREDICTIONS_DIR / "negative_bias_subcategories"
VALID_PATTERNS = {"A", "B", "C", "D"}


@dataclass
class SubcategoryPayload:
    """A negatively biased review awaiting pattern classification."""
    venue: str
    paper_id: str
    review_id: str
    title: str
    abstract: str
    review_text: str


# ─── DATA LOADING ─────────────────────────
def find_bias_results(venue: VenueConfig) -> Path | None:
    """Locate the most recent bias_results file for a venue."""
    venue_dir = PREDICTIONS_DIR / venue.short_name
    matches = sorted(venue_dir.glob("bias_results_*.jsonl"))
    return matches[-1] if matches else None


def load_negative_reviews(venues: list[VenueConfig]) -> list[SubcategoryPayload]:
    """Load every review that bias detection flagged as negatively biased."""
    payloads: list[SubcategoryPayload] = []

    for venue in venues:
        results_file = find_bias_results(venue)
        if not results_file:
            logger.warning(
                f"No bias_results_*.jsonl in {PREDICTIONS_DIR / venue.short_name}; "
                f"extract llm_predictions.zip or run run_bias_detection.py first"
            )
            continue

        negative_ids = {
            r["review_id"]
            for r in load_jsonl(results_file)
            if any(b.get("type") == "negative" for b in r.get("biases", []))
        }
        if not negative_ids:
            logger.info(f"{venue.name}: no negatively biased reviews")
            continue

        # Pull title/abstract/review text for those reviews out of the raw corpus.
        count = 0
        for paper in load_paper_records(venue):
            for review in paper.reviews:
                review_id = review.get("note_id") or review.get("rid")
                if review_id not in negative_ids:
                    continue
                text = get_review_text(review)
                if not text:
                    continue
                payloads.append(SubcategoryPayload(
                    venue=venue.short_name,
                    paper_id=paper.paper_id,
                    review_id=review_id,
                    title=paper.title,
                    abstract=paper.abstract,
                    review_text=text,
                ))
                count += 1

        logger.info(f"{venue.name}: {count:,} of {len(negative_ids):,} negative reviews resolved")

    return payloads


# ─── PROMPT BUILDING ─────────────────────
def build_prompt(template: str, title: str, abstract: str, review_text: str) -> str:
    """Build the prompt by substituting placeholders."""
    return (
        template
        .replace("{title}", title)
        .replace("{abstract}", abstract)
        .replace("{review}", review_text)
    )


# ─── RESPONSE PARSING ─────────────────────
def parse_response(response_text: str) -> dict[str, str]:
    """Parse the LLM JSON response into pattern / evidence / reasoning."""
    try:
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No JSON object found in response")
        data = json.loads(response_text[json_start:json_end])

        pattern = str(data.get("pattern", "")).strip().upper()[:1]
        if pattern not in VALID_PATTERNS:
            raise ValueError(f"Invalid pattern {data.get('pattern')!r}")

        return {
            "predicted_pattern": pattern,
            "predicted_evidence": str(data.get("evidence", "")).strip(),
            "reasoning": str(data.get("reasoning", "")).strip(),
            "parse_error": "",
        }
    except Exception as e:
        # Record the failure instead of emitting a blank pattern that would be
        # indistinguishable from a genuine classification.
        logger.error(f"Error parsing LLM response: {e}")
        logger.debug("Raw response: %s", response_text[:1000])
        return {
            "predicted_pattern": "",
            "predicted_evidence": "",
            "reasoning": "",
            "parse_error": f"{type(e).__name__}: {e}",
        }


# ─── REVIEW PROCESSING ────────────────────
def process_review(
    payload: SubcategoryPayload,
    client,
    model_name: str,
    prompt_template: str,
    results: list[dict[str, Any]],
    lock: threading.Lock,
) -> tuple[str, bool, str]:
    """Classify a single review and collect the result."""
    try:
        prompt = build_prompt(
            prompt_template, payload.title, payload.abstract, payload.review_text
        )
        response_text = call_llm(client, model_name, prompt)
        parsed = parse_response(response_text)

        record = {
            "title": payload.title,
            "review_id": payload.review_id,
            "paper_id": payload.paper_id,
            "venue": payload.venue,
            "bias_type": "negative",
            "review_text": payload.review_text,
            "llm_response": response_text,
            "predicted_pattern": parsed["predicted_pattern"],
            "predicted_evidence": parsed["predicted_evidence"],
            "reasoning": parsed["reasoning"],
            "gold_pattern": None,
            "gold_evidence": "",
        }
        if parsed["parse_error"]:
            record["parse_error"] = parsed["parse_error"]

        with lock:
            results.append(record)
        return payload.review_id, True, ""

    except Exception as e:
        logger.error(f"Failed {payload.venue}::{payload.review_id}: {e}")
        return payload.review_id, False, str(e)


def write_predictions(results: list[dict[str, Any]]) -> None:
    """Write one predictions.json per venue."""
    by_venue: dict[str, list[dict[str, Any]]] = {}
    for record in results:
        by_venue.setdefault(record["venue"], []).append(record)

    for venue, records in sorted(by_venue.items()):
        out_dir = OUTPUT_ROOT / venue
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "predictions.json"
        out_file.write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"Wrote {len(records):,} records to {out_file}")


# ─── MAIN ─────────────────────────────────
def main():
    """Main entry point."""
    print("=" * 60)
    print("NEGATIVE BIAS SUBCATEGORY CLASSIFICATION")
    print("=" * 60)

    provider = get_llm_provider()
    client = create_llm_client(provider)
    model_name = client.get_model_name()
    if not model_name:
        env_var = "GOOGLE_CLOUD_MODEL" if "Google" in client.provider_name else "OPENROUTER_MODEL"
        sys.exit(f"Model name not set. Set {env_var} in your .env file for {client.provider_name}")

    prompt_version = get_prompt_version(PROMPT_FILENAME)
    prompt_template = load_prompt_template(prompt_version, PROMPT_FILENAME)

    venues = get_venue_selection()
    payloads = load_negative_reviews(venues)
    if not payloads:
        sys.exit("No negatively biased reviews found. Run run_bias_detection.py first.")

    limit = get_item_limit()
    if limit:
        payloads = payloads[:limit]

    max_workers = get_max_workers()

    if not confirm_run(
        task_label="Negative bias subcategory classification",
        prompt_version=prompt_version,
        venues=venues,
        total_items=len(payloads),
        already_processed=0,
        model_name=model_name,
        provider_name=client.provider_name,
        max_workers=max_workers,
        output_file=OUTPUT_ROOT / "<venue>" / "predictions.json",
        is_resume=False,
    ):
        sys.exit("Aborted.")

    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_review, p, client, model_name, prompt_template, results, lock
            ): p
            for p in payloads
        }
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 25 == 0:
                logger.info(f"  {done:,}/{len(payloads):,} classified")

    write_predictions(results)
    logger.info(f"Done. {len(results):,} of {len(payloads):,} reviews classified.")


if __name__ == "__main__":
    main()
