"""Core processing: async parallel chunk extraction with context caching."""

import asyncio
import json
import time

from google import genai
from google.genai import types

from gemini_ocr.config import GEMINI_API_KEY
from rag_enrichment.schema import ContextualRetrievalPipelineSchema
from rag_enrichment.prompts import generate_strict_extraction_prompt

MODEL_ID = "gemini-2.5-flash"
CONCURRENT_CHUNKS = 10
MAX_RETRIES = 2
RETRY_DELAY = 3

# Gemini 2.5 Flash pricing (per 1M tokens, prompts <= 200K)
# prompt_token_count already includes cached_content_token_count
COST_UNCACHED_INPUT_PER_M = 0.30
COST_CACHED_INPUT_PER_M = 0.03
COST_OUTPUT_PER_M = 2.50


def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def _calc_cost(meta):
    """Calculate cost from usage metadata."""
    total_input = getattr(meta, "prompt_token_count", 0) or 0
    cached_input = getattr(meta, "cached_content_token_count", 0) or 0
    output_tokens = getattr(meta, "candidates_token_count", 0) or 0
    uncached_input = total_input - cached_input

    cost_usd = (
        (uncached_input / 1_000_000) * COST_UNCACHED_INPUT_PER_M
        + (cached_input / 1_000_000) * COST_CACHED_INPUT_PER_M
        + (output_tokens / 1_000_000) * COST_OUTPUT_PER_M
    )
    return {
        "input_tokens": uncached_input,
        "cached_tokens": cached_input,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }


def _build_row(pdf_name: str, chunk: dict, parsed: dict) -> dict:
    final_embedding_text = (
        f"Context: {parsed['contextual_situation']}\n\n"
        f"Chunk Content: {chunk['text']}"
    )
    return {
        "source_file": pdf_name,
        "page_number": chunk["page_number"],
        "final_embedding_text": final_embedding_text,
        "raw_text": chunk["text"],
        "contextual_situation": parsed["contextual_situation"],
        "source_publisher": parsed["source_publisher"],
        "report_title": parsed["report_title"],
        "report_type": parsed["report_type"],
        "publish_date_or_year": parsed["publish_date_or_year"],
        "time_horizon": parsed["time_horizon"],
        "forecast_years_upto": parsed["forecast_years_upto"],
        "zone": parsed["zone"],
        "city": parsed["city"],
        "micro_market": parsed["micro_market"],
        "corridor_tags": ", ".join(parsed["corridor_tags"]),
        "asset_class": parsed["asset_class"],
        "sub_asset_class": parsed["sub_asset_class"],
        "economic_lens": ", ".join(parsed["economic_lens"]),
        "content_intent": parsed["content_intent"],
        "content_certainty": parsed["content_certainty"],
        "stakeholder_lens_discussed": ", ".join(parsed["stakeholder_lens_discussed"]),
        "macro_event_anchors": ", ".join(parsed["macro_event_anchors"]),
        "methodology_basis": parsed["methodology_basis"],
        "data_sources": ", ".join(parsed["data_sources"]),
        "comparison_axes": ", ".join(parsed["comparison_axes"]),
        "linked_assets_json": json.dumps(parsed["linked_assets"]),
    }


async def _process_chunk(aclient, semaphore, cache_name, pdf_name, idx, total, chunk):
    """Process a single chunk with retry logic."""
    page = chunk["page_number"]
    prompt = generate_strict_extraction_prompt(chunk["text"])
    last_error = None

    async with semaphore:
        for attempt in range(MAX_RETRIES + 1):
            try:
                t_start = time.monotonic()
                response = await asyncio.wait_for(
                    aclient.models.generate_content(
                        model=MODEL_ID,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            cached_content=cache_name,
                            response_mime_type="application/json",
                            response_schema=ContextualRetrievalPipelineSchema,
                            temperature=0.0,
                        )
                    ),
                    timeout=90,
                )
                api_time = time.monotonic() - t_start

                cost_info = _calc_cost(response.usage_metadata) if response.usage_metadata else None
                validated = ContextualRetrievalPipelineSchema.model_validate_json(response.text)
                row = _build_row(pdf_name, chunk, validated.model_dump())

                cost_str = ""
                if cost_info:
                    cost_str = (
                        f" | in={cost_info['input_tokens']} "
                        f"cached={cost_info['cached_tokens']} "
                        f"out={cost_info['output_tokens']} "
                        f"${cost_info['cost_usd']:.5f}"
                    )
                print(f"    chunk {idx + 1}/{total} (p{page}) OK {api_time:.1f}s{cost_str}")
                return {"status": "ok", "row": row, "cost": cost_info}

            except asyncio.TimeoutError:
                last_error = f"timeout after 90s (attempt {attempt + 1}/{MAX_RETRIES + 1})"
                print(f"    chunk {idx + 1}/{total} (p{page}) TIMEOUT attempt {attempt + 1}")
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)
            except Exception as e:
                last_error = str(e) or type(e).__name__
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(RETRY_DELAY)

    print(f"    chunk {idx + 1}/{total} (p{page}) FAILED after {MAX_RETRIES + 1} attempts — {last_error}")
    return {"status": "failed", "page": page, "error": last_error}


async def _process_report_async(pdf_name, full_markdown, chunks, workers):
    client = _get_client()
    aclient = client.aio

    print(f"  Creating context cache ({len(full_markdown):,} chars)...")
    cache = await aclient.caches.create(
        model=MODEL_ID,
        config=types.CreateCachedContentConfig(
            contents=[f"This is the complete source real estate report document content:\n\n{full_markdown}"],
            ttl="3600s",
        )
    )
    print(f"  Cache created: {cache.name}")

    semaphore = asyncio.Semaphore(workers)

    tasks = [
        _process_chunk(aclient, semaphore, cache.name, pdf_name, i, len(chunks), chunk)
        for i, chunk in enumerate(chunks)
    ]

    results = await asyncio.gather(*tasks)

    try:
        await aclient.caches.delete(name=cache.name)
        print(f"  Cache deleted")
    except Exception:
        print(f"  Cache already expired")

    return results


def process_report(
    pdf_name: str,
    full_markdown: str,
    chunks: list[dict],
    workers: int = CONCURRENT_CHUNKS,
) -> dict:
    """Process all chunks async with retries. Returns stats dict with records and errors."""
    t0 = time.time()

    results = asyncio.run(_process_report_async(pdf_name, full_markdown, chunks, workers))

    records = []
    failed_pages = []
    total_cost = 0.0
    total_input = 0
    total_cached = 0
    total_output = 0

    for r in results:
        if r["status"] == "ok":
            records.append(r["row"])
            if r.get("cost"):
                total_cost += r["cost"]["cost_usd"]
                total_input += r["cost"]["input_tokens"]
                total_cached += r["cost"]["cached_tokens"]
                total_output += r["cost"]["output_tokens"]
        else:
            failed_pages.append({"page": r["page"], "error": r["error"]})

    elapsed = time.time() - t0

    return {
        "records": records,
        "total_pages": len(chunks),
        "succeeded": len(records),
        "failed": len(failed_pages),
        "failed_pages": failed_pages,
        "elapsed_seconds": round(elapsed, 1),
        "cost": {
            "total_usd": round(total_cost, 5),
            "uncached_input_tokens": total_input,
            "cached_read_tokens": total_cached,
            "output_tokens": total_output,
        },
    }
