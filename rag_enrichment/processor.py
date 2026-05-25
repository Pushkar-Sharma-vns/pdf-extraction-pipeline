"""Core processing: context cache + per-chunk Gemini metadata extraction."""

import json

from google import genai
from google.genai import types

from gemini_ocr.config import GEMINI_API_KEY
from rag_enrichment.schema import ContextualRetrievalPipelineSchema
from rag_enrichment.prompts import generate_strict_extraction_prompt

MODEL_ID = "gemini-2.5-flash"


def _get_client():
    return genai.Client(api_key=GEMINI_API_KEY)


def process_report(pdf_name: str, full_markdown: str, chunks: list[dict]) -> list[dict]:
    """Process all chunks of a report against a cached full-document context.

    Returns list of flattened row dicts ready for DataFrame/JSON export.
    """
    client = _get_client()
    processed = []

    print(f"  Creating context cache for {pdf_name} ({len(full_markdown):,} chars)...")
    cache = client.caches.create(
        model=MODEL_ID,
        config=types.CreateCachedContentConfig(
            contents=[f"This is the complete source real estate report document content:\n\n{full_markdown}"],
            ttl="600s",
        )
    )

    try:
        for idx, chunk in enumerate(chunks):
            print(f"    chunk {idx + 1}/{len(chunks)} (page {chunk['page_number']})...", end=" ")

            prompt = generate_strict_extraction_prompt(chunk["text"])

            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        cached_content=cache.name,
                        response_mime_type="application/json",
                        response_schema=ContextualRetrievalPipelineSchema,
                        temperature=0.0,
                    )
                )

                validated = ContextualRetrievalPipelineSchema.model_validate_json(response.text)
                parsed = validated.model_dump()

                final_embedding_text = (
                    f"Context: {parsed['contextual_situation']}\n\n"
                    f"Chunk Content: {chunk['text']}"
                )

                row = {
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
                processed.append(row)
                print("OK")

            except Exception as e:
                print(f"FAILED — {e}")
                continue

    finally:
        client.caches.delete(name=cache.name)
        print(f"  Cache deleted for {pdf_name}")

    return processed
