def generate_strict_extraction_prompt(chunk_text: str) -> str:
    return f"""
You are a highly analytical, strict real estate data parser processing chunks for a specialized vector database payload.

Your output must be divided logically into two operations:
1. Provide a two-sentence contextual summary positioning this chunk within the parent document.
2. Categorize the text snippet against the strict property metadata taxonomy.

=== MANDATE 1: ZERO HALLUCINATION CONTEXT SELECTION ===
- Write exactly a 2-sentence context anchoring this specific chunk text back to the parent document.
- It must convey the who, when, and where of the report so a reader looking ONLY at the chunk can interpret the data point seamlessly.
- You must rely solely on the facts directly visible in either the cached full report or this chunk text.
- NEVER append external knowledge, guess unknown figures, or speculate on trends not explicitly typed out.
- Do not add conversational fluff or introductory text. Start directly with the context.

=== MANDATE 2: TAXONOMY MATCHING CRITERIA ===
- Review the specific snippet and map its properties to the required keys.
- For categoricals/enums (e.g., source_publisher, report_type, time_horizon, asset_class, sub_asset_class, macro_event_anchors, comparison_axes, etc.), you must choose strictly from the permitted choices.
- If a parameter, micro-market, event anchor, or tracking element is NOT mentioned or implied unmistakably by the chunk text, you MUST output null or an empty array. Do not guess or extrapolate.

Target Text Chunk to Analyze:
\"\"\"
{chunk_text}
\"\"\"
"""
