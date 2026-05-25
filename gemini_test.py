import time
from google import genai
from google.genai import types

# --- Setup ---
import os
from dotenv import load_dotenv
load_dotenv()
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
PDF_PATH = "india-real-estate-office-and-residential-market-jan-mar-2026-12790.pdf"

# 2026 Pricing Constants (USD per 1M tokens)
PRICE_INPUT_1M = 0.50
PRICE_OUTPUT_1M = 3.00

# Initialize Client
client = genai.Client(api_key=GEMINI_API_KEY)

def run_gemini_3_extraction():
    # 1. Upload File
    print(f"📤 Uploading {PDF_PATH}...")
    file_upload = client.files.upload(file=PDF_PATH)
    
    while file_upload.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(1)
        file_upload = client.files.get(name=file_upload.name)

    if file_upload.state.name == "FAILED":
        print("\n❌ File processing failed.")
        return

    print(f"\n✅ File ready: {file_upload.name}")

    # 2. Modern Extraction Prompt
    prompt = """
    Perform a high-fidelity extraction of the first 10 pages of this PDF.
    Extract the page-wise data. In each page:
    - Transcribe narrative text.
    - Convert all tables into clean Markdown format.
    - Identify charts/graphs and convert visual data points into Markdown tables.
    """

    print("🧠 Reasoning with Gemini 3 Flash Preview...")
    
    # Using the correct 2026 preview model ID
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=[file_upload, prompt],
        config=types.GenerateContentConfig(
            temperature=0.1,
            # Gemini 3 models allow setting thinking_level for reasoning tasks
            # thinking_level=types.ThinkingLevel.MINIMAL 
        )
    )

    # 3. Save Results
    with open("extraction_results.md", "w", encoding="utf-8") as f:
        f.write(response.text)

    # 4. Token & Cost Calculation
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count
    output_tokens = usage.candidates_token_count
    total_tokens = usage.total_token_count

    # Calculate cost (USD)
    cost_input = (input_tokens / 1_000_000) * PRICE_INPUT_1M
    cost_output = (output_tokens / 1_000_000) * PRICE_OUTPUT_1M
    total_cost = cost_input + cost_output

    print("\n" + "="*30)
    print("📊 EXTRACTION STATISTICS")
    print(f"Input Tokens:  {input_tokens:,}")
    print(f"Output Tokens: {output_tokens:,}")
    print(f"Total Tokens:  {total_tokens:,}")
    print("-"*30)
    print(f"Estimated Cost: ${total_cost:.6f} USD")
    print("="*30)
    print(f"🎉 Results saved to extraction_results.md")

if __name__ == "__main__":
    run_gemini_3_extraction()