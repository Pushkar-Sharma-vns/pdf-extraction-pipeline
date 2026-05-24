import os
import json
import fitz  # PyMuPDF
from mlx_vlm import load, generate
from mlx_vlm.utils import load_image

# --- Configuration ---
JSON_PATH = "./test_results/pdf_model.json"
PDF_PATH = "india-real-estate-office-and-residential-market-jan-mar-2026-12790.pdf"
VLM_MODEL = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
OUTPUT_MD = "./test_results/final_hybrid_report.md"

def get_high_res_crop(pdf_path, page_num, bbox):
    """Crops a specific section of a PDF page at high resolution."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    # GLM-OCR bboxes are often normalized to 1000x1000 or 72dpi. 
    # We convert to a fitz Rect (x1, y1, x2, y2)
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    
    # Render at 4x zoom (approx 300 DPI) for better OCR accuracy
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=rect)
    img_path = f"temp_crop_p{page_num}.png"
    pix.save(img_path)
    return img_path

def run_orchestrator():
    # 1. Load the Layout Metadata
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # 2. Load Qwen2.5-VL-7B on M4 Pro GPU
    print("🚀 Loading Qwen2.5-VL-7B for chart reasoning...")
    model, processor = load(VLM_MODEL)

    final_content = []

    # 3. Process the JSON sequence
    for i, item in enumerate(data):
        label = item.get("label")
        content = item.get("content", "")
        
        # Add standard text/tables directly
        if label != "figure":
            final_content.append(f"{content}\n")

        # Detect Figure and trigger VLM
        if label == "figure" or (label == "text" and "chart" in content.lower()):
            print(f"📊 Investigating potential chart at index {i}...")
            
            # Get coordinates and page
            bbox = item.get("bbox_2d")
            page_num = item.get("page_idx", 0)
            
            # Crop clean high-res image from original PDF
            crop_path = get_high_res_crop(PDF_PATH, page_num, bbox)
            
            # 4. Prompt Qwen with "Relevance Check"
            vision_img = load_image(crop_path)
            prompt = (
                "<|user|>\n<|vision_start|><|image_pad|><|vision_end|>"
                "Task: Analyze this image. If it is a data-driven chart or graph (bar, line, pie, etc.), "
                "extract the data points into a Markdown table. "
                "If it is just a photo, map, or decoration with no extractable data, reply ONLY with 'IGNORE'. "
                "<|assistant|>\n"
            )
            
            vlm_output = generate(model, processor, vision_img, prompt, max_tokens=1024)
            
            if "IGNORE" not in vlm_output:
                print(f"✅ Data extracted from chart on page {page_num}")
                final_content.append(f"\n> **[VLM Chart Data]**\n\n{vlm_output}\n")
            else:
                print(f"⏩ Irrelevant figure ignored.")

    # 5. Merge and Save
    with open(OUTPUT_MD, "w") as f:
        f.writelines(final_content)
    
    print(f"🎉 Final report generated: {OUTPUT_MD}")

if __name__ == "__main__":
    run_orchestrator()