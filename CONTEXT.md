# Project Context: Document Intelligence Pipeline

## 1. Environment & Hardware

### Local Development (MacBook Pro M4 Pro)
- **Machine:** MacBook Pro M4 Pro (24GB Unified Memory)
- **Python:** 3.12 (venv)
- **GLM-OCR Backend:** Ollama running `glm-ocr:latest` (0.9B) on port 11434
- **Config:** `config.yaml` → `api_mode: ollama_generate`
- **Fallback config:** `config_ollama.yaml`

### Production (EC2 — planned)
- **Instance:** g4dn.xlarge (1x NVIDIA T4 16GB VRAM, 4 vCPU, 16GB RAM)
- **GLM-OCR Backend:** vLLM serving `zai-org/GLM-OCR` on port 8080
- **Config:** `config.yaml` → `api_mode: openai`
- **Why vLLM:** Continuous batching, speculative decoding, 8+ concurrent workers. ~6x faster than Ollama for batch processing.

## 2. Architecture Overview

```
GCS Bucket: re_reports/
├── <source>/                    (jll, cushman, crematrix, savills)
│   ├── pdfs/                    ← Original PDFs
│   └── extracted_pdfs/
│       ├── bengaluru_data/      ← PDFs mentioning Bengaluru/Karnataka
│       │   ├── <stem>_gemini.md              (Gemini extraction)
│       │   └── <stem>_extracted_images/      (GLM-OCR polygon crops)
│       │       ├── page1_chart_idx5.jpg
│       │       ├── page3_image_idx2.jpg
│       │       └── classification_results.json  (Gemini classification)
│       └── india_data/          ← All other PDFs
│           └── (same structure)
```

## 3. Two Extraction Pipelines

### Pipeline A: Gemini OCR (`gemini_ocr/`)
- **What:** Batch PDF → Markdown extraction using Gemini 3 Flash Preview
- **Input:** PDFs from GCS bucket (`<source>/pdfs/`)
- **Process:** Split PDF into 10-page chunks → upload to Gemini → extract text/tables/charts → classify as bengaluru_data or india_data
- **Output:** `<stem>_gemini.md` uploaded to `<source>/extracted_pdfs/{category}/`
- **Status:** Complete for all 4 sources (894 PDFs processed)

### Pipeline B: GLM-OCR + Image Extraction (`glm_postprocess/`)
- **What:** Layout detection + polygon-based image extraction + image classification
- **Input:** PDFs from GCS (discovered via existing `_gemini.md` files in `extracted_pdfs/`)
- **Process:** Two phases (see below)
- **Output:** Cropped images + `classification_results.json` in GCS

## 4. GLM-OCR Pipeline — Phase 1 & Phase 2

### Phase 1: Extract Images
1. Read `_gemini.md` filenames in `extracted_pdfs/{category}/` to discover processed PDFs
2. Download original PDF from `<source>/pdfs/<stem>.pdf`
3. Run `glmocr parse` → produces `_model.json` (polygon coordinates) + `layout_vis/` (rendered page JPGs)
4. Crop pictorial regions (image, chart, figure_title) using bbox_2d coordinates mapped from GLM-OCR's 1000-normalized grid to pixel space
5. Filter crops < 100x100 px
6. Upload crops to `<source>/extracted_pdfs/{category}/<stem>_extracted_images/`

**Naming:** `page<N>_<label>_idx<I>.jpg` — page number (1-based), label type, item index from GLM-OCR JSON

### Phase 2: Classify Images
1. List all images in `<stem>_extracted_images/` folder in GCS
2. Download each image as bytes from GCS
3. Send to Gemini via `types.Part.from_bytes()` with async workers (20 concurrent via `asyncio.Semaphore`)
4. Gemini returns structured JSON: `{is_relevant, confidence_score, category, reasoning}`
5. Upload `classification_results.json` to the same GCS folder

**Classification criteria:** Relevant = charts, graphs, property photos, maps, floor plans, infographics with RE metrics. Not relevant = logos, headshots, icons, decorative elements, stock imagery.

**Pydantic schema:** `ImageClassification` with `is_relevant` (bool), `confidence_score` (float), `category` (str), `reasoning` (str)

### Running the Pipeline

```bash
# Phase 1 + Phase 2 chained (single command)
python -m glm_postprocess.orchestrator --source jll --category bengaluru_data --limit 5 --classify

# Phase 1 only
python -m glm_postprocess.orchestrator --source jll --workers 8

# Phase 2 only (on already-uploaded images)
python -m glm_postprocess.classifier --source jll --workers 20

# Resume interrupted runs
python -m glm_postprocess.orchestrator --source jll --resume --workers 8
python -m glm_postprocess.classifier --source jll --resume
```

## 5. GLM-OCR Backend: Ollama (Local) vs vLLM (Prod)

### Ollama (Local Development)
```yaml
# config.yaml
ocr_api:
  api_host: localhost
  api_port: 11434
  api_path: /api/generate
  model: glm-ocr:latest
  api_mode: ollama_generate
```
- Sequential processing, ~4 min/PDF
- Good for development and testing
- Can run with `OLLAMA_NUM_PARALLEL=4` for modest parallelism

### vLLM (Production on EC2)
```yaml
# config.yaml
ocr_api:
  api_host: <EC2_IP>
  api_port: 8080
  api_path: /v1/chat/completions
  model: glm-ocr
  api_mode: openai
```
- Continuous batching, speculative decoding
- 8+ concurrent workers, ~1-2 min/PDF
- Launch: `vllm serve zai-org/GLM-OCR --port 8080 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' --served-model-name glm-ocr --gpu-memory-utilization 0.9`

### Switching
```bash
cp config_ollama.yaml config.yaml   # → Ollama
# Edit config.yaml manually         # → vLLM (change api_host, api_port, api_mode)
```

## 6. Module Structure

```
gemini_ocr/               # Gemini PDF extraction pipeline
  config.py               # GEMINI_API_KEY, GCS_BUCKET, etc. (from .env)
  gcs_client.py           # GCS operations (list, download, upload)
  extractor.py            # Gemini chunk + extract logic
  pipeline.py             # GCS→Gemini→GCS orchestration + classify_pdf()

glm_postprocess/           # GLM-OCR image extraction + classification
  config.py               # PICTORIAL_LABELS, NORM, GLMOCR_BIN
  image_extractor.py      # Polygon-based cropping from layout_vis
  gcs_upload.py           # Image upload to GCS + classify_pdf re-export
  classifier.py           # Phase 2: async Gemini image classification
  orchestrator.py         # CLI: Phase 1 → Phase 2 end-to-end

Root scripts (legacy):
  extract_images.py       # Original standalone image extractor
  enrich_report.py        # Qwen2.5-VL enrichment (MLX, Apple Silicon only)
  hybrid_orchestrator.py  # Original batch GLM-OCR + enrich runner
```

## 7. Test Results (5 JLL Bengaluru PDFs)

| PDF | Images Extracted | Relevant | Not Relevant |
|---|---|---|---|
| asia-pacific-capital-tracker-autumn-2025 | 73 | 72 | 1 |
| asia-pacific-capital-tracker | 71 | 69 | 2 |
| india-data-centre-market-dynamics-h1-2025 | 38 | 18 | 20 |
| india-retail-market-dynamics-q1-2025 | 12 | 8 | 4 |
| pulse-real-estate-monthly-monitor-apr-2025 | 37 | 35 | 2 |
| **Total** | **231** | **202 (87%)** | **29** |
