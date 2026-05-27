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

# Phase 2 only (on already-uploaded images, bengaluru only)
python -m glm_postprocess.classifier --source jll --category bengaluru_data --workers 20

# Resume interrupted runs
python -m glm_postprocess.orchestrator --source jll --category bengaluru_data --resume --workers 8
python -m glm_postprocess.classifier --source jll --category bengaluru_data --resume

# Classify a single folder directly
python -m glm_postprocess.classifier --folder jll/extracted_pdfs/bengaluru_data/report_extracted_images
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
  classifier.py           # Phase 2: async Gemini image classification (retry + parallel download)
  orchestrator.py         # CLI: Phase 1 → Phase 2 end-to-end

rag_enrichment/            # Contextual chunking + metadata extraction for RAG
  schema.py               # 25+ field Pydantic schema with strict Literal enums
  prompts.py              # Zero-hallucination extraction prompt
  chunker.py              # Split markdown on <!-- page N --> markers
  processor.py            # Gemini context caching + async parallel chunk extraction
  orchestrator.py         # CLI: GCS list → download → chunk → process → upload JSON+Excel

Root scripts:
  link_assets.py          # Match classified images → RAG metadata linked_assets by page
  extract_images.py       # Original standalone image extractor (legacy)
  enrich_report.py        # Qwen2.5-VL enrichment, MLX Apple Silicon only (legacy)
  hybrid_orchestrator.py  # Original batch GLM-OCR + enrich runner (legacy)
```

## 6a. RAG Enrichment Pipeline

Page-wise chunking + contextual metadata extraction for Qdrant RAG:
1. Download `_gemini.md` from GCS
2. Split into page chunks (`<!-- page N -->` markers)
3. Create Gemini context cache with full report (tokens paid once)
4. Async parallel extraction (10 concurrent) — 25+ structured fields per chunk via Pydantic schema
5. Retry failed chunks (2 retries with backoff)
6. Upload `_rag_metadata.json` + `_rag_metadata.xlsx` to GCS

```bash
python -m rag_enrichment.orchestrator --source jll --category bengaluru_data --limit 1 --local-output rag_enrichment/output
python -m rag_enrichment.orchestrator --source jll --resume
```

## 6b. Link Assets Script

Matches classified images to RAG metadata records by PDF stem + page number.
Replaces `linked_assets` (which has `image_url: null` from Gemini) with actual GCS image URLs + classification data from Phase 2.

```bash
python link_assets.py --source jll
python link_assets.py --source cushman --dry-run
```

## 7. End-to-End Pipeline Order

```
Step 1: Gemini PDF extraction        → _gemini.md in GCS
        python -m gemini_ocr.pipeline --source jll --workers 3

Step 2: GLM-OCR image extraction     → _extracted_images/ in GCS
        python -m glm_postprocess.orchestrator --source jll --category bengaluru_data --workers 3

Step 3: Image classification         → classification_results.json in GCS
        python -m glm_postprocess.classifier --source jll --category bengaluru_data --workers 20

Step 4: RAG metadata extraction      → _rag_metadata.json + .xlsx in GCS
        python -m rag_enrichment.orchestrator --source jll --category bengaluru_data

Step 5: Link classified images       → updates linked_assets in _rag_metadata.json
        python link_assets.py --source jll
```

## 8. Test Results

### Image Classification (All 4 Sources — Bengaluru Data)

| Source | Folders | Images | Relevant | Not Relevant | Errors | Time |
|---|---|---|---|---|---|---|
| jll | 13 | 367 | 309 (84%) | 58 | 0 | 209s |
| cushman | 31 | 138 | 122 (88%) | 16 | 0 | 229s |
| crematrix | 3 | 19 | 14 (74%) | 5 | 0 | 34s |
| savills | 2 | 18 | 15 (83%) | 3 | 0 | 75s |
| **Total** | **49** | **542** | **460 (85%)** | **82** | **0** | **547s** |

### RAG Enrichment (1 JLL Report Test)

| Metric | Value |
|---|---|
| Pages chunked | 37 |
| Records extracted | 37 (async, 10 concurrent) |
| Failed | 0 (retry recovered 1) |
| Time | 77s |
| Linked images | 72 across 34 pages |
| india-retail-market-dynamics-q1-2025 | 12 | 8 | 4 |
| pulse-real-estate-monthly-monitor-apr-2025 | 37 | 35 | 2 |
| **Total** | **231** | **202 (87%)** | **29** |
