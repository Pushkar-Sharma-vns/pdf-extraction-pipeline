## GLM-OCR

[中文阅读](README_zh.md)

<div align="center">
<img src=resources/logo.svg width="40%"/>
</div>
<p align="center">
    👋 Join our <a href="resources/WECHAT.md" target="_blank">WeChat</a> and <a href="https://discord.gg/QR7SARHRxK" target="_blank">Discord</a> community
    <br>
    📖 Check out the GLM-OCR <a href="https://arxiv.org/abs/2603.10910" target="_blank">technical report</a>
    <br>
    📍 Use GLM-OCR's <a href="https://docs.z.ai/guides/vlm/glm-ocr" target="_blank">API</a>
</p>

### Model Introduction

GLM-OCR is a multimodal OCR model for complex document understanding, built on the GLM-V encoder–decoder architecture. It introduces Multi-Token Prediction (MTP) loss and stable full-task reinforcement learning to improve training efficiency, recognition accuracy, and generalization. The model integrates the CogViT visual encoder pre-trained on large-scale image–text data, a lightweight cross-modal connector with efficient token downsampling, and a GLM-0.5B language decoder. Combined with a two-stage pipeline of layout analysis and parallel recognition based on PP-DocLayout-V3, GLM-OCR delivers robust and high-quality OCR performance across diverse document layouts.

**Key Features**

- **State-of-the-Art Performance**: Achieves a score of 94.62 on OmniDocBench V1.5, ranking #1 overall, and delivers state-of-the-art results across major document understanding benchmarks, including formula recognition, table recognition, and information extraction.

- **Optimized for Real-World Scenarios**: Designed and optimized for practical business use cases, maintaining robust performance on complex tables, code-heavy documents, seals, and other challenging real-world layouts.

- **Efficient Inference**: With only 0.9B parameters, GLM-OCR supports deployment via vLLM, SGLang, and Ollama, significantly reducing inference latency and compute cost, making it ideal for high-concurrency services and edge deployments.

- **Easy to Use**: Fully open-sourced and equipped with a comprehensive [SDK](https://github.com/zai-org/GLM-OCR) and inference toolchain, offering simple installation, one-line invocation, and smooth integration into existing production pipelines.

### News & Updates

- **[2026.3.12]** GLM-OCR SDK now supports agent-friendly Skill mode — just `pip install glmocr` + set API key, ready to use via CLI or Python with no GPU or YAML config needed. See: [GLM-OCR Skill](skills/glmocr/SKILL.md)
- **[2026.3.12]** GLM-OCR Technical Report is now available. See: [GLM-OCR Technical Report](https://arxiv.org/abs/2603.10910)
- **[2026.2.12]** Fine-tuning tutorial based on LLaMA-Factory is now available. See: [GLM-OCR Fine-tuning Guide](examples/finetune/README.md)

### Download Model

| Model   | Download Links                                                                                                              | Precision |
| ------- | --------------------------------------------------------------------------------------------------------------------------- | --------- |
| GLM-OCR | [🤗 Hugging Face](https://huggingface.co/zai-org/GLM-OCR)<br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/GLM-OCR) | BF16      |

## GLM-OCR SDK

We provide an SDK for using GLM-OCR more efficiently and conveniently.

### Install SDK

Choose the lightest installation that matches your scenario:

```bash
# Cloud / MaaS + local images / PDFs (fastest install)
pip install glmocr

# Self-hosted pipeline (layout detection)
pip install "glmocr[selfhosted]"

# Flask service support
pip install "glmocr[server]"
```

Install from source for development:

```bash
# Install from source
git clone https://github.com/zai-org/glm-ocr.git
cd glm-ocr
uv venv --python 3.12 --seed && source .venv/bin/activate
uv pip install -e .
```

### Model Deployment

Two ways to use GLM-OCR:

#### Option 1: Zhipu MaaS API (Recommended for Quick Start)

Use the hosted cloud API – no GPU needed. The cloud service runs the complete GLM-OCR pipeline internally, so the SDK simply forwards your request and returns the result.

1. Get an API key from https://open.bigmodel.cn
2. Configure `config.yaml`:

```yaml
pipeline:
  maas:
    enabled: true # Enable MaaS mode
    api_key: your-api-key # Required
```

That's it! When `maas.enabled=true`, the SDK acts as a thin wrapper that:

- Forwards your documents to the Zhipu cloud API
- Returns the results directly (Markdown + JSON layout details)
- No local processing, no GPU required

Input note (MaaS): the upstream API accepts `file` as a URL or a `data:<mime>;base64,...` data URI.
If you have raw base64 without the `data:` prefix, wrap it as a data URI (recommended). The SDK will
auto-wrap local file paths / bytes / raw base64 into a data URI when calling MaaS.

API documentation: https://docs.bigmodel.cn/cn/guide/models/vlm/glm-ocr

#### Option 2: Self-host with vLLM / SGLang

Deploy the GLM-OCR model locally for full control. The SDK provides the complete pipeline: layout detection, parallel region OCR, and result formatting.

Install the self-hosted extra first:

```bash
pip install "glmocr[selfhosted]"
```

##### Using vLLM

Install vLLM:

```bash
docker pull vllm/vllm-openai:v0.19.0-ubuntu2404
```

Or using with pip:

```bash
pip install -U "vllm>=0.19.0"
```

Launch the service:

```bash
pip install "transformers>=5.3.0"

vllm serve zai-org/GLM-OCR  --port 8080 --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' --served-model-name glm-ocr
```

>Note
  Add `--max-model-len` and `--gpu-memory-utilization` according to Your own machine to handle large image/pdf

##### Using SGLang

Install SGLang:

```bash
docker pull lmsysorg/sglang:v0.5.10
```

Or using with pip:

```bash
pip install "sglang>=0.5.10"
```

Launch the service:

```bash
SGLANG_ENABLE_SPEC_V2=1 sglang serve --model-path zai-org/GLM-OCR --port 8080 --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --served-model-name glm-ocr
```

>Note
  Add `--context-len` and `--mem-fraction-static` according to Your own machine to handle large image/pdf


#### Option 3: Ollama/MLX

For specialized deployment scenarios, see the detailed guides:

- **[Apple Silicon with mlx-vlm](examples/mlx-deploy/README.md)** - Optimized for Apple Silicon Macs
- **[Ollama Deployment](examples/ollama-deploy/README.md)** - Simple local deployment with Ollama

#### Option 4: SDK Server + Client (GPU-less Client)

Deploy the SDK Server on a GPU machine, then use any machine as a client — no GPU needed on the client side. The client connects via the MaaS-compatible protocol, pointing `api_url` at your self-hosted server.

```yaml
# Client config.yaml
pipeline:
  maas:
    enabled: true
    api_url: http://<SERVER_IP>:5002/glmocr/parse
    api_key: any-string    # self-hosted server does not validate keys
    verify_ssl: false
```

See the full guide: **[Self-hosted SDK Server + Client](examples/self-host/README.md)**

#### Update Configuration

After launching the service, configure `config.yaml`:

```yaml
pipeline:
  maas:
    enabled: false # Disable MaaS mode (default)
  ocr_api:
    api_host: localhost # or your vLLM/SGLang server address
    api_port: 8080
```

### SDK Usage Guide

#### CLI

```bash
# Parse a single image
glmocr parse examples/source/code.png

# Parse a directory
glmocr parse examples/source/

# Set output directory
glmocr parse examples/source/code.png --output ./results/

# Use a custom config
glmocr parse examples/source/code.png --config my_config.yaml

# Enable debug logging with profiling
glmocr parse examples/source/code.png --log-level DEBUG

# Run layout detection on CPU (keep GPU free for OCR model)
glmocr parse examples/source/code.png --layout-device cpu

# Run layout detection on a specific GPU
glmocr parse examples/source/code.png --layout-device cuda:1

# Override any config value via --set (dotted path, repeatable)
glmocr parse examples/source/code.png --set pipeline.ocr_api.api_port 8080
glmocr parse examples/source/ --set pipeline.layout.use_polygon true --set logging.level DEBUG
```

#### Python API

```python
from glmocr import GlmOcr, parse

# Simple function
result = parse("image.png")
result = parse(["img1.png", "img2.jpg"])
result = parse("https://example.com/image.png")
result.save(output_dir="./results")

# Note: a list is treated as pages of a single document.

# Class-based API
with GlmOcr() as parser:
    result = parser.parse("image.png")
    print(result.json_result)
    result.save()

# Place layout model on CPU (useful when GPU is reserved for OCR)
with GlmOcr(layout_device="cpu") as parser:
    result = parser.parse("image.png")

# Place layout model on a specific GPU
with GlmOcr(layout_device="cuda:1") as parser:
    result = parser.parse("image.png")
```

#### Flask Service

Install the optional server dependency first:

```bash
pip install "glmocr[server]"
```

```bash
# Start service
python -m glmocr.server

# With debug logging
python -m glmocr.server --log-level DEBUG

# Call API
curl -X POST http://localhost:5002/glmocr/parse \
  -H "Content-Type: application/json" \
  -d '{"images": ["./example/source/code.png"]}'
```

Semantics:

- `images` can be a string or a list.
- A list is treated as pages of a single document.
- For multiple independent documents, call the endpoint multiple times (one document per request).


### Modular Architecture

GLM-OCR uses composable modules for easy customization:

| Component             | Description                            |
| --------------------- | -------------------------------------- |
| `PageLoader`          | Preprocessing and image encoding       |
| `OCRClient`           | Calls the GLM-OCR model service        |
| `PPDocLayoutDetector` | PP-DocLayout layout detection          |
| `ResultFormatter`     | Post-processing, outputs JSON/Markdown |

You can extend the behavior by creating custom pipelines:

```python
from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.postprocess import ResultFormatter


class MyPipeline:
  def __init__(self, config):
    self.page_loader = PageLoader(config)
    self.ocr_client = OCRClient(config)
    self.formatter = ResultFormatter(config)

  def process(self, request_data):
    # Implement your own processing logic
    pass
```

## Custom Pipelines

### Gemini OCR Pipeline (`gemini_ocr/`)

Batch PDF extraction using Google Gemini, with GCS bucket integration and Bengaluru/India classification.

#### Setup

```bash
pip install google-cloud-storage google-genai pymupdf
```

Configure `.env` at the project root:

```env
GEMINI_API_KEY=your-gemini-api-key
GCS_CREDENTIALS=portal-data-bucket.json
GCS_BUCKET=re_reports
```

#### Commands

```bash
# Dry run — list PDFs without processing
./venv/bin/python -m gemini_ocr.pipeline --dry-run

# Process one source
./venv/bin/python -m gemini_ocr.pipeline --source jll

# Process a single PDF
./venv/bin/python -m gemini_ocr.pipeline --blob jll/pdfs/report.pdf

# Parallel workers (3x faster)
./venv/bin/python -m gemini_ocr.pipeline --source cushman --workers 3

# Resume interrupted batch (skips already-uploaded PDFs)
./venv/bin/python -m gemini_ocr.pipeline --source cushman --workers 3 --resume

# Classify PDFs only (bengaluru vs india, no extraction, free)
./venv/bin/python -m gemini_ocr.pipeline --source jll --classify-only
```

Output lands at: `gs://<bucket>/<source>/extracted_pdfs/{bengaluru_data|india_data}/<pdfname>_gemini.md`

---

### GLM-OCR Post-Processing Pipeline (`glm_postprocess/`)

End-to-end pipeline: GLM-OCR parse → image extraction (polygon-based cropping) → GCS upload → image classification.

#### Backend Configuration (`config.yaml`)

GLM-OCR needs an inference backend for text recognition. Two options:

**vLLM (recommended — faster, supports batching):**

```yaml
# config.yaml
pipeline:
  maas:
    enabled: false
  layout:
    model_dir: 'PaddlePaddle/PP-DocLayoutV3_safetensors'
  ocr_api:
    api_host: localhost       # or EC2 IP for remote GPU
    api_port: 8080
    api_path: /v1/chat/completions
    model: glm-ocr
    api_mode: openai
```

Start vLLM (on GPU machine):

```bash
pip install "vllm>=0.19.0" "transformers>=5.3.0"

vllm serve zai-org/GLM-OCR \
  --port 8080 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
  --served-model-name glm-ocr \
  --gpu-memory-utilization 0.9
```

**Ollama (fallback — simpler setup, slower):**

```yaml
# config.yaml (or copy config_ollama.yaml → config.yaml)
pipeline:
  maas:
    enabled: false
  layout:
    model_dir: 'PaddlePaddle/PP-DocLayoutV3_safetensors'
  ocr_api:
    api_host: localhost
    api_port: 11434
    api_path: /api/generate
    model: glm-ocr:latest
    api_mode: ollama_generate
```

Start Ollama with parallel support:

```bash
OLLAMA_NUM_PARALLEL=4 ollama serve
ollama run glm-ocr
```

#### Phase 1: Extract Images (Local Mode)

Process existing GLM-OCR results in `test_results/`:

```bash
./venv/bin/python -m glm_postprocess.orchestrator --local --min-size 100
```

#### Phase 1: Extract Images (GCS Mode)

Reads already Gemini-extracted PDFs from GCS, downloads the original PDF, runs GLM-OCR, extracts images, uploads back:

```bash
# Process all PDFs from a source (parallel with vLLM)
./venv/bin/python -m glm_postprocess.orchestrator --source jll --workers 8

# Resume (skip PDFs with existing images in GCS)
./venv/bin/python -m glm_postprocess.orchestrator --source savills --resume --workers 8

# Dry run — list what would be processed
./venv/bin/python -m glm_postprocess.orchestrator --source savills --dry-run
```

Output: `gs://<bucket>/<source>/extracted_pdfs/{bengaluru_data|india_data}/<pdfname>_extracted_images/`

Image naming: `page<N>_<label>_idx<I>.jpg` (1-based page, label type, item index from GLM-OCR JSON).

#### Phase 2: Classify Images

Async Gemini classification — determines if each extracted image is relevant to real estate. Uses 20 concurrent async workers via `asyncio.Semaphore`.

```bash
# Run Phase 2 standalone on already-uploaded images (bengaluru only)
./venv/bin/python -m glm_postprocess.classifier --source jll --category bengaluru_data --workers 20

# Or chain Phase 1 + Phase 2 in one command
./venv/bin/python -m glm_postprocess.orchestrator --source jll --category bengaluru_data --workers 8 --classify

# Classify-only mode (no extraction)
./venv/bin/python -m glm_postprocess.orchestrator --classify-only --source jll --category bengaluru_data

# Resume (skip folders that already have classification_results.json)
./venv/bin/python -m glm_postprocess.classifier --source jll --category bengaluru_data --resume

# Classify a single folder directly
./venv/bin/python -m glm_postprocess.classifier --folder jll/extracted_pdfs/bengaluru_data/report_extracted_images
```

Output: `classification_results.json` saved in the same GCS folder as images.

Features: parallel GCS image download (threaded), retry with backoff (2 retries per image), per-folder timing, log file at `glm_postprocess/output/classify_log_<source>_<category>.txt`.

---

### RAG Enrichment Pipeline (`rag_enrichment/`)

Page-wise chunking of Gemini-extracted markdown + contextual metadata extraction for Qdrant RAG.

For each `_gemini.md` in GCS:
1. Split into page-wise chunks using `<!-- page N -->` markers
2. Create Gemini context cache with full document (pay input tokens once)
3. Extract 25+ structured metadata fields per chunk via async parallel Gemini calls (10 concurrent, with retry)
4. Upload `_rag_metadata.json` + `_rag_metadata.xlsx` to GCS

Metadata includes: `contextual_situation`, `source_publisher`, `report_type`, `city`, `zone`, `micro_market`, `asset_class`, `economic_lens`, `content_intent`, `linked_assets`, and 15+ more strict enum fields.

```bash
# Process 1 report (test)
./venv/bin/python -m rag_enrichment.orchestrator --source jll --limit 1 --local-output rag_enrichment/output

# All bengaluru_data for a source
./venv/bin/python -m rag_enrichment.orchestrator --source jll --category bengaluru_data

# Resume interrupted run
./venv/bin/python -m rag_enrichment.orchestrator --source jll --resume

# Dry run
./venv/bin/python -m rag_enrichment.orchestrator --source jll --dry-run
```

Output: `gs://<bucket>/<source>/extracted_pdfs/{bengaluru_data|india_data}/<pdfname>_rag_metadata.json` and `_rag_metadata.xlsx`

Log: `rag_enrichment/output/rag_log_<source>_<category>.txt`

---

### Link Assets (`link_assets.py`)

Matches classified images to RAG metadata records by PDF stem + page number, and replaces `linked_assets` with actual GCS image URLs + classification data.

Requires both `_rag_metadata.json` (from RAG enrichment) and `classification_results.json` (from image classifier) to exist for a PDF.

```bash
# Link images for a source
./venv/bin/python link_assets.py --source jll

# Dry run — list which PDFs can be linked
./venv/bin/python link_assets.py --source jll --dry-run

# Limit to first N
./venv/bin/python link_assets.py --source cushman --limit 5
```

Log: `link_assets_log_<source>_<category>.txt`

---

## Star History

<a href="https://www.star-history.com/?repos=zai-org%2FGLM-OCR&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/image?repos=zai-org/GLM-OCR&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/image?repos=zai-org/GLM-OCR&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/image?repos=zai-org/GLM-OCR&type=date&legend=top-left" />
 </picture>
</a>

## Acknowledgement

This project is inspired by the excellent work of the following projects and communities:

- [PP-DocLayout-V3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [MinerU](https://github.com/opendatalab/MinerU)

## License

The Code of this repo is under Apache License 2.0.

The GLM-OCR model is released under the MIT License.

The complete OCR pipeline integrates [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3) for document layout analysis, which is licensed under the Apache License 2.0. Users should comply with both licenses when using this project.

## Citation

If you find GLM-OCR useful in your research, please cite our technical report:

```bibtex
@misc{duan2026glmocrtechnicalreport,
      title={GLM-OCR Technical Report},
      author={Shuaiqi Duan and Yadong Xue and Weihan Wang and Zhe Su and Huan Liu and Sheng Yang and Guobing Gan and Guo Wang and Zihan Wang and Shengdong Yan and Dexin Jin and Yuxuan Zhang and Guohong Wen and Yanfeng Wang and Yutao Zhang and Xiaohan Zhang and Wenyi Hong and Yukuo Cen and Da Yin and Bin Chen and Wenmeng Yu and Xiaotao Gu and Jie Tang},
      year={2026},
      eprint={2603.10910},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.10910},
}
```
