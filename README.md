# Brain Tumor Clinical Decision Support Pipeline

An end-to-end agentic pipeline for brain tumor MRI analysis, clinical reasoning, and structured report generation. Built with LangGraph, nnU-Net, BioClinicalBERT, and Llama 3.

## Overview

The system processes brain MRI scans through three phases:

| Phase | Purpose | Components |
|-------|---------|------------|
| **Phase 1** — Imaging | MRI preprocessing, tumor segmentation, radiomics feature extraction, brain region mapping | nnU-Net, PyRadiomics, Harvard-Oxford Atlas |
| **Phase 2** — Intelligence | Tumor classification, similar case retrieval, AI clinical reasoning, report generation | WHO CNS5, RANO criteria, BioClinicalBERT, Llama 3 |
| **Phase 3** — Clinical | Patient memory, physician review, structured CAP report, visualization | ChromaDB, HITL validation, matplotlib |

## Project Structure

```
├── main.py                    # Entry point & CLI
├── pipeline/graph.py          # LangGraph DAG for Phase 1
├── agents/orchestrator.py     # LangGraph DAG for Phase 2 & 3
├── preprocessing/mri_prep.py  # N4 bias correction, skull stripping, normalization
├── segmentation/nnunet_infer.py  # nnU-Net inference + GT fallback
├── radiomics/feature_extractor.py  # PyRadiomics + manual fallback
├── clinical_features/
│   ├── location_mapper.py     # Atlas-based brain region mapping
│   └── symptom_builder.py     # Clinical profile & symptom inference
├── agents/
│   ├── tumor_analysis.py      # WHO/RANO classification via MCP tools
│   ├── similarity_agent.py    # Embedding generation + case retrieval
│   ├── clinical_reasoning.py  # Llama 3 reasoning (or rule-based fallback)
│   └── report_agent.py        # JSON report consolidation
├── embeddings/generator.py    # BioClinicalBERT embeddings
├── similarity/vector_store.py # Weaviate + NumPy cosine fallback
├── mcp_servers/               # MCP tool servers
│   ├── who_classification.py  # WHO CNS5 tumor grading
│   ├── rano_criteria.py       # RANO response assessment
│   └── cap_report.py          # CAP structured pathology report
├── memory/patient_memory.py   # ChromaDB patient history
├── validation/hitl.py         # Human-in-the-loop review
├── visualization/viewer.py    # 5-panel MRI visualization
├── evaluation/                # Dice, Hausdorff, ICC, Precision@K metrics
├── utils/h5_adapter.py        # BraTS H5 → 3D volume reconstruction
├── Dockerfile
└── docker-compose.yml
```

## Quick Start with Docker Compose

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- At least **8 GB RAM** allocated to Docker (Llama 3 + pipeline)
- The BraTS2020 dataset already present in `BraTS2020_training_data/`

### Run the Pipeline

```bash
# Start all services (Ollama, Weaviate, pipeline) and process 1 patient
docker compose up --build

# Process more patients (edit max_patients in docker-compose.yml or override)
docker compose run --rm pipeline \
  --data_dir /app/BraTS2020_training_data/content/data \
  --format h5 --phase all --output_dir /app/outputs \
  --skip_hitl --max_patients 5

# Run only Phase 1 (imaging — no Ollama/Weaviate needed)
docker compose run --rm pipeline \
  --data_dir /app/BraTS2020_training_data/content/data \
  --format h5 --phase 1 --output_dir /app/outputs \
  --max_patients 1

# Shut down all services
docker compose down
```

### Run with Evaluation

```bash
docker compose run --rm pipeline \
  --data_dir /app/BraTS2020_training_data/content/data \
  --format h5 --phase all --output_dir /app/outputs \
  --skip_hitl --max_patients 1 --evaluate
```

## Local Setup (without Docker)

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start External Services

```bash
# Ollama (for Llama 3 clinical reasoning)
brew install ollama
ollama serve &
ollama pull llama3

# Weaviate (optional — falls back to NumPy cosine similarity)
docker run -d -p 8080:8080 semitechnologies/weaviate:1.28.2
```

### Run the Pipeline

```bash
# Full pipeline on 1 patient
python main.py \
  --data_dir ./BraTS2020_training_data/content/data \
  --format h5 --phase all --output_dir ./outputs \
  --skip_hitl --max_patients 1

# Single patient by ID
python main.py \
  --data_dir ./BraTS2020_training_data/content/data \
  --format h5 --phase all --output_dir ./outputs \
  --skip_hitl --patient_id volume_1

# NIfTI format
python main.py \
  --data_dir /path/to/nifti_patients \
  --format nifti --phase all --output_dir ./outputs

# Custom Ollama URL
python main.py \
  --data_dir ./BraTS2020_training_data/content/data \
  --format h5 --phase all --output_dir ./outputs \
  --ollama_url http://192.168.1.5:11434
```

## CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | *required* | Path to dataset (NIfTI folders or H5 slices) |
| `--output_dir` | `./outputs` | Output directory for all results |
| `--format` | `h5` | Dataset format: `h5` or `nifti` |
| `--phase` | `all` | Pipeline phase: `1`, `2`, `3`, `23`, or `all` |
| `--patient_id` | `None` | Process a single patient (NIfTI mode) |
| `--max_patients` | `None` | Limit number of patients (H5 mode) |
| `--skip_hitl` | `False` | Auto-approve physician validation |
| `--ollama_url` | `http://localhost:11434` | Ollama API endpoint |
| `--evaluate` | `False` | Run evaluation metrics after pipeline |

## Output Structure

```
outputs/
├── preprocessed/          # Bias-corrected, normalized .npy volumes
├── segmentation/          # Binary tumor masks (.nii.gz)
├── radiomics/             # Shape, intensity, texture features (.json)
├── clinical_features/     # Clinical profiles with symptoms (.json)
├── embeddings/            # BioClinicalBERT 768-dim vectors (.npy)
├── reports/
│   ├── *_report.json      # Full analysis reports
│   └── cap/               # CAP structured pathology reports
├── memory/                # Patient scan history
├── visualizations/        # MRI slices, overlays, 3D renders (.png)
├── chromadb/              # Persistent patient memory DB
└── evaluation/            # Dice, Hausdorff, ICC metrics (.json)
```

## Models Used

| Model | Purpose | Source | Fallback |
|-------|---------|--------|----------|
| nnU-Net (BraTS2021) | Tumor segmentation | Zenodo (auto-downloaded) | Ground-truth BraTS mask |
| BioClinicalBERT | Clinical text embeddings | HuggingFace | Manual feature vector |
| Llama 3 | Clinical reasoning | Ollama (local) | Rule-based text generation |
| Harvard-Oxford Atlas | Brain region mapping | nilearn | Heuristic geometry |

## Notes

- **No training involved** — all models are used for inference only. The BraTS2020 dataset serves as evaluation/demo data.
- **Graceful degradation** — every component has a fallback. The pipeline runs without GPU, Ollama, or Weaviate, but output quality decreases.
- **macOS compatible** — runs natively or via Docker Desktop. No CUDA required (CPU inference throughout).
- **HITL validation** — use `--skip_hitl` for batch/automated runs. Without it, the pipeline prompts for physician review in the terminal.
