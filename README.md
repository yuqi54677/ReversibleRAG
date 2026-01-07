# ReversibleRAG

## Overview
Multi-hop retrieval-augmented generation pipeline with:
- Question decomposition (leaf/branch/nest) via LLM.
- Sentence-level semantic retrieval with SentenceTransformers.
- Reasoning tree execution (sub-question answering + aggregation).
- Optional verification (LLM-as-judge).
- CSV evaluation (F1 / exact match).

## Structure
- `src/rag/`
  - `config.py` — defaults for paths/models.
  - `data/loader.py` — HotpotQA / MuSiQue / 2Wiki loaders (relative data paths).
  - `llm.py` — OpenAI chat wrapper, decomposition, answering helpers.
  - `retriever.py` — sentence flattening + similarity ranking.
  - `reasoning/tree.py` — reasoning nodes, constructor, executor.
  - `verification.py` — LLM-based verification helpers (Gemini/OpenAI).
  - `metrics.py` — EM/F1 utilities.
  - `scripts/run_rag.py` — run pipeline over a dataset.
  - `scripts/evaluate.py` — evaluate a CSV.
- `data/` — place MuSiQue JSONL files here.
- `outputs/` — CSV run outputs (created automatically).

## Setup
```
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Environment variables:
- `OPENAI_API_KEY` (required for LLM calls)
- `GEMINI_API_KEY` (optional; only for verification helpers)

## Usage
Run pipeline (examples):
```
python -m rag.scripts.run_rag --dataset hotpotqa --num-samples 50
python -m rag.scripts.run_rag --dataset musique --data-dir data --num-samples 20
python -m rag.scripts.run_rag --dataset 2wiki --start-idx 72 --num-samples 28
```
Outputs are written to `outputs/<dataset>_output.csv`.

Evaluate a CSV:
```
python -m rag.scripts.evaluate --csv outputs/hotpotqa_output.csv
```

## Notes
- Decomposition/answering use OpenAI chat (configurable via `--model`).
- Retrieval uses `sentence-transformers/all-MiniLM-L6-v2` (configurable).
- Verification pipeline is available in `rag.verification` but not wired into the executor by default.
```
