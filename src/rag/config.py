from pathlib import Path

# Default paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Model defaults
SENTENCE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_MODEL = "gpt-4o"

# Retrieval
TOP_K_SENTENCES = 8
