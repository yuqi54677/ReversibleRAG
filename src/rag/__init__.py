"""
RAG package for multi-hop QA.

Modules:
- data.loader: dataset loading utilities
- llm: LLM wrappers and QA prompts
- retriever: sentence-level semantic retrieval
- reasoning.tree: reasoning tree construction/execution
- verification: LLM-as-judge scoring helpers
- metrics: evaluation helpers
"""

__all__ = [
    "data",
    "llm",
    "retriever",
    "reasoning",
    "verification",
    "metrics",
]
