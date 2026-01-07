import argparse
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from rag import config
from rag.data.loader import load_dataset_by_name
from rag.llm import LLM
from rag.reasoning.tree import TreeConstructor, TreeExecutor
from rag.retriever import Retriever


def ensure_output_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=["question", "expected_answer", "answer"])
    df.to_csv(path, index=False)
    return df


def run(args):
    dataset = load_dataset_by_name(
        args.dataset, base_path=args.data_dir, start_idx=args.start_idx, num_samples=args.num_samples
    )
    llm = LLM(model_name=args.model)
    stmodel = SentenceTransformer(args.sentence_model)
    retriever = Retriever(stmodel, args.dataset)

    output_path = args.output_dir / f"{args.dataset}_output.csv"
    df = ensure_output_csv(output_path)
    idx = len(df)

    for q in dataset:
        question = q["question"]
        expected_answer = q.get("answer", "")
        flat_sentences = retriever.flatten_context_sentences(q)

        constructor = TreeConstructor(llm)
        tree_root = constructor.build_tree(question)

        executor = TreeExecutor(retriever, llm, top_k=args.top_k)
        final_answer = executor.answer_tree(tree_root, flat_sentences)

        df.loc[idx] = [question, expected_answer, final_answer]
        df.to_csv(output_path, index=False)
        idx += 1

        print(f"[{idx}] Q: {question}")
        print(f"Expected: {expected_answer}")
        print(f"Answer:   {final_answer}\n")
        time.sleep(args.delay)


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi-hop RAG pipeline.")
    parser.add_argument("--dataset", choices=["hotpotqa", "musique", "2wiki"], required=True)
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR, help="Base data directory (for musique).")
    parser.add_argument("--output-dir", type=Path, default=config.OUTPUT_DIR, help="Directory for CSV outputs.")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=-1, help="-1 to use full split")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_SENTENCES)
    parser.add_argument("--model", type=str, default=config.OPENAI_MODEL, help="OpenAI chat model name")
    parser.add_argument(
        "--sentence-model",
        type=str,
        default=config.SENTENCE_MODEL_NAME,
        help="SentenceTransformer model for retrieval",
    )
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to sleep between examples")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
