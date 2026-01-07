import csv
import re
from typing import Tuple

import pandas as pd


def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        return re.sub(r"[^\w\s]", "", text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(truth_tokens)
    num_same = sum(min(pred_tokens.count(tok), truth_tokens.count(tok)) for tok in common)

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 1.0 if pred_tokens == truth_tokens else 0.0
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match_score(path: str) -> float:
    df = pd.read_csv(path)
    count = 0
    for _, row in df.iterrows():
        normalized_prediction = normalize_answer(str(row["answer"]))
        normalized_ground_truth = normalize_answer(str(row["expected_answer"]))
        if normalized_prediction == normalized_ground_truth:
            count += 1
    score = count / len(df)
    return score


def compute_avg_f1(csv_file_path: str) -> Tuple[int, float]:
    total = 0
    f1_sum = 0.0

    with open(csv_file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            expected = row["expected_answer"]
            actual = row["answer"]
            f1 = f1_score(actual, expected)
            f1_sum += f1
            total += 1

    avg_f1 = f1_sum / total if total > 0 else 0.0
    return total, avg_f1
