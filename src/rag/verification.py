import os
import textwrap
from typing import Any, Dict, List

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)


class GeminiChatAdapter:
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self._model = genai.GenerativeModel(model_name)
        self.chat = self._Chat(self)

    class _Chat:
        def __init__(self, outer: "GeminiChatAdapter"):
            self.completions = outer._Completions(outer)

    class _Completions:
        def __init__(self, outer: "GeminiChatAdapter"):
            self.outer = outer

        def create(self, model: str, messages: List[Dict[str, str]], max_tokens: int = 10, temperature: float = 0.0):
            parts = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                prefix = "System: " if role == "system" else "User: "
                parts.append(prefix + content)
            prompt = "\n\n".join(parts)

            resp = self.outer._model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            text = ""
            if getattr(resp, "candidates", None):
                cand = resp.candidates[0]
                content = getattr(cand, "content", None)
                if content:
                    cparts = getattr(content, "parts", None) or []
                    chunks = []
                    for p in cparts:
                        t = getattr(p, "text", "")
                        if t:
                            chunks.append(t)
                    text = "".join(chunks).strip()

            class _Message:
                def __init__(self, content):
                    self.content = content

            class _Choice:
                def __init__(self, message):
                    self.message = message

            class _Response:
                def __init__(self, choices):
                    self.choices = choices

            return _Response([_Choice(_Message(text))])


class Evaluator:
    def __init__(self, client):
        self.client = client

    def score_accuracy(self, question: str, answer: str, evidences: List[str]) -> float:
        evidence_text = "\n\n".join([f"[Evidence {i+1}]\n{ev}" for i, ev in enumerate(evidences)])
        system_prompt = textwrap.dedent(
            """
			You are an accuracy evaluator. Score the answer from 0.0 to 1.0 based on support.
			0.8–1.0: strongly supported; 0.5–0.79: partially; 0.2–0.49: weak; 0–0.19: wrong/unsupported.
			Return only the number.
			"""
        )
        user_prompt = f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidences:\n{evidence_text}\n\nWhat is the score?"
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        score_str = response.choices[0].message.content.strip()
        try:
            score = float(score_str)
        except Exception:
            import re

            match = re.search(r"0\.\d+|1\.0|1|0", score_str)
            score = float(match.group(0)) if match else 0.0
        return max(0.0, min(score, 1.0))


class AttrScoreModel:
    def __init__(self, client, model_name: str = "gpt-4o-mini"):
        self.client = client
        self.model_name = model_name

    def score_attr(self, question: str, answer: str, evidences: List[str]) -> str:
        claim = f"Question: {question}\nAnswer: {answer}"
        reference = "\n".join(f"- {e}" for e in evidences)
        system_prompt = textwrap.dedent(
            """
			You are an Attribution Validator.
			Classify the claim vs reference as: Attributable, Extrapolatory, or Contradictory.
			Respond with exactly one word.
			"""
        )
        user_prompt = f"Claim:\n{claim}\n\nReference:\n{reference}\n\nWhat is the relationship?"
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,
            max_tokens=30,
        )
        raw = response.choices[0].message.content.strip()
        label = raw.split()[0].lower()
        if "attrib" in label:
            return "Attributable"
        if "contrad" in label:
            return "Contradictory"
        if "extra" in label:
            return "Extrapolatory"
        return "Extrapolatory"


class Verifier:
    def __init__(self, llm_judge: Evaluator, attr_model: AttrScoreModel, threshold: float = 0.75):
        self.llm_judge = llm_judge
        self.attr_model = attr_model
        self.threshold = threshold

    @staticmethod
    def _map_attr_label_to_score(label: str) -> float:
        label = label.strip().lower()
        if label == "attributable":
            return 1.0
        if label == "contradictory":
            return 0.0
        if label == "extrapolatory":
            return 0.5
        return 0.0

    def verify(self, question: str, answer: str, evidences: List[str]) -> Dict[str, Any]:
        accuracy = self.llm_judge.score_accuracy(question, answer, evidences)
        attr_label = self.attr_model.score_attr(question, answer, evidences)
        credibility = self._map_attr_label_to_score(attr_label)
        final_score = (accuracy ** 0.6) * (credibility ** 0.4)
        confidence_score = 1 if final_score >= self.threshold else 0
        return {
            "accuracy": accuracy,
            "credibility": credibility,
            "attr_label": attr_label,
            "final_score": final_score,
            "confidence_score": confidence_score,
        }


def run_full_verifier_pipeline(question: str, answer: str, evidence: str) -> Dict[str, Any]:
    client = GeminiChatAdapter()
    llm_judge = Evaluator(client)
    attr_model = AttrScoreModel(client)
    verifier = Verifier(llm_judge, attr_model, threshold=0.75)
    evidences = [evidence] if evidence is not None else []
    result = verifier.verify(question, answer, evidences)
    return {
        "pred_label": result["attr_label"],
        "confidence_score": result["confidence_score"],
        "accuracy": result["accuracy"],
        "credibility": result["credibility"],
        "final_score": result["final_score"],
    }
