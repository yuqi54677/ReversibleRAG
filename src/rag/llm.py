import json
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def extract_json(text: str, fallback=None):
    if text is None:
        return fallback

    # Try fenced JSON blocks
    codeblock_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if not codeblock_match:
        codeblock_match = re.search(r"```(.*?\{.*?\}.*?)```", text, flags=re.DOTALL)

    if codeblock_match:
        json_part = codeblock_match.group(1).strip()
        try:
            return json.loads(json_part)
        except Exception:
            pass

    # First JSON object in text
    brace_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if brace_match:
        json_part = brace_match.group(0)
        try:
            return json.loads(json_part)
        except Exception:
            pass

    # Direct parse
    try:
        return json.loads(text.strip())
    except Exception:
        pass

    return fallback if fallback is not None else {"parse_error": True, "raw_text": text}


def normalize_outer_template(raw: str, placeholder: str = "{inner}") -> str:
    if not raw:
        return placeholder

    raw = raw.replace("[ANSWER]", placeholder)
    if placeholder in raw:
        return raw

    formatter = string.Formatter()
    field_names = [fname for _, fname, _, _ in formatter.parse(raw) if fname]
    if field_names:
        raw = raw.replace("{" + field_names[0] + "}", placeholder)
    if "{}" in raw:
        raw = raw.replace("{}", placeholder)
    if placeholder not in raw:
        raw = raw + f" {placeholder}"
    return raw


def parse_subquestions(response: str) -> List[str]:
    subquestions: List[str] = []
    lines = response.strip().split("\n")
    for line in lines:
        if not line.strip():
            continue
        line_lower = line.strip().lower()
        if (
            line_lower.startswith("subquestion")
            or line_lower.startswith("question")
            or line_lower.startswith("here are")
            or line_lower.startswith("breakdown")
            or line_lower == "subquestions:"
            or line_lower == "questions:"
        ):
            continue
        cleaned = re.sub(r"^[\d\w]+[\.\)]\s*", "", line.strip())
        cleaned = re.sub(r"^[-•*]\s*", "", cleaned)
        if len(cleaned) < 5:
            continue
        subquestions.append(cleaned)
    return subquestions


@dataclass
class LLM:
    model_name: str = "gpt-4o"
    client: Optional[OpenAI] = None

    def __post_init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = self.client or OpenAI(api_key=api_key)

    def chat(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[LLM_CALL_FAILED] Error: {e}\nPrompt was:\n{prompt}"

    def decompose_question(self, question: str) -> Dict[str, Any]:
        system_prompt = (
            "You are a Question Planning module for multi-hop QA. "
            "Classify the question as 'leaf', 'branch', or 'nest'. "
            "Return JSON only."
        )
        schema = """
Return a JSON object with:
- type: one of ["leaf","branch","nest"]
- If type='branch': sub_questions: list[str]
- If type='nest': inner_question: str, outer_template: str with placeholder {inner}
"""
        examples = """
Example leaf:
Input: Who wrote Pride and Prejudice?
{ "type": "leaf" }

Example branch:
Input: Which singer was born in X and won the Y award?
{ "type": "branch", "sub_questions": ["Which singer won the Y award?", "Which singer was born in X?"] }

Example nest:
Input: What is the population of the city where Albert Einstein was born?
{ "type": "nest", "inner_question": "Where was Albert Einstein born?", "outer_template": "What is the population of {inner}?" }
"""
        prompt = f"{system_prompt}\n\nSchema:\n{schema}\n{examples}\n\nQuestion:\n{question}\n\nReturn only JSON."
        raw = self.chat(prompt)
        extracted = extract_json(raw, fallback={})
        if extracted.get("type") == "nest":
            outer_template = extracted.get("outer_template", "")
            extracted["outer_template"] = normalize_outer_template(outer_template)
        return extracted

    def answer_subquestions(self, subquestion: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_blocks = []
        for i, doc in enumerate(retrieved_docs, start=1):
            evidence_blocks.append(f"[Doc {i}]\n{doc.get('text', doc)}")
        evidence_text = "\n\n".join(evidence_blocks)
        prompt = f"""
You are a concise, evidence-based assistant.
Subquestion:
{subquestion}

Evidence:
{evidence_text}

Rules:
- Use only evidence above.
- If insufficient, answer "unknown".
- If yes/no style, answer only "yes" or "no".
- Provide brief reasoning and cite doc numbers.

Respond in JSON:
{{"answer": str, "reasoning": str, "supporting_docs": [int,...]}}
"""
        raw_output = self.chat(prompt)
        parsed_output = extract_json(raw_output, fallback={})
        if "answer" not in parsed_output:
            parsed_output["answer"] = "unknown"
        if "reasoning" not in parsed_output:
            parsed_output["reasoning"] = ""
        if "supporting_docs" not in parsed_output or not isinstance(parsed_output["supporting_docs"], list):
            parsed_output["supporting_docs"] = []
        return parsed_output
