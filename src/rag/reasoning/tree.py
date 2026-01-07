from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from rag.llm import LLM
from rag.retriever import Retriever


class NodeKind(str, Enum):
    ROOT = "ROOT"
    BRANCH = "BRANCH"
    NEST = "NEST"
    LEAF = "LEAF"


@dataclass
class ReasoningNode:
    id: str
    question: str
    kind: NodeKind
    parent: Optional["ReasoningNode"] = None
    children: List["ReasoningNode"] = field(default_factory=list)
    answer: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "ReasoningNode") -> None:
        child.parent = self
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0 and self.kind == NodeKind.LEAF


class TreeConstructor:
    def __init__(self, llm: LLM):
        self.llm = llm
        self._next_id = 0

    def _new_id(self) -> str:
        self._next_id += 1
        return f"n{self._next_id}"

    def build_tree(self, root_question: str) -> ReasoningNode:
        root = ReasoningNode(id=self._new_id(), question=root_question, kind=NodeKind.ROOT)
        self._expand_node(root)
        return root

    def _expand_node(self, node: ReasoningNode) -> None:
        plan = self.llm.decompose_question(node.question)

        if plan.get("type") == "leaf":
            node.kind = NodeKind.LEAF
            return

        if plan.get("type") == "branch":
            node.kind = NodeKind.BRANCH
            for sub_q in plan.get("sub_questions", []):
                child = ReasoningNode(
                    id=self._new_id(),
                    question=sub_q,
                    kind=NodeKind.LEAF,
                )
                node.add_child(child)
                self._expand_node(child)
            return

        if plan.get("type") == "nest":
            node.kind = NodeKind.NEST
            inner_q = plan.get("inner_question", "")
            outer_template = plan.get("outer_template", "{inner}")
            child = ReasoningNode(
                id=self._new_id(),
                question=inner_q,
                kind=NodeKind.LEAF,
                meta={"outer_template": outer_template},
            )
            node.add_child(child)
            self._expand_node(child)
            return

        # Fallback: treat as leaf
        node.kind = NodeKind.LEAF


class TreeExecutor:
    def __init__(self, retriever: Retriever, llm: LLM, top_k: int = 8):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    def answer_tree(self, root: ReasoningNode, sentences: List[Dict[str, Any]]) -> str:
        self._answer_node(root, sentences)
        return root.answer or ""

    def _answer_node(self, node: ReasoningNode, sentences: List[Dict[str, Any]]) -> None:
        if node.kind == NodeKind.LEAF:
            retrieved = self.retriever.rank_sentences(node.question, sentences, top_k=self.top_k)
            node.answer = self.llm.answer_subquestions(node.question, retrieved)["answer"]
            return

        for child in node.children:
            self._answer_node(child, sentences)

        if node.kind == NodeKind.BRANCH:
            node.answer = self._branch_aggregate(node)
        elif node.kind == NodeKind.NEST:
            node.answer = self._nest_aggregate(node, sentences)
        elif node.kind == NodeKind.ROOT:
            if len(node.children) == 1:
                node.answer = node.children[0].answer
            else:
                node.answer = self._branch_aggregate(node)

    def _branch_aggregate(self, node: ReasoningNode) -> str:
        child_info = "\n".join(f"Sub-question: {c.question}\nAnswer: {c.answer}" for c in node.children)
        prompt = (
            "Aggregate sub-answers to answer the final question.\n"
            "Use only the provided sub-answers; be concise.\n\n"
            f"{child_info}\n\nFinal question:\n{node.question}\n\nAnswer:"
        )
        return self.llm.chat(prompt).strip()

    def _nest_aggregate(self, node: ReasoningNode, sentences: List[Dict[str, Any]]) -> str:
        if not node.children:
            return ""
        inner = node.children[0]
        inner_answer = inner.answer or ""
        outer_template = inner.meta.get("outer_template", "{inner}")
        outer_q = outer_template.format(inner=inner_answer)
        retrieved = self.retriever.rank_sentences(outer_q, sentences, top_k=self.top_k)
        return self.llm.answer_subquestions(outer_q, retrieved)["answer"]
