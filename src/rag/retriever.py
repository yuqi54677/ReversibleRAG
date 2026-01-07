from typing import List, Dict

from sentence_transformers import SentenceTransformer, util


class Retriever:
    def __init__(self, model: SentenceTransformer, dataset_name: str):
        self.model = model
        self.dataset_name = dataset_name

    def flatten_context_sentences(self, example):
        if self.dataset_name == "hotpotqa":
            return self.flatten_context_sentences_hotpotqa(example)
        if self.dataset_name == "musique":
            return self.flatten_context_sentences_musique(example)
        if self.dataset_name == "2wiki":
            return self.flatten_context_sentences_2wiki(example)
        raise ValueError(f"Unsupported dataset {self.dataset_name}")

    def flatten_context_sentences_hotpotqa(self, example):
        flat = []
        titles = example["context"]["title"]
        sentences_per_title = example["context"]["sentences"]
        for title, sent_list in zip(titles, sentences_per_title):
            for i, sent in enumerate(sent_list):
                flat.append({"title": title, "sent_id": i, "text": sent})
        return flat

    def flatten_context_sentences_2wiki(self, example):
        flat = []
        context = example["context"]
        for title, sent_list in context:
            for sent_idx, sent in enumerate(sent_list):
                if not sent or not sent.strip():
                    continue
                flat.append({"title": title, "sent_id": sent_idx, "text": sent})
        return flat

    def flatten_context_sentences_musique(self, example):
        flat = []
        for paragraph in example["paragraphs"]:
            title = paragraph["title"]
            sentences = paragraph["paragraph_text"].split(". ")
            for i, sent in enumerate(sentences):
                sent = sent.strip()
                if not sent:
                    continue
                if i < len(sentences) - 1 and not sent.endswith("."):
                    sent = sent + "."
                flat.append({"title": title, "sent_id": i, "text": sent})
        return flat

    def rank_sentences(self, question: str, sentences: List[Dict], top_k: int = 8):
        q_emb = self.model.encode([question], convert_to_tensor=True)
        s_texts = [s["text"] for s in sentences]
        s_embs = self.model.encode(s_texts, convert_to_tensor=True)
        scores = util.cos_sim(q_emb, s_embs)[0]
        top_k = min(top_k, len(sentences))
        top_scores, top_indices = scores.topk(top_k)

        ranked = []
        for score, idx in zip(top_scores, top_indices):
            s = sentences[idx.item()]
            ranked.append(
                {
                    "title": s["title"],
                    "sent_id": s["sent_id"],
                    "text": s["text"],
                    "score": float(score.item()),
                }
            )
        return ranked
