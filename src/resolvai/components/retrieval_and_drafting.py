import numpy as np
import pandas as pd
import yaml

from resolvai import logger
from resolvai.entity.config_entity import RetrievalConfig
from resolvai.utils.common import save_bin, load_bin, append_jsonl
from resolvai.utils.llm_client import LLMClient


class RetrievalDrafting:
    def __init__(self, config: RetrievalConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        self.embed_model = SentenceTransformer(config.embedding_model)
        self.llm_call = LLMClient.get(config.llm_provider, config.llm_model)
        self.pairs_df = None
        self.index_matrix = None

    def build_index(self) -> None:
        pairs_df = pd.read_csv(self.config.cleaned_pairs_path)
        embeddings = self.embed_model.encode(pairs_df["customer_text_clean"].tolist(), show_progress_bar=False)
        save_bin({"pairs_df": pairs_df, "index_matrix": embeddings}, self.config.index_cache_path)
        self.pairs_df, self.index_matrix = pairs_df, embeddings
        logger.info(f"retrieval index built: {embeddings.shape} -> {self.config.index_cache_path}")

    def load_index(self) -> None:
        cached = load_bin(self.config.index_cache_path)
        self.pairs_df, self.index_matrix = cached["pairs_df"], cached["index_matrix"]

    def _ensure_loaded(self) -> None:
        if self.pairs_df is None:
            self.load_index()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a / (np.linalg.norm(a) + 1e-9)
        b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return b @ a

    def retrieve(self, customer_text: str, top_k: int = None) -> list:
        self._ensure_loaded()
        top_k = top_k or self.config.top_k_retrieval
        q_emb = self.embed_model.encode([customer_text])[0]
        sims = self._cosine_sim(q_emb, self.index_matrix)
        top_idx = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_idx:
            row = self.pairs_df.iloc[idx]
            results.append(
                {
                    "similarity": float(sims[idx]),
                    "customer_text": row["customer_text_clean"],
                    "brand_reply": row["brand_text_clean"],
                    "is_boilerplate": bool(row["is_boilerplate"]),
                    "pair_index": int(idx),
                }
            )
        return results

    def generate_reply(self, customer_text: str, retrieved: list) -> str:
        examples_block = "\n\n".join(
            f"Example {i+1} (similarity {r['similarity']:.2f}):\n"
            f"Customer: {r['customer_text']}\nBrand reply: {r['brand_reply']}"
            for i, r in enumerate(retrieved)
        )
        prompt = f"""You are a customer support agent. Draft a reply to the new customer message
below, grounded in how this brand has historically resolved similar issues (the examples). Match
the brand's tone. Do not invent policies, refunds, or tracking numbers not implied by the examples.

{examples_block}

New customer message: "{customer_text}"

Reply:"""
        return self.llm_call(prompt, max_tokens=200)

    def run_and_log(self, customer_text: str, intent: str, intent_confidence: float, decision: dict) -> dict:
        retrieved = self.retrieve(customer_text)
        reply = self.generate_reply(customer_text, retrieved)
        record = {
            "customer_text": customer_text,
            "intent": intent,
            "intent_confidence": intent_confidence,
            "retrieved": retrieved,
            "reply": reply,
            "decision": decision,
        }
        append_jsonl(self.config.generation_log_path, record)
        return record
