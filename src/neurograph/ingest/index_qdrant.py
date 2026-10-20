import re
from pathlib import Path

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridIndex:
    # dense (BGE in embedded Qdrant) + keyword (BM25), fused with reciprocal rank fusion
    def __init__(self, chunks: list[dict], path: Path, collection: str, model_name: str, query_prefix: str):
        self.chunks, self.collection, self.query_prefix = chunks, collection, query_prefix
        self.model = SentenceTransformer(model_name)
        self.client = QdrantClient(path=str(path))
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in chunks])

    def ready(self) -> bool:
        # an index built earlier in this run (same chunks) can be reused
        return (self.client.collection_exists(self.collection)
                and self.client.count(self.collection).count == len(self.chunks))

    def build(self, batch_size: int) -> int:
        texts = [f"{c['title']}\n{c['text']}" for c in self.chunks]  # title helps short chunks
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(self.collection, vectors_config=models.VectorParams(
            size=vecs.shape[1], distance=models.Distance.COSINE))
        self.client.upsert(self.collection, points=[
            models.PointStruct(id=i, vector=v.tolist(), payload={k: c[k] for k in ("id", "url", "title", "group", "text")})
            for i, (c, v) in enumerate(zip(self.chunks, vecs))])
        return vecs.shape[1]

    def dense(self, query: str, k: int) -> list[int]:
        q = self.model.encode(self.query_prefix + query, normalize_embeddings=True)
        return [p.id for p in self.client.query_points(self.collection, query=q.tolist(), limit=k).points]

    def keyword(self, query: str, k: int) -> list[int]:
        scores = self.bm25.get_scores(tokenize(query))
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k]

    def hybrid(self, query: str, k: int, rrf_k: int, depth: int = 20) -> list[int]:
        fused: dict[int, float] = {}
        for ranking in (self.dense(query, depth), self.keyword(query, depth)):
            for rank, i in enumerate(ranking):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused, key=lambda i: -fused[i])[:k]
