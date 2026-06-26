from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from backend.models.schemas import ClaimElement, ExtractedDocument
from backend.services import rag_retriever


class _FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def compute_score(self, pairs, normalize=True):
        return self.scores[:len(pairs)]


class _FakeQdrantClient:
    def __init__(self, path):
        self.path = path
        self.collections = {}
        self.closed = False

    def collection_exists(self, collection_name):
        return collection_name in self.collections

    def get_collection(self, collection_name):
        if collection_name not in self.collections:
            raise RuntimeError("missing")
        return self.collections[collection_name]

    def count(self, collection_name, exact=True):
        class _Count:
            def __init__(self, count):
                self.count = count

        return _Count(len(self.collections.get(collection_name, [])))

    def create_collection(self, collection_name, vectors_config):
        self.collections[collection_name] = []

    def delete_collection(self, collection_name):
        self.collections.pop(collection_name, None)

    def upsert(self, collection_name, points):
        self.collections[collection_name] = list(points)

    def close(self):
        self.closed = True


class _FakePointStruct:
    def __init__(self, id, vector, payload):
        self.id = id
        self.vector = vector
        self.payload = payload


class _FakeVectorParams:
    def __init__(self, size, distance):
        self.size = size
        self.distance = distance


class _FakeDistance:
    COSINE = "cosine"


class RagRerankerTests(unittest.TestCase):
    def setUp(self):
        rag_retriever._runtime_status.update({
            "dense": "not_attempted",
            "qdrant": "not_attempted",
            "bm25": "not_attempted",
            "reranker": "not_attempted",
            "fallback_reason": "",
        })
        rag_retriever._reranker_inference_failed = False
        rag_retriever._reranker_load_failed = False
        rag_retriever._reranker_model = None
        rag_retriever._qdrant_handle_cache.clear()
        rag_retriever._query_embedding_cache.clear()

    def test_reranker_uses_best_element_score_per_candidate(self):
        chunks = [
            rag_retriever._SearchChunk("a", "alpha", "alpha"),
            rag_retriever._SearchChunk("b", "beta", "beta"),
        ]
        elements = [
            ClaimElement(label="A", text="first", importance="3"),
            ClaimElement(label="B", text="second", importance="3"),
        ]
        model = _FakeReranker([0.1, 0.2, 0.8, 0.4])

        with patch.object(rag_retriever, "_get_reranker_model", return_value=model):
            ranked = rag_retriever._rerank_candidates(elements, chunks, [0, 1])

        self.assertEqual([idx for _score, idx in ranked], [1, 0])
        self.assertEqual(rag_retriever._runtime_status["reranker"], "active")

    def test_reranker_failure_returns_none_for_rrf_fallback(self):
        class BrokenReranker:
            def compute_score(self, pairs, normalize=True):
                raise RuntimeError("boom")

        chunks = [rag_retriever._SearchChunk("a", "alpha", "alpha")]
        elements = [ClaimElement(label="A", text="first", importance="3")]

        with patch.object(rag_retriever, "_get_reranker_model", return_value=BrokenReranker()):
            ranked = rag_retriever._rerank_candidates(elements, chunks, [0])

        self.assertIsNone(ranked)
        self.assertEqual(rag_retriever._runtime_status["reranker"], "failed")
        self.assertIn("boom", rag_retriever._runtime_status["fallback_reason"])

    def test_query_embedding_cache_reuses_same_query(self):
        calls = []

        def fake_build(texts, model):
            calls.append(tuple(texts))
            return rag_retriever.np.ones((len(texts), 3), dtype=rag_retriever.np.float32)

        with patch.object(rag_retriever, "_build_dense_embeddings", side_effect=fake_build):
            first = rag_retriever._cached_query_embedding("same query", object())
            second = rag_retriever._cached_query_embedding("same query", object())

        self.assertEqual(len(calls), 1)
        self.assertTrue((first == second).all())

    def test_build_qdrant_index_reuses_cached_handle(self):
        doc = ExtractedDocument(
            filename="doc.pdf",
            doc_id="DOC1",
            doc_index=0,
            pdf_path=str(Path("uploads") / "job123" / "pdfs" / "doc.pdf"),
            raw_text="sample text",
        )
        chunks = [rag_retriever._SearchChunk("c1", "alpha", "alpha", doc_id="DOC1")]
        fake_model = object()
        fake_client = _FakeQdrantClient("unused")

        with (
            patch.object(rag_retriever, "_get_bge_model", return_value=fake_model),
            patch.object(rag_retriever, "get_or_build_dense_index", return_value=rag_retriever.np.ones((1, 3), dtype=rag_retriever.np.float32)),
            patch("qdrant_client.QdrantClient", return_value=fake_client),
            patch("qdrant_client.models.PointStruct", _FakePointStruct),
            patch("qdrant_client.models.VectorParams", _FakeVectorParams),
            patch("qdrant_client.models.Distance", _FakeDistance),
        ):
            first = rag_retriever.build_qdrant_index(doc, chunks, Path("uploads"))
            second = rag_retriever.build_qdrant_index(doc, chunks, Path("uploads"))

        self.assertIs(first, second)
        self.assertEqual(fake_client.count(first[1], exact=True).count, 1)


if __name__ == "__main__":
    unittest.main()
