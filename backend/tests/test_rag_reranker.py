from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.models.schemas import ClaimElement
from backend.services import rag_retriever


class _FakeReranker:
    def __init__(self, scores):
        self.scores = scores

    def compute_score(self, pairs, normalize=True):
        return self.scores[:len(pairs)]


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


if __name__ == "__main__":
    unittest.main()
