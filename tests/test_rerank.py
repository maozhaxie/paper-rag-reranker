"""Reranking contracts and retrieval integration, with no paid API calls."""
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from configuration import Settings, signature, validate
import reranker
import vector_retriever as retriever
from server import create_app


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.cfg = replace(Settings(), key="TEST-SECRET")
        self.rows = [{"id": n, "text": text, "source": "paper.pdf", "page": n,
                      "chunk_id": n * 10, "l2_distance": 1 + n / 10}
                     for n, text in enumerate(["参考文献", "变量定义", "研究样本为2008—2022年"], 1)]

    def test_reordered_evidence_keeps_metadata_and_renumbers_citations(self):
        result = reranker.select_ranked_sources(self.rows, [
            {"index": 0, "relevance_score": .1}, {"index": 2, "relevance_score": .9}], 2)
        self.assertEqual([s["id"] for s in result], [1, 2])
        self.assertEqual((result[0]["page"], result[0]["chunk_id"], result[0]["vector_rank"]), (3, 30, 3))
        self.assertEqual(result[0]["text"], self.rows[2]["text"])
        self.assertEqual(result[0]["l2_distance"], self.rows[2]["l2_distance"])
        self.assertNotIn("rerank_score", self.rows[2])

    def test_rejects_incomplete_duplicate_or_invalid_provider_results(self):
        valid = {"index": 0, "relevance_score": .5}
        for results in [None, [], [valid, valid], [valid, {"index": 3, "relevance_score": .2}],
                        [valid, {"index": True, "relevance_score": .2}],
                        [valid, {"index": 1, "relevance_score": float("nan")}],
                        [valid, {"index": 1, "relevance_score": True}],
                        [valid, {"index": 1, "relevance_score": 1.1}], [valid, None]]:
            with self.subTest(results=results), self.assertRaises(reranker.RerankError):
                reranker.select_ranked_sources(self.rows, results, 2)

    def test_http_contract_uses_text_only_and_maps_provider_indices(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps({
            "results": [{"index": 2, "relevance_score": .9}]}).encode()
        with patch("urllib.request.build_opener", return_value=opener):
            result = reranker.rerank_sources("样本年份？", self.rows, 1, self.cfg)
        req = opener.open.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(body["documents"], [s["text"] for s in self.rows])
        self.assertEqual(body["top_n"], 1)
        self.assertEqual(body["model"], "qwen3-rerank")
        self.assertEqual(req.full_url, self.cfg.rerank_url)
        self.assertEqual(result[0]["chunk_id"], 30)

    def test_empty_candidates_never_call_provider(self):
        with patch("urllib.request.build_opener") as request:
            self.assertEqual(reranker.rerank_sources("q", [], 5, self.cfg), [])
            request.assert_not_called()

    def test_http_failure_is_actionable_and_never_echoes_body(self):
        opener = MagicMock()
        opener.open.side_effect = HTTPError(self.cfg.rerank_url, 403, "TEST-SECRET", {}, BytesIO(b"TEST-SECRET"))
        with patch("urllib.request.build_opener", return_value=opener), self.assertRaises(reranker.RerankError) as ctx:
            reranker.rerank_sources("q", self.rows, 1, self.cfg)
        self.assertIn("403", str(ctx.exception))
        self.assertNotIn("TEST-SECRET", str(ctx.exception))

    def test_rejects_insecure_url(self):
        with self.assertRaises(reranker.RerankError):
            reranker.rerank_sources("q", self.rows, 1, replace(self.cfg, rerank_url="http://example.com"))

    def test_candidate_recall_and_vector_only_switch(self):
        hits = [{"distance": row["l2_distance"], "entity": row} for row in self.rows]
        db, client = MagicMock(), MagicMock()
        db.search.return_value = [hits]
        with patch.object(retriever, "check_receipt"), patch.object(retriever, "ensure_collection", return_value="paper"), \
             patch.object(retriever, "model_client", return_value=client), \
             patch.object(retriever, "db_client", return_value=db), \
             patch.object(retriever, "embed", return_value=[[1., 0.]]), \
             patch.object(retriever, "rerank_sources", return_value=[self.rows[2]]) as rank:
            result = retriever.search_similar_content("q", 1, self.cfg)
            self.assertEqual(db.search.call_args.kwargs["limit"], 20)
            self.assertEqual(len(rank.call_args.args[1]), 3)
            self.assertEqual(result[0]["chunk_id"], 30)
            db.close.assert_called_once()
            client.close.assert_called_once()
            rank.reset_mock()
            result = retriever.search_similar_content("q", 1, self.cfg, rerank=False)
            self.assertEqual(db.search.call_args.kwargs["limit"], 1)
            self.assertEqual(len(result), 1)
            rank.assert_not_called()

    def test_reranking_does_not_change_database_signature(self):
        with TemporaryDirectory() as directory:
            paper = Path(directory) / "signature-fixture.pdf"
            paper.write_bytes(b"Sample input for the file signature")
            config = replace(self.cfg, paper=paper)
            self.assertEqual(signature(config), signature(replace(config, rerank_enabled=False, candidate_k=40)))
        with self.assertRaises(ValueError):
            validate(replace(self.cfg, candidate_k=100))

    def test_http_switch_validation_and_friendly_error(self):
        client = create_app().test_client()
        for endpoint in ["search", "query"]:
            response = client.post("/api/vector/" + endpoint, json={"question": "q", "rerank": "false"})
            self.assertEqual(response.status_code, 400)
        with patch.object(retriever, "search_similar_content", return_value=[]) as search:
            response = client.post("/api/vector/search", json={"question": "q", "rerank": False})
            self.assertEqual(response.status_code, 200)
            search.assert_called_once_with("q", 5, rerank=False)
        with patch.object(retriever, "search_similar_content", side_effect=reranker.RerankError("请核对重排权限")):
            response = client.post("/api/vector/search", json={"question": "q"})
            self.assertEqual(response.status_code, 502)
            self.assertIn("重排权限", response.json["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
