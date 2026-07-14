"""Unit tests for BM25 tokenization and RRF fusion."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bm25_index import extract_document_codes, tokenize_for_bm25
from hybrid_retrieval import fuse_dense_bm25, rrf_score
from retrieval_query_analysis import analyze_query


def test_doc_code_tokenization():
    toks = tokenize_for_bm25("Find DNV-CG-0264 and MASS Code guidance")
    assert "dnv-cg-0264" in toks or "0264" in toks
    assert "mass" in toks
    codes = extract_document_codes("See LR Notice No.1 Section 15")
    assert any("notice" in c.lower() for c in codes)


def test_rrf_fusion_prefers_both_lists():
    dense_ids = ["a", "b", "c"]
    dense_dist = {"a": 0.1, "b": 0.2, "c": 0.3}
    dense_meta = {
        "a": {"doc_id": "d1", "source": "DNV", "file_name": "DNV-CG-0264.pdf"},
        "b": {"doc_id": "d2", "source": "ABS", "file_name": "abs.pdf"},
        "c": {"doc_id": "d3", "source": "DNV", "file_name": "other.pdf"},
    }
    dense_doc = {"a": "autonomous", "b": "other", "c": "vessel"}

    class Hit:
        chunk_id = "a"
        rank = 1
        score = 5.0
        meta = dense_meta["a"]
        document = dense_doc["a"]

    signals = analyze_query("DNV autonomous Smart Vessel")
    fused = fuse_dense_bm25(
        dense_ids,
        dense_dist,
        dense_meta,
        dense_doc,
        [Hit()],
        query="DNV autonomous Smart Vessel",
        signals=signals,
        priority_doc_ids=set(),
        society="DNV",
    )
    assert fused[0].chunk_id == "a"
    assert rrf_score(1) > rrf_score(2)


if __name__ == "__main__":
    test_doc_code_tokenization()
    test_rrf_fusion_prefers_both_lists()
    print("ok")
