# core/retriever.py
"""
★ 并行多路检索 — Dense + Sparse 同时执行 + 批量 Milvus 查询
"""
import time
from typing import List, Dict, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
import jieba

from config.settings import rag_config
from core.vector_store import MilvusVectorStore
from core.logger import logger, timed


class HybridRetriever:
    def __init__(self, vector_store: MilvusVectorStore):
        self.vs = vector_store
        self.config = rag_config.retriever
        self._bm25 = None
        self._corpus: List[Dict] = []
        self._executor = ThreadPoolExecutor(max_workers=4)

    @timed("BM25索引构建")
    def build_bm25_index(self, documents: List[Document]):
        self._corpus = []
        tokenized = []
        for doc in documents:
            tokens = list(jieba.cut(doc.page_content))
            tokenized.append(tokens)
            self._corpus.append({"content": doc.page_content, "metadata": doc.metadata})
        self._bm25 = BM25Okapi(tokenized, k1=self.config.bm25_k1, b=self.config.bm25_b)
        logger.info(f"BM25 索引: {len(tokenized)} 文档")

    def _sparse_search(self, query: str, top_k: int) -> List[Dict]:
        if self._bm25 is None:
            return []
        tokens = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "content": self._corpus[i]["content"],
                "score": float(scores[i]),
                "source_file": self._corpus[i]["metadata"].get("source_file", ""),
                "keywords": ",".join(self._corpus[i]["metadata"].get("keywords", [])),
                "method": "sparse",
            }
            for i in top_idx if scores[i] > 0
        ]

    def _rrf(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        doc_map = {}
        for rlist in result_lists:
            for rank, doc in enumerate(rlist):
                key = doc["content"][:100]
                scores[key] += 1.0 / (k + rank + 1)
                if key not in doc_map:
                    doc_map[key] = doc
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [{**doc_map[k], "rrf_score": scores[k]} for k in sorted_keys]

    @timed("混合检索(并行)")
    def hybrid_search(self, query: str, top_k: int = None) -> List[Dict]:
        """★ Dense + Sparse 并行执行"""
        top_k = top_k or self.config.top_k_final

        # ★ 并行: Dense 和 Sparse 同时跑
        dense_future = self._executor.submit(
            self.vs.search, query, self.config.top_k_dense
        )
        sparse_future = self._executor.submit(
            self._sparse_search, query, self.config.top_k_sparse
        )

        dense_results = dense_future.result(timeout=5)
        sparse_results = sparse_future.result(timeout=5)

        logger.info(f"Dense: {len(dense_results)}, Sparse: {len(sparse_results)}")
        fused = self._rrf([dense_results, sparse_results])
        return fused[:top_k]

    @timed("多查询批量检索")
    def multi_query_search(self, queries: List[str], top_k: int = None) -> List[Dict]:
        """
        ★ 核心优化: 多个查询的向量检索合并成一次批量请求
        原始: N 次 Milvus 查询 → N * RTT
        优化: 1 次批量查询 + N 次 BM25(内存) → 1 * RTT
        """
        top_k = top_k or self.config.top_k_final

        # ★ 批量 Dense 检索 (1 次网络请求)
        start = time.perf_counter()
        all_dense = self.vs.search_batch(queries, top_k=self.config.top_k_dense)
        dense_ms = (time.perf_counter() - start) * 1000
        logger.info(f"批量 Dense 检索 {len(queries)} 个查询: {dense_ms:.0f}ms")

        # Sparse 检索 (纯内存计算, 很快)
        all_sparse = [
            self._sparse_search(q, self.config.top_k_sparse)
            for q in queries
        ]

        # 合并所有结果用 RRF
        all_lists = []
        for dense, sparse in zip(all_dense, all_sparse):
            for d in dense:
                d["method"] = "dense"
            all_lists.append(dense)
            all_lists.append(sparse)

        fused = self._rrf(all_lists)
        return fused[:top_k]