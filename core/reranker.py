# core/reranker.py
"""
★ 轻量重排序 — 优先用小模型, 降级策略
"""
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate

from config.settings import rag_config
from core.async_llm_pool import llm_pool
from core.logger import logger, timed


RERANK_BATCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """评估以下文档与查询的相关性。对每个文档给出1-10的评分。
输出格式: 每行一个数字, 依次对应每个文档。只输出数字, 不解释。"""),
    ("human", """查询: {query}

{documents}

评分(每行一个数字):"""),
])


class Reranker:
    def __init__(self):
        self.config = rag_config.retriever
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            try:
                from FlagEmbedding import FlagReranker
                self._cross_encoder = FlagReranker(self.config.reranker_model, use_fp16=True)
            except Exception as e:
                logger.warning(f"Cross-Encoder 不可用: {e}")
        return self._cross_encoder

    @timed("Cross-Encoder重排序")
    def cross_encoder_rerank(self, query: str, docs: List[Dict], top_k: int = None) -> List[Dict]:
        top_k = top_k or self.config.reranker_top_k
        encoder = self._get_cross_encoder()
        if encoder is None:
            return self.llm_batch_rerank(query, docs, top_k)

        pairs = [[query, d["content"][:512]] for d in docs]  # ★ 截断到 512 加速
        scores = encoder.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        return sorted(docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

    @timed("LLM批量重排序")
    def llm_batch_rerank(self, query: str, docs: List[Dict], top_k: int = None) -> List[Dict]:
        """
        ★ 优化: 将所有文档放在一个 Prompt 里, 一次 LLM 调用完成全部评分
        原始: N 次 LLM 调用
        优化: 1 次 LLM 调用
        """
        top_k = top_k or self.config.reranker_top_k

        docs_text = "\n\n".join(
            f"[文档{i+1}]\n{d['content'][:400]}"
            for i, d in enumerate(docs)
        )

        result = llm_pool.call(
            RERANK_BATCH_PROMPT,
            {"query": query, "documents": docs_text},
            use_fast=True,  # ★ 用快速模型
        )

        if result:
            try:
                lines = [l.strip() for l in result.strip().split("\n") if l.strip()]
                scores = []
                for line in lines:
                    # 提取数字
                    nums = ''.join(c for c in line if c.isdigit() or c == '.')
                    if nums:
                        scores.append(float(nums))
                for d, s in zip(docs, scores):
                    d["rerank_score"] = min(max(s, 0), 10) / 10.0
            except Exception as e:
                logger.warning(f"重排序解析失败: {e}")
                for i, d in enumerate(docs):
                    d["rerank_score"] = 1.0 / (i + 1)
        else:
            for i, d in enumerate(docs):
                d["rerank_score"] = 1.0 / (i + 1)

        ranked = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
        # 相关性过滤
        filtered = [d for d in ranked if d.get("rerank_score", 0) >= 0.3]
        return (filtered or ranked[:top_k])[:top_k]

    def rerank(self, query: str, docs: List[Dict], method: str = "auto", top_k: int = None) -> List[Dict]:
        if not docs:
            return []
        if method == "auto":
            # ★ 自动选择: 如果有 GPU 就用 Cross-Encoder, 否则用 LLM 批量
            if self._get_cross_encoder():
                return self.cross_encoder_rerank(query, docs, top_k)
            return self.llm_batch_rerank(query, docs, top_k)
        elif method == "cross_encoder":
            return self.cross_encoder_rerank(query, docs, top_k)
        else:
            return self.llm_batch_rerank(query, docs, top_k)