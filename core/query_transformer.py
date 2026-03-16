# core/query_transformer.py
"""
★ 并发查询变换 — 4 种变换并行执行, 延迟从 ~2400ms 降到 ~800ms
"""
from typing import List, Dict, Optional

from langchain_core.prompts import ChatPromptTemplate

from config.settings import rag_config
from core.async_llm_pool import llm_pool
from core.logger import logger, timed


# ─── 预定义 Prompt 模板 ─────────────────────────────
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "将用户查询重写为更适合知识库检索的形式。保持原意,补充上下文,消除歧义。只输出重写结果。"),
    ("human", "原始查询: {query}\n重写:"),
])

HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "根据问题撰写一段150-200字的专业文档片段,像从权威文档中摘录的。即使不确定答案也请合理生成。"),
    ("human", "问题: {query}\n文档片段:"),
])

STEP_BACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "给定具体问题,生成一个更通用/更高层次的'退一步'问题,用于检索背景知识。只输出问题。"),
    ("human", "原始问题: {query}\n退一步问题:"),
])

SUB_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "将复杂问题拆解为2-3个更简单的子问题。每行一个子问题,不编号。"),
    ("human", "问题: {query}\n子问题:"),
])


class QueryTransformer:
    """★ 并发查询变换器"""

    def __init__(self):
        self.config = rag_config.query_transform

    @timed("查询变换(并发)")
    def transform(self, query: str, timeout_ms: float = None) -> Dict:
        """
        ★ 核心优化: 所有启用的变换方法并行调用 LLM
        原始: 串行 4 次 → ~2400ms
        优化: 并行 4 次 → ~800ms (取决于最慢的一个)
        """
        timeout_s = (timeout_ms / 1000) if timeout_ms else rag_config.llm.request_timeout

        # 构造并发任务列表
        tasks = []
        task_names = []

        if self.config.enable_query_rewrite:
            tasks.append({"prompt": REWRITE_PROMPT, "variables": {"query": query}, "use_fast": True})
            task_names.append("rewrite")

        if self.config.enable_hyde:
            tasks.append({"prompt": HYDE_PROMPT, "variables": {"query": query}, "use_fast": True})
            task_names.append("hyde")

        if self.config.enable_step_back:
            tasks.append({"prompt": STEP_BACK_PROMPT, "variables": {"query": query}, "use_fast": True})
            task_names.append("step_back")

        if self.config.enable_sub_query:
            tasks.append({"prompt": SUB_QUERY_PROMPT, "variables": {"query": query}, "use_fast": True})
            task_names.append("sub_query")

        if not tasks:
            return {"original": query, "queries": [query]}

        # ★ 并发执行
        results = llm_pool.call_concurrent(tasks, timeout=timeout_s)

        # 解析结果
        output = {"original": query, "queries": [query]}

        for name, result in zip(task_names, results):
            if result is None:
                continue

            if name == "rewrite":
                output["rewritten"] = result
                output["queries"].append(result)
            elif name == "hyde":
                output["hyde_document"] = result
                output["queries"].append(result)
            elif name == "step_back":
                output["step_back"] = result
                output["queries"].append(result)
            elif name == "sub_query":
                subs = [q.strip() for q in result.split("\n") if q.strip() and len(q.strip()) > 5]
                output["sub_queries"] = subs[:self.config.max_sub_queries]
                output["queries"].extend(subs[:self.config.max_sub_queries])

        # 去重
        seen = set()
        unique = []
        for q in output["queries"]:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        output["queries"] = unique

        logger.info(f"查询变换: 1 → {len(unique)} 个查询 (并发完成)")
        return output

    def rewrite_only(self, query: str) -> str:
        """★ 极速模式: 只做查询改写"""
        result = llm_pool.call(REWRITE_PROMPT, {"query": query}, use_fast=True)
        return result or query