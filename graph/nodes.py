# graph/nodes.py
"""
★ 核心修复: 模块注册表 + 依赖注入

问题: 原来 _get_module() 每次 new 新实例, 与 Pipeline 中的实例完全隔离
修复: Pipeline 在初始化时将自己的实例注册到全局注册表, 节点函数从注册表获取
"""
import time
import hashlib
from typing import Any, Dict, List, Optional

from graph.state import RAGGraphState, StepTiming
from core.logger import logger


# ═══════════════════════════════════════════════════════
# ★ 全局模块注册表 (依赖注入容器)
# ═══════════════════════════════════════════════════════
class ModuleRegistry:
    """
    模块注册表: Pipeline 注册实例 → 节点函数取用

    解决的问题:
      Pipeline 和 Graph 节点使用同一套模块实例,
      确保 vector_store._collection 是已初始化的状态
    """
    _instance = None
    _modules: Dict[str, Any] = {}

    @classmethod
    def get_instance(cls) -> "ModuleRegistry":
        if cls._instance is None:
            cls._instance = ModuleRegistry()
        return cls._instance

    def register(self, name: str, module: Any):
        """Pipeline 调用: 注册一个模块实例"""
        self._modules[name] = module
        logger.debug(f"模块注册: {name} → {type(module).__name__}")

    def get(self, name: str) -> Any:
        """节点函数调用: 获取已注册的模块实例"""
        if name not in self._modules:
            raise RuntimeError(
                f"模块 '{name}' 未注册! "
                f"已注册: {list(self._modules.keys())}. "
                f"请确保 Pipeline 已初始化."
            )
        return self._modules[name]

    def is_registered(self, name: str) -> bool:
        return name in self._modules

    def clear(self):
        self._modules.clear()

    @property
    def registered_modules(self) -> List[str]:
        return list(self._modules.keys())


# 全局注册表单例
registry = ModuleRegistry.get_instance()


def _get(name: str) -> Any:
    """节点函数获取模块的快捷方式"""
    return registry.get(name)


def _make_timing(name: str, start: float, status: str = "success", **meta) -> StepTiming:
    return StepTiming(
        name=name,
        duration_ms=(time.perf_counter() - start) * 1000,
        status=status,
        metadata=meta,
    )


def _elapsed(state: RAGGraphState) -> float:
    return sum(t["duration_ms"] for t in state.get("step_timings", []))


def _remaining_budget(state: RAGGraphState) -> float:
    return state["budget_ms"] - _elapsed(state)


# ═══════════════════════════════════════════════════════
# Node: 缓存检查
# ═══════════════════════════════════════════════════════
def cache_check_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    cache = _get("cache")

    # 精确缓存
    cached = cache.query_cache.get(state["cache_key"])
    if cached is not None:
        logger.info(f"[{state['trace_id']}] 🎯 精确缓存命中")
        return {
            "cache_hit": True,
            "answer": cached["answer"],
            "sources": cached.get("sources", []),
            "step_timings": [_make_timing("cache_check", start, cache_type="exact")],
        }

    # 语义缓存
    try:
        vs = _get("vector_store")
        q_emb = vs.embed_service.embed_query(state["query"])
        sem_cached = cache.semantic_cache.get(q_emb)
        if sem_cached is not None:
            logger.info(f"[{state['trace_id']}] 🎯 语义缓存命中")
            return {
                "cache_hit": True,
                "answer": sem_cached["answer"],
                "sources": sem_cached.get("sources", []),
                "step_timings": [_make_timing("cache_check", start, cache_type="semantic")],
            }
    except Exception:
        pass

    return {
        "cache_hit": False,
        "step_timings": [_make_timing("cache_check", start, cache_type="miss")],
    }


# ═══════════════════════════════════════════════════════
# Node: 路由决策
# ═══════════════════════════════════════════════════════
def route_query_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    router = _get("router")

    remaining = _remaining_budget(state)
    route = router.decide_mode(
        state["query"],
        forced_mode=state.get("forced_mode"),
        remaining_budget_ms=remaining,
    )

    return {
        "pipeline_mode": route["mode"].value,
        "query_complexity": router.classify_query(state["query"]),
        "steps_config": route["steps"],
        "step_timings": [_make_timing("route_query", start, mode=route["mode"].value)],
    }


# ═══════════════════════════════════════════════════════
# Node: 查询改写 (NORMAL 模式)
# ═══════════════════════════════════════════════════════
def rewrite_query_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    transformer = _get("transformer")

    rewritten = transformer.rewrite_only(state["query"])
    new_queries = [rewritten] if rewritten != state["query"] else []

    return {
        "rewritten_query": rewritten,
        "all_queries": new_queries,
        "step_timings": [_make_timing("rewrite_query", start, rewritten=rewritten[:50])],
    }


# ═══════════════════════════════════════════════════════
# Node: 全量查询变换 (FULL 模式, 并发)
# ═══════════════════════════════════════════════════════
def transform_all_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    transformer = _get("transformer")

    timeout_ms = min(_remaining_budget(state) * 0.4, 800)
    result = transformer.transform(state["query"], timeout_ms=timeout_ms)

    new_queries = [q for q in result["queries"] if q != state["query"]]

    return {
        "rewritten_query": result.get("rewritten", ""),
        "hyde_document": result.get("hyde_document", ""),
        "step_back_query": result.get("step_back", ""),
        "sub_queries": result.get("sub_queries", []),
        "all_queries": new_queries,
        "step_timings": [_make_timing(
            "transform_all", start,
            query_count=len(result["queries"]),
        )],
    }


# ═══════════════════════════════════════════════════════
# Node: 混合检索
# ═══════════════════════════════════════════════════════
def hybrid_search_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    retriever = _get("retriever")  # ★ 从注册表获取, 和 Pipeline 是同一个实例

    search_query = state.get("rewritten_query") or state["query"]
    from config.settings import rag_config

    try:
        results = retriever.hybrid_search(
            search_query,
            top_k=rag_config.retriever.top_k_final * 2,
        )
    except Exception as e:
        logger.error(f"混合检索失败: {e}")
        results = []

    return {
        "retrieved_documents": results,
        "retrieval_method": "hybrid_single",
        "step_timings": [_make_timing("hybrid_search", start, count=len(results))],
    }


# ═══════════════════════════════════════════════════════
# Node: 多查询批量检索
# ═══════════════════════════════════════════════════════
def multi_query_search_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    retriever = _get("retriever")

    queries = state.get("all_queries", [state["query"]])
    if not queries:
        queries = [state["query"]]

    from config.settings import rag_config

    try:
        results = retriever.multi_query_search(
            queries,
            top_k=rag_config.retriever.top_k_final * 2,
        )
    except Exception as e:
        logger.error(f"多查询检索失败: {e}")
        results = []

    return {
        "retrieved_documents": results,
        "retrieval_method": f"multi_query({len(queries)})",
        "step_timings": [_make_timing(
            "multi_query_search", start,
            query_count=len(queries), result_count=len(results),
        )],
    }


# ═══════════════════════════════════════════════════════
# Node: 重排序
# ═══════════════════════════════════════════════════════
def rerank_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    reranker = _get("reranker")
    from config.settings import rag_config

    docs = state.get("retrieved_documents", [])
    if not docs:
        return {
            "reranked_documents": [],
            "step_timings": [_make_timing("rerank", start, status="skipped", reason="no_docs")],
        }

    remaining = _remaining_budget(state)
    if remaining < 300:
        top_k = rag_config.retriever.reranker_top_k
        return {
            "reranked_documents": docs[:top_k],
            "step_timings": [_make_timing("rerank", start, method="budget_truncate")],
        }

    reranked = reranker.rerank(
        query=state["query"], docs=docs,
        method="auto", top_k=rag_config.retriever.reranker_top_k,
    )

    return {
        "reranked_documents": reranked,
        "step_timings": [_make_timing("rerank", start, before=len(docs), after=len(reranked))],
    }


def skip_rerank_node(state: RAGGraphState) -> dict:
    from config.settings import rag_config
    docs = state.get("retrieved_documents", [])
    return {
        "reranked_documents": docs[:rag_config.retriever.reranker_top_k],
        "step_timings": [StepTiming(
            name="rerank", duration_ms=0, status="skipped",
            metadata={"reason": "fast_mode"},
        )],
    }


# ═══════════════════════════════════════════════════════
# Node: 答案生成
# ═══════════════════════════════════════════════════════
def generate_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    synthesizer = _get("synthesizer")

    docs = state.get("reranked_documents") or state.get("retrieved_documents", [])
    if not docs:
        return {
            "answer": "抱歉，未在知识库中找到相关信息。",
            "context": "", "sources": [], "quality_score": 0.0,
            "step_timings": [_make_timing("generate", start, status="no_context")],
        }

    result = synthesizer.generate(state["query"], docs)

    return {
        "answer": result["answer"],
        "context": result["context"],
        "sources": result["sources"],
        "step_timings": [_make_timing(
            "generate", start,
            answer_len=len(result["answer"]),
            source_count=len(result["sources"]),
        )],
    }


# ═══════════════════════════════════════════════════════
# Node: 质量评估
# ═══════════════════════════════════════════════════════
def quality_gate_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    query = state["query"]

    score = 1.0
    if len(answer) < 30:
        score -= 0.3
    if "[来源" not in answer and len(sources) > 0:
        score -= 0.1
    refuse_keywords = ["无法回答", "没有找到", "无法确定", "资料不足"]
    if any(kw in answer for kw in refuse_keywords):
        score -= 0.2

    import jieba
    query_words = set(jieba.cut(query))
    answer_words = set(jieba.cut(answer))
    coverage = len(query_words & answer_words) / max(len(query_words), 1)
    if coverage < 0.3:
        score -= 0.2

    score = max(0.0, min(1.0, score))

    return {
        "quality_score": score,
        "step_timings": [_make_timing("quality_gate", start, score=f"{score:.2f}")],
    }


# ═══════════════════════════════════════════════════════
# Node: 幻觉检测
# ═══════════════════════════════════════════════════════
def hallucination_check_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    remaining = _remaining_budget(state)

    if remaining < 800:
        return {
            "hallucination_check": {"has_hallucination": False, "skipped": True},
            "step_timings": [_make_timing(
                "hallucination_check", start,
                status="skipped", reason=f"budget({remaining:.0f}ms)",
            )],
        }

    synthesizer = _get("synthesizer")
    result = synthesizer.check_hallucination(
        state.get("answer", ""),
        state.get("context", ""),
    )

    return {
        "hallucination_check": result,
        "step_timings": [_make_timing(
            "hallucination_check", start,
            has_hallucination=result.get("has_hallucination", False),
        )],
    }


# ═══════════════════════════════════════════════════════
# Node: 答案修正
# ═══════════════════════════════════════════════════════
def refine_answer_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    synthesizer = _get("synthesizer")

    refined = synthesizer.refine(state["query"], state.get("context", ""))
    new_retry = state.get("retry_count", 0) + 1

    return {
        "answer": refined,
        "retry_count": new_retry,
        "step_timings": [_make_timing("refine_answer", start, retry=new_retry)],
    }


# ═══════════════════════════════════════════════════════
# Node: 写缓存
# ═══════════════════════════════════════════════════════
def write_cache_node(state: RAGGraphState) -> dict:
    start = time.perf_counter()
    cache = _get("cache")

    cache_data = {
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
    }
    cache.query_cache.set(state["cache_key"], cache_data)

    try:
        vs = _get("vector_store")
        q_emb = vs.embed_service.embed_query(state["query"])
        cache.semantic_cache.set(q_emb, state["cache_key"], cache_data)
    except Exception:
        pass

    return {
        "step_timings": [_make_timing("write_cache", start)],
    }


# ═══════════════════════════════════════════════════════
# Node: 汇总
# ═══════════════════════════════════════════════════════
def finalize_node(state: RAGGraphState) -> dict:
    total = sum(t["duration_ms"] for t in state.get("step_timings", []))
    return {"total_duration_ms": total}