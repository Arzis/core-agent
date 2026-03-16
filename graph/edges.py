# graph/edges.py
"""
★ LangGraph 条件边: 控制图的分支走向

每个函数接收 state, 返回下一个节点的名称
"""
from graph.state import RAGGraphState
from core.logger import logger


def should_use_cache(state: RAGGraphState) -> str:
    """缓存命中 → END, 否则继续"""
    if state.get("cache_hit"):
        logger.info(f"[{state['trace_id']}] 🎯 缓存命中, 跳过后续步骤")
        return "finalize"
    return "route_query"


def route_by_mode(state: RAGGraphState) -> str:
    """
    根据管线模式选择查询变换策略:
      FAST   → 直接检索
      NORMAL → 仅改写
      FULL   → 全量变换
    """
    mode = state.get("pipeline_mode", "normal")
    logger.info(f"[{state['trace_id']}] 路由: mode={mode}")

    if mode == "fast":
        return "hybrid_search"           # 跳过查询变换
    elif mode == "normal":
        return "rewrite_query"
    else:  # full
        return "transform_all"


def route_to_retrieval(state: RAGGraphState) -> str:
    """
    变换后选择检索策略:
      多查询 → multi_query_search
      单查询 → hybrid_search
    """
    queries = state.get("all_queries", [])
    if len(queries) > 2:
        return "multi_query_search"
    return "hybrid_search"


def route_to_rerank(state: RAGGraphState) -> str:
    """
    是否重排序:
      FAST → 跳过
      其他 → 重排序
    """
    mode = state.get("pipeline_mode", "normal")
    if mode == "fast":
        return "skip_rerank"
    return "rerank"


def route_after_quality(state: RAGGraphState) -> str:
    """
    质量门控后的路由:
      - 质量好 (≥0.6) → 写缓存
      - 质量差 + FULL 模式 + 未超重试 → 幻觉检测
      - 其他 → 写缓存 (接受当前答案)
    """
    score = state.get("quality_score", 1.0)
    mode = state.get("pipeline_mode", "")
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    remaining = state["budget_ms"] - sum(t["duration_ms"] for t in state.get("step_timings", []))

    if score >= 0.6:
        logger.info(f"[{state['trace_id']}] ✅ 质量通过 ({score:.2f})")
        return "write_cache"

    if mode == "full" and retry < max_retries and remaining > 1500:
        logger.info(
            f"[{state['trace_id']}] ⚠️ 质量不足 ({score:.2f}), "
            f"进入幻觉检测 (retry={retry})"
        )
        return "hallucination_check"

    logger.info(f"[{state['trace_id']}] 质量一般 ({score:.2f}), 但不再重试")
    return "write_cache"


def route_after_hallucination(state: RAGGraphState) -> str:
    """
    幻觉检测后:
      有幻觉 + 还能重试 → 修正
      否则 → 写缓存
    """
    check = state.get("hallucination_check", {})
    has_hallucination = check.get("has_hallucination", False)
    retry = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    remaining = state["budget_ms"] - sum(t["duration_ms"] for t in state.get("step_timings", []))

    if has_hallucination and retry < max_retries and remaining > 1200:
        logger.info(f"[{state['trace_id']}] 🔄 发现幻觉, 修正重试 (retry={retry})")
        return "refine_answer"

    return "write_cache"