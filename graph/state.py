"""
★ LangGraph 状态定义
  - 所有节点共享的状态容器
  - 使用 TypedDict + Annotated 实现状态合并策略
"""
import operator
import time
import uuid
from typing import Any, TypedDict, Optional, Dict, List, Annotated


def merge_lists(a: list, b: list) -> list:
    """合并两个列表 (用于并行节点结果合并)"""
    return a + b

def merge_dicts(a: dict, b: dict) -> dict:
    """合并字典, 后者覆盖前者"""
    return {**a, **b}

def keep_latest(a: Any, b: Any) -> Any:
    """保留最新值"""
    return b

# ─── 步骤耗时记录 ─────────────────────────────────
class StepTiming(TypedDict):
    name: str
    duration_ms: float
    status: str
    metadata: dict[str, Any]

# ─── 核心图状态 ───────────────────────────────────
class RAGGraphState(TypedDict):
    """
    RAG 图的全局状态, 所有节点读写此对象

    Annotated 的第二个参数是 reducer:
    - operator.add: 列表追加
    - keep_latest: 覆盖
    """
    # ─ 输入 ─
    query: str  # 原始查询
    budget_ms: float  # 延迟预算
    forced_mode: Optional[str]  # 强制模式

    # ─ 路由 ─
    pipeline_mode: str  # fast/normal/full
    query_complexity: str  # simple/medium/complex
    steps_config: Dict[str, bool]  # 各步骤开关

    # ─ 查询变换 ─
    rewritten_query: str
    hyde_document: str
    step_back_query: str
    sub_queries: List[str]
    all_queries: Annotated[List[str], merge_lists]  # ★ 并行节点合并

    # ─ 检索 ─
    retrieved_documents: List[Dict[str, Any]]
    retrieval_method: str

    # ─ 重排序 ─
    reranked_documents: List[Dict[str, Any]]

    # ─ 生成 ─
    context: str
    answer: str
    sources: List[Dict[str, Any]]

    # ─ 质量控制 ─
    hallucination_check: Dict[str, Any]
    quality_score: float  # 0-1
    retry_count: int  # 已重试次数
    max_retries: int  # 最大重试

    # ─ 缓存 ─
    cache_hit: bool
    cache_key: str

    # ─ 追踪 ─
    trace_id: str
    start_time: float
    step_timings: Annotated[List[StepTiming], operator.add]  # ★ 自动追加
    total_duration_ms: float
    error: Optional[str]

def create_initial_state(
    query: str,
    budget_ms: float = 3000,
    forced_mode: str = None,
) -> RAGGraphState:
    """创建初始状态"""
    return RAGGraphState(
        query=query,
        budget_ms=budget_ms,
        forced_mode=forced_mode,
        pipeline_mode="",
        query_complexity="",
        steps_config={},
        rewritten_query="",
        hyde_document="",
        step_back_query="",
        sub_queries=[],
        all_queries=[query],
        retrieved_documents=[],
        retrieval_method="",
        reranked_documents=[],
        context="",
        answer="",
        sources=[],
        hallucination_check={},
        quality_score=0.0,
        retry_count=0,
        max_retries=2,
        cache_hit=False,
        cache_key=query.strip().lower(),
        trace_id=str(uuid.uuid4())[:8],
        start_time=time.perf_counter(),
        step_timings=[],
        total_duration_ms=0.0,
        error=None,
    )