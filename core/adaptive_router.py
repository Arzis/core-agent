# core/adaptive_router.py
"""
★ 自适应管线路由器
根据查询复杂度 + 已消耗时间 + 延迟预算, 动态决定执行哪些步骤

三档管线:
  FULL:    查询变换 → 多路检索 → 重排序 → 生成 → 幻觉检测 → 修正
  NORMAL:  查询改写 → 混合检索 → 重排序 → 生成
  FAST:    直接检索 → 生成
"""
from typing import Dict, Any
from enum import Enum

from langchain_core.prompts import ChatPromptTemplate

from config.settings import rag_config
from core.logger import logger


class PipelineMode(Enum):
    FAST = "fast"
    NORMAL = "normal"
    FULL = "full"


CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """判断用户查询的复杂度。只输出一个词: simple / medium / complex

simple: 单一事实性问题, 如"公司有多少员工?"
medium: 需要综合信息, 如"各业务板块的营收对比"
complex: 多条件/分析性问题, 如"分析营收下降原因并给出建议"
"""),
    ("human", "{query}"),
])


class AdaptiveRouter:
    """★ 自适应管线路由"""

    def __init__(self):
        self.latency_cfg = rag_config.latency

    def classify_query(self, query: str) -> str:
        """快速判断查询复杂度"""
        # ★ 基于规则的快速判断 (省去 LLM 调用)
        q_len = len(query)
        keyword_complex = any(w in query for w in ["分析", "对比", "原因", "建议", "趋势", "预测", "为什么", "如何"])
        keyword_multi = any(w in query for w in ["以及", "同时", "并且", "哪些", "分别"])

        if q_len < 15 and not keyword_complex:
            return "simple"
        elif keyword_complex or (keyword_multi and q_len > 30):
            return "complex"
        else:
            return "medium"

    def decide_mode(
        self,
        query: str,
        forced_mode: str = None,
        remaining_budget_ms: float = None,
    ) -> Dict[str, Any]:
        """
        决定管线模式和各步骤配置
        返回: {"mode": PipelineMode, "steps": {...}}
        """
        if forced_mode:
            mode = PipelineMode(forced_mode)
        else:
            complexity = self.classify_query(query)
            logger.info(f"查询复杂度: {complexity}")

            mode_map = {
                "simple": PipelineMode.FAST,
                "medium": PipelineMode.NORMAL,
                "complex": PipelineMode.FULL,
            }
            mode = mode_map.get(complexity, PipelineMode.NORMAL)

        # ★ 如果预算紧张, 降级
        if remaining_budget_ms is not None:
            if remaining_budget_ms < 1500 and mode == PipelineMode.FULL:
                mode = PipelineMode.NORMAL
                logger.info(f"预算不足({remaining_budget_ms:.0f}ms), 降级: FULL → NORMAL")
            elif remaining_budget_ms < 800 and mode == PipelineMode.NORMAL:
                mode = PipelineMode.FAST
                logger.info(f"预算不足({remaining_budget_ms:.0f}ms), 降级: NORMAL → FAST")

        # 根据模式生成步骤配置
        steps = self._get_steps(mode)
        logger.info(f"管线模式: {mode.value} → {[s for s, v in steps.items() if v]}")
        return {"mode": mode, "steps": steps}

    def _get_steps(self, mode: PipelineMode) -> Dict[str, bool]:
        if mode == PipelineMode.FAST:
            return {
                "query_transform": False,
                "multi_query_retrieval": False,
                "hybrid_retrieval": True,     # 至少做混合检索
                "rerank": False,
                "generation": True,
                "hallucination_check": False,
                "refine": False,
            }
        elif mode == PipelineMode.NORMAL:
            return {
                "query_transform": True,      # 只做改写
                "multi_query_retrieval": False,
                "hybrid_retrieval": True,
                "rerank": True,
                "generation": True,
                "hallucination_check": False,
                "refine": False,
            }
        else:  # FULL
            return {
                "query_transform": True,      # 全部变换
                "multi_query_retrieval": True,
                "hybrid_retrieval": True,
                "rerank": True,
                "generation": True,
                "hallucination_check": True,
                "refine": True,
            }

    def should_check_hallucination(self, answer: str, sources: list, remaining_ms: float) -> bool:
        """★ 动态决定是否做幻觉检测"""
        # 预算不够就不检测
        if remaining_ms < 1000:
            return False
        # 没有引用来源 → 应该检测
        if not sources:
            return True
        # 答案太长 → 可能有幻觉, 应该检测
        if len(answer) > 500:
            return True
        return False