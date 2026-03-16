# graph/rag_graph.py
"""
★ 主 RAG 图: 构建、编译、执行

使用 LangGraph 的 StateGraph 构建有向图:
  - 节点 (Node): 每个处理步骤
  - 条件边 (Conditional Edge): 动态路由
  - 循环 (Cycle): 质量不达标时重试
"""
import time
import uuid
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from graph.state import RAGGraphState, create_initial_state
from graph.nodes import (
    cache_check_node,
    route_query_node,
    rewrite_query_node,
    transform_all_node,
    hybrid_search_node,
    multi_query_search_node,
    rerank_node,
    skip_rerank_node,
    generate_node,
    quality_gate_node,
    hallucination_check_node,
    refine_answer_node,
    write_cache_node,
    finalize_node,
)
from graph.edges import (
    should_use_cache,
    route_by_mode,
    route_to_retrieval,
    route_to_rerank,
    route_after_quality,
    route_after_hallucination,
)
from core.logger import logger, perf_tracker, console

from rich.table import Table
from rich import box


def build_rag_graph() -> StateGraph:
    """
    构建 RAG 状态图

    ┌─ START ─┐
    │         ▼
    │   cache_check ──── hit ──▶ finalize ──▶ END
    │         │ miss
    │         ▼
    │   route_query
    │     │ FAST    │ NORMAL     │ FULL
    │     ▼         ▼            ▼
    │ hybrid    rewrite      transform_all
    │ _search   _query          │
    │     │         │            ▼
    │     │         │     ┌── route_to_retrieval
    │     │         ▼     │ (>2 queries)  │ (≤2)
    │     │    hybrid     ▼               ▼
    │     │    _search  multi_query   hybrid
    │     │         │    _search      _search
    │     │         └────────┬────────────┘
    │     ▼                  ▼
    │  ┌── route_to_rerank ──┐
    │  │ FAST      │ other   │
    │  ▼           ▼         │
    │ skip      rerank       │
    │ _rerank      │         │
    │  └───────┬───┘         │
    │          ▼             │
    │      generate          │
    │          ▼             │
    │    quality_gate        │
    │     │ pass    │ fail   │
    │     ▼         ▼        │
    │ write_cache  hallucination_check
    │     │              │
    │     │         ┌────┴────┐
    │     │    no_issue    has_issue
    │     │         │         ▼
    │     │         │    refine_answer ──▶ generate (loop!)
    │     │         ▼
    │     │    write_cache
    │     ▼
    │  finalize
    │     ▼
    └── END
    """
    graph = StateGraph(RAGGraphState)

    # ─── 添加所有节点 ─────────────────────────────
    graph.add_node("cache_check", cache_check_node)
    graph.add_node("route_query", route_query_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("transform_all", transform_all_node)
    graph.add_node("hybrid_search", hybrid_search_node)
    graph.add_node("multi_query_search", multi_query_search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("skip_rerank", skip_rerank_node)
    graph.add_node("generate", generate_node)
    graph.add_node("quality_gate", quality_gate_node)
    graph.add_node("hallucination_check", hallucination_check_node)
    graph.add_node("refine_answer", refine_answer_node)
    graph.add_node("write_cache", write_cache_node)
    graph.add_node("finalize", finalize_node)

    # ─── 入口边 ──────────────────────────────────
    graph.add_edge(START, "cache_check")

    # ─── 缓存检查 → 命中/未命中 ─────────────────
    graph.add_conditional_edges(
        "cache_check",
        should_use_cache,
        {
            "finalize": "finalize",
            "route_query": "route_query",
        },
    )

    # ─── 路由 → FAST/NORMAL/FULL ────────────────
    graph.add_conditional_edges(
        "route_query",
        route_by_mode,
        {
            "hybrid_search": "hybrid_search",      # FAST: 直接检索
            "rewrite_query": "rewrite_query",       # NORMAL: 先改写
            "transform_all": "transform_all",       # FULL: 全量变换
        },
    )

    # ─── NORMAL: 改写后 → 检索 ──────────────────
    graph.add_edge("rewrite_query", "hybrid_search")

    # ─── FULL: 变换后 → 选择检索方式 ────────────
    graph.add_conditional_edges(
        "transform_all",
        route_to_retrieval,
        {
            "multi_query_search": "multi_query_search",
            "hybrid_search": "hybrid_search",
        },
    )

    # ─── 所有检索 → 选择是否重排 ────────────────
    graph.add_conditional_edges(
        "hybrid_search",
        route_to_rerank,
        {
            "rerank": "rerank",
            "skip_rerank": "skip_rerank",
        },
    )
    graph.add_conditional_edges(
        "multi_query_search",
        route_to_rerank,
        {
            "rerank": "rerank",
            "skip_rerank": "skip_rerank",
        },
    )

    # ─── 重排 → 生成 ────────────────────────────
    graph.add_edge("rerank", "generate")
    graph.add_edge("skip_rerank", "generate")

    # ─── 生成 → 质量评估 ────────────────────────
    graph.add_edge("generate", "quality_gate")

    # ─── 质量门控 → 通过/重试 ───────────────────
    graph.add_conditional_edges(
        "quality_gate",
        route_after_quality,
        {
            "write_cache": "write_cache",
            "hallucination_check": "hallucination_check",
        },
    )

    # ─── 幻觉检测 → 修正/通过 ───────────────────
    graph.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination,
        {
            "refine_answer": "refine_answer",
            "write_cache": "write_cache",
        },
    )

    # ─── ★ 修正 → 回到生成 (循环!) ──────────────
    graph.add_edge("refine_answer", "generate")

    # ─── 写缓存 → 汇总 → END ───────────────────
    graph.add_edge("write_cache", "finalize")
    graph.add_edge("finalize", END)

    return graph


class RAGGraphExecutor:
    """
    ★ RAG 图执行器
    - 编译图
    - 执行查询
    - 流式输出
    - 检查点持久化
    """

    def __init__(self, enable_checkpointing: bool = True):
        logger.info("🔨 构建 RAG Graph...")

        self.graph = build_rag_graph()
        self.enable_checkpointing = enable_checkpointing

        # ★ 检查点: 支持中断恢复 & 历史回放
        self.checkpointer = MemorySaver() if enable_checkpointing else None
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)

        logger.info("✅ RAG Graph 编译完成")

    def _build_config(self, thread_id: str = None) -> dict:
        """
        构造 LangGraph 执行配置

        ★ 核心修复:
          - 如果启用了 checkpointer, 必须提供 thread_id
          - 如果调用方没传, 自动生成一个临时 thread_id
        """
        if not self.enable_checkpointing:
            return {}

        # 没传 thread_id → 自动生成 (每次查询独立, 不共享历史)
        if thread_id is None:
            thread_id = f"rag_{uuid.uuid4().hex[:8]}"

        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    def invoke(
        self,
        query: str,
        budget_ms: float = 3000,
        mode: str = None,
        thread_id: str = None,
    ) -> Dict[str, Any]:
        """
        执行一次完整查询
        """
        initial_state = create_initial_state(
            query=query,
            budget_ms=budget_ms,
            forced_mode=mode,
        )

        config = self._build_config(thread_id)

        logger.info(
            f"▶ 执行 RAG Graph: query='{query[:50]}', "
            f"budget={budget_ms}ms, thread={config.get('configurable', {}).get('thread_id', 'N/A')}"
        )
        start = time.perf_counter()

        final_state = self.compiled.invoke(initial_state, config=config)

        total_ms = (time.perf_counter() - start) * 1000

        # 打印耗时报告
        self._print_trace_report(final_state, total_ms)

        return {
            "query": query,
            "answer": final_state.get("answer", ""),
            "sources": final_state.get("sources", []),
            "pipeline_mode": final_state.get("pipeline_mode", ""),
            "quality_score": final_state.get("quality_score", 0),
            "hallucination_check": final_state.get("hallucination_check", {}),
            "cache_hit": final_state.get("cache_hit", False),
            "total_duration_ms": total_ms,
            "budget_ms": budget_ms,
            "over_budget": total_ms > budget_ms,
            "retry_count": final_state.get("retry_count", 0),
            "step_timings": final_state.get("step_timings", []),
        }

    def stream(
        self,
        query: str,
        budget_ms: float = 3000,
        mode: str = None,
        thread_id: str = None,
    ):
        """
        ★ 流式执行: 逐步输出每个节点的结果
        用于前端实时展示处理进度
        """
        initial_state = create_initial_state(
            query=query, budget_ms=budget_ms, forced_mode=mode,
        )

        # ★ 修复: 同样使用统一方法
        config = self._build_config(thread_id)

        logger.info(f"▶ 流式执行 RAG Graph: '{query[:50]}'")

        for event in self.compiled.stream(
                initial_state,
                config=config,
                stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                timings = node_output.get("step_timings", [])
                for t in timings:
                    icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(t["status"], "❓")
                    console.print(
                        f"  {icon} [cyan]{t['name']:.<30}[/cyan] "
                        f"[magenta]{t['duration_ms']:>7.1f}ms[/magenta]  "
                        f"[dim]{t.get('metadata', {})}[/dim]"
                    )
                yield {"node": node_name, "output": node_output}

    def _print_trace_report(self, state: Dict, total_ms: float):
        """打印耗时报告"""
        timings = state.get("step_timings", [])
        budget = state.get("budget_ms", 0)

        table = Table(
            title=f"🔍 RAG Graph Trace [{state.get('trace_id', '?')}]",
            box=box.ROUNDED, show_lines=True,
        )
        table.add_column("节点", style="cyan", width=28)
        table.add_column("耗时(ms)", style="magenta", justify="right", width=10)
        table.add_column("占比", style="green", justify="right", width=8)
        table.add_column("状态", justify="center", width=6)
        table.add_column("详情", style="dim", max_width=40)

        timed_sum = sum(t["duration_ms"] for t in timings)

        for t in timings:
            pct = (t["duration_ms"] / timed_sum * 100) if timed_sum > 0 else 0
            icon = {"success": "✅", "skipped": "⏭️", "failed": "❌"}.get(t["status"], "❓")
            meta = ", ".join(f"{k}={v}" for k, v in t.get("metadata", {}).items())

            table.add_row(
                t["name"],
                f"{t['duration_ms']:.1f}" if t["duration_ms"] > 0 else "-",
                f"{pct:.1f}%" if t["duration_ms"] > 0 else "-",
                icon,
                meta[:40] if meta else "-",
            )

        over = total_ms > budget if budget > 0 else False
        style = "bold red" if over else "bold green"
        table.add_row(
            "[bold]TOTAL (wall)[/bold]",
            f"[{style}]{total_ms:.1f}[/]",
            f"budget: {budget:.0f}" if budget > 0 else "-",
            "⚠️" if over else "✅",
            f"mode={state.get('pipeline_mode', '?')}, retries={state.get('retry_count', 0)}",
        )

        console.print(table)

    def get_graph_image(self, output_path: str = "rag_graph.png"):
        """★ 导出图可视化"""
        try:
            img_data = self.compiled.get_graph().draw_mermaid_png()
            with open(output_path, "wb") as f:
                f.write(img_data)
            logger.info(f"图已导出: {output_path}")
        except Exception as e:
            logger.warning(f"图导出失败 (需要安装 graphviz): {e}")
            # Fallback: 输出 Mermaid 文本
            mermaid = self.compiled.get_graph().draw_mermaid()
            mermaid_path = output_path.replace(".png", ".md")
            with open(mermaid_path, "w") as f:
                f.write(f"```mermaid\n{mermaid}\n```")
            logger.info(f"Mermaid 图已导出: {mermaid_path}")

    def get_graph_mermaid(self) -> str:
        """获取 Mermaid 格式的图定义"""
        return self.compiled.get_graph().draw_mermaid()