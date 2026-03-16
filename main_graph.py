# main.py
"""
Enterprise RAG — LangGraph 版

使用:
  python main.py demo                          # 完整演示
  python main.py query "问题"                   # 自动模式
  python main.py query "问题" --fast            # 极速模式
  python main.py query "问题" --full            # 完整模式
  python main.py stream "问题"                  # 流式输出
  python main.py graph                         # 导出图
"""
import argparse
from pathlib import Path

from core.rag_pipeline import EnterpriseRAGPipeline
from core.cache import cache_manager
from core.logger import console

from rich.panel import Panel
from rich.table import Table
from rich import box


def create_demo_docs():
    demo_dir = Path("./demo_docs")
    demo_dir.mkdir(exist_ok=True)
    docs = {
        "company.txt": """ABC科技有限公司成立于2015年，总部位于北京海淀区。
公司专注于人工智能和大数据技术，拥有员工1200人，研发人员占比65%。
核心业务：
1. 智能客服解决方案 - 已服务超过500家企业客户
2. 数据分析平台 - 提供实时数据分析和可视化服务
3. AI模型训练服务 - 为企业提供定制化AI模型训练
2023年年度营收15.8亿元人民币，同比增长32%。
2024年上半年营收9.2亿元，预计全年突破20亿元。""",

        "finance.txt": """2023年度财务报告摘要
营收情况：全年营收15.8亿元，较2022年12.0亿元增长31.7%。
- 智能客服业务收入8.5亿元，占比53.8%，同比增长28%
- 数据分析平台收入4.2亿元，占比26.6%，同比增长45%
- AI模型训练服务收入3.1亿元，占比19.6%，同比增长25%
利润：净利润2.4亿元，净利润率15.2%，毛利率62.5%。
研发投入3.2亿元，占营收20.3%，新增专利87项，累计356项。""",

        "product.txt": """智能客服系统V3.0产品说明
核心功能：
1. 多轮对话引擎 - 支持上下文理解，最长50轮对话
2. 知识库管理 - 自动化知识抽取和更新
3. 情感分析 - 实时分析客户情绪
4. 多渠道接入 - 网页、APP、微信、电话
技术指标：意图识别准确率96.5%，平均响应时间200ms，并发10000QPS
定价：基础版5万/年(100坐席)，专业版15万/年(500坐席)，企业版定制。""",
    }
    for name, content in docs.items():
        (demo_dir / name).write_text(content.strip(), encoding="utf-8")
    return str(demo_dir)


def demo():
    console.print(Panel(
        "[bold green]🚀 Enterprise RAG — LangGraph Edition[/bold green]\n"
        "StateGraph 状态图 | 条件路由 | 循环修正 | 流式输出",
        border_style="green",
    ))

    pipeline = EnterpriseRAGPipeline()

    # 导出图
    console.print("\n[bold yellow]📊 导出 LangGraph 可视化[/bold yellow]")
    mermaid = pipeline.get_graph_mermaid()
    console.print(f"[dim]{mermaid[:200]}...[/dim]")
    pipeline.export_graph("rag_graph.png")

    # 构建知识库
    demo_path = create_demo_docs()
    console.print("\n[bold yellow]📚 构建知识库[/bold yellow]")
    pipeline.build_knowledge_base(demo_path, force_recreate=True)

    # ─── 测试用例 ─────────────────────────────────
    test_cases = [
        {
            "query": "公司有多少员工？",
            "mode": "fast",
            "budget": 1000,
            "desc": "🏃 FAST 模式 — 简单事实查询",
        },
        {
            "query": "公司2023年各业务板块营收占比是多少？",
            "mode": None,         # 自动路由
            "budget": 3000,
            "desc": "🚶 AUTO 模式 — 中等复杂度",
        },
        {
            "query": "分析公司的营收增长趋势，各业务增长率对比，以及研发投入产出比",
            "mode": "full",
            "budget": 8000,
            "desc": "🔬 FULL 模式 — 复杂分析",
        },
    ]

    results = []
    for tc in test_cases:
        console.print(f"\n{'═' * 60}")
        console.print(f"[bold cyan]{tc['desc']}[/bold cyan]")
        console.print(f"[bold]Q: {tc['query']}[/bold]")

        result = pipeline.query(
            question=tc["query"],
            mode=tc["mode"],
            budget_ms=tc["budget"],
        )
        results.append(result)

        console.print(f"\n[green]📝 答案:[/green] {result.answer[:300]}...")
        console.print(
            f"[dim]模式: {result.pipeline_mode} | "
            f"耗时: {result.total_duration_ms:.0f}ms | "
            f"质量: {result.quality_score:.2f} | "
            f"重试: {result.retry_count} | "
            f"超预算: {result.over_budget}[/dim]"
        )

    # ─── 缓存命中测试 ────────────────────────────
    console.print(f"\n{'═' * 60}")
    console.print("[bold yellow]⚡ 缓存命中测试[/bold yellow]")
    cache_result = pipeline.query("公司有多少员工？")
    console.print(
        f"[green]缓存命中: {cache_result.cache_hit} | "
        f"耗时: {cache_result.total_duration_ms:.1f}ms[/green]"
    )

    # ─── 流式查询演示 ────────────────────────────
    console.print(f"\n{'═' * 60}")
    console.print("[bold yellow]🌊 流式查询演示[/bold yellow]")
    for event in pipeline.stream_query("智能客服系统的定价是多少？", budget_ms=3000):
        pass  # stream_query 内部已打印

    # ─── 汇总报告 ─────────────────────────────────
    console.print(f"\n{'═' * 60}")
    console.print("[bold yellow]📊 汇总报告[/bold yellow]")

    summary = Table(title="查询结果汇总", box=box.ROUNDED, show_lines=True)
    summary.add_column("查询", max_width=35)
    summary.add_column("模式", width=8)
    summary.add_column("耗时(ms)", justify="right", width=10)
    summary.add_column("预算(ms)", justify="right", width=10)
    summary.add_column("质量", justify="right", width=6)
    summary.add_column("状态", justify="center", width=6)

    for r, tc in zip(results, test_cases):
        over = r.total_duration_ms > tc["budget"]
        summary.add_row(
            r.query[:35],
            r.pipeline_mode,
            f"{r.total_duration_ms:.0f}",
            f"{tc['budget']}",
            f"{r.quality_score:.2f}",
            "⚠️" if over else "✅",
        )
    console.print(summary)

    # 缓存统计
    cache_manager.print_stats()


def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG — LangGraph")
    parser.add_argument("action", choices=["demo", "build", "query", "stream", "graph"])
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--source", type=str, default="./demo_docs")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--budget", type=int, default=3000)

    args = parser.parse_args()

    if args.action == "demo":
        demo()

    elif args.action == "build":
        pipe = EnterpriseRAGPipeline()
        pipe.build_knowledge_base(args.source, force_recreate=True)

    elif args.action == "query":
        if not args.question:
            print("用法: python main.py query '你的问题'")
            return
        pipe = EnterpriseRAGPipeline()
        mode = "fast" if args.fast else ("full" if args.full else None)
        result = pipe.query(args.question, mode=mode, budget_ms=args.budget)
        console.print(f"\n[bold green]答案:[/bold green] {result.answer}")

    elif args.action == "stream":
        if not args.question:
            print("用法: python main.py stream '你的问题'")
            return
        pipe = EnterpriseRAGPipeline()
        for event in pipe.stream_query(args.question, budget_ms=args.budget):
            pass

    elif args.action == "graph":
        pipe = EnterpriseRAGPipeline()
        pipe.export_graph("rag_graph.png")
        console.print("[green]图已导出: rag_graph.png[/green]")
        mermaid = pipe.get_graph_mermaid()
        console.print(f"\nMermaid:\n{mermaid}")


if __name__ == "__main__":
    main()