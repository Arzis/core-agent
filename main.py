# main.py
"""
低延迟企业 RAG 系统
  python main.py demo           # 完整演示
  python main.py query "问题"    # 自动模式查询
  python main.py query "问题" --fast   # 极速模式
"""
import argparse
from pathlib import Path
from core.rag_pipeline import EnterpriseRAGPipeline
from core.logger import perf_tracker, console
from core.cache import cache_manager
from rich.panel import Panel


def create_demo_docs():
    demo_dir = Path("./demo_docs")
    demo_dir.mkdir(exist_ok=True)
    docs = {
        "company.txt": """
ABC科技有限公司成立于2015年，总部位于北京海淀区。
公司专注于人工智能和大数据技术，拥有员工1200人，研发人员占比65%。

核心业务：
1. 智能客服解决方案 - 已服务超过500家企业客户
2. 数据分析平台 - 提供实时数据分析和可视化服务
3. AI模型训练服务 - 为企业提供定制化AI模型训练

2023年年度营收15.8亿元人民币，同比增长32%。
2024年上半年营收9.2亿元，预计全年突破20亿元。
""",
        "finance.txt": """
2023年度财务报告摘要

营收情况：全年营收15.8亿元，较2022年12.0亿元增长31.7%。
- 智能客服业务收入8.5亿元，占比53.8%，同比增长28%
- 数据分析平台收入4.2亿元，占比26.6%，同比增长45%
- AI模型训练服务收入3.1亿元，占比19.6%，同比增长25%

利润：净利润2.4亿元，净利润率15.2%，毛利率62.5%。
研发投入3.2亿元，占营收20.3%，新增专利87项，累计356项。
""",
        "product.txt": """
智能客服系统V3.0产品说明

核心功能：
1. 多轮对话引擎 - 支持上下文理解，最长50轮对话
2. 知识库管理 - 自动化知识抽取和更新
3. 情感分析 - 实时分析客户情绪
4. 多渠道接入 - 网页、APP、微信、电话

技术指标：
- 意图识别准确率: 96.5%
- 平均响应时间: 200ms
- 并发处理: 10000 QPS

定价：基础版5万/年(100坐席)，专业版15万/年(500坐席)，企业版定制。
""",
    }
    for name, content in docs.items():
        (demo_dir / name).write_text(content.strip(), encoding="utf-8")
    return str(demo_dir)


def demo():
    console.print(Panel(
        "[bold green]🚀 Enterprise RAG — 低延迟优化版[/bold green]\n"
        "LangChain + LlamaIndex + Milvus\n"
        "优化: 并发LLM / 批量检索 / 多级缓存 / 自适应路由",
        border_style="green",
    ))

    pipeline = EnterpriseRAGPipeline()

    # 构建知识库
    demo_path = create_demo_docs()
    console.print("\n[bold yellow]📚 构建知识库[/bold yellow]")
    pipeline.build_knowledge_base(demo_path, force_recreate=True)

    # 测试查询
    test_cases = [
        # (问题, 模式, 预期模式说明)
        ("公司有多少员工？", "fast", "简单问题 → 极速模式"),
        ("公司2023年各业务板块营收占比是多少？", None, "中等问题 → 自动路由"),
        ("分析公司的营收增长趋势，各业务的增长率对比，以及研发投入回报", "full", "复杂问题 → 完整模式"),
    ]

    for q, mode, desc in test_cases:
        console.print(f"\n{'═' * 60}")
        console.print(f"[bold cyan]{desc}[/bold cyan]")
        console.print(f"[bold]Q: {q}[/bold]")

        if mode == "fast":
            result = pipeline.fast_query(q)
        elif mode == "full":
            result = pipeline.full_query(q)
        else:
            result = pipeline.query(q)

        console.print(f"\n[green]📝 答案:[/green] {result.answer[:300]}")
        console.print(
            f"[dim]模式: {result.pipeline_mode} | "
            f"耗时: {result.metadata.get('total_duration_ms', 0):.0f}ms | "
            f"超预算: {result.metadata.get('over_budget', False)}[/dim]"
        )

    # ★ 缓存命中测试
    console.print(f"\n{'═' * 60}")
    console.print("[bold yellow]⚡ 缓存命中测试 (重复查询)[/bold yellow]")
    result = pipeline.query("公司有多少员工？")
    console.print(
        f"缓存命中! 耗时: {result.metadata.get('total_duration_ms', 0):.0f}ms"
    )

    # 汇总
    console.print(f"\n{'═' * 60}")
    perf_tracker.print_summary()
    cache_manager.print_stats()


def main():
    parser = argparse.ArgumentParser(description="Low-Latency Enterprise RAG")
    parser.add_argument("action", choices=["demo", "build", "query"])
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--source", type=str, default="./demo_docs")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--budget", type=int, default=None, help="延迟预算(ms)")

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
        print(f"\n答案: {result.answer}")


if __name__ == "__main__":
    main()