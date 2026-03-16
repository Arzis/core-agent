# core/rag_pipeline.py
"""
★ 基于 LangGraph 的 RAG Pipeline
保持与原有 API 兼容, 底层用图执行
"""
import uuid
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from langchain_core.documents import Document

from core.adaptive_router import AdaptiveRouter
from core.async_llm_pool import llm_pool
from core.cache import cache_manager
from core.query_transformer import QueryTransformer
from core.reranker import Reranker
from core.response_synthesizer import ResponseSynthesizer
from graph.nodes import registry
from graph.rag_graph import RAGGraphExecutor
from graph.state import RAGGraphState
from core.document_processor import DocumentProcessor
from core.vector_store import MilvusVectorStore
from core.retriever import HybridRetriever
from core.logger import logger, console
from rich.panel import Panel


@dataclass
class RAGResult:
    query: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    hallucination_check: Dict = field(default_factory=dict)
    pipeline_mode: str = ""
    quality_score: float = 0.0
    total_duration_ms: float = 0.0
    cache_hit: bool = False
    over_budget: bool = False
    retry_count: int = 0
    step_timings: List[Dict] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnterpriseRAGPipeline:
    """
    企业级 RAG Pipeline — LangGraph 版

    底层使用 LangGraph StateGraph 编排所有步骤,
    支持条件路由、循环修正、流式输出。
    """

    def __init__(self):
        console.print(Panel(
            "[bold green]🚀 Enterprise RAG Pipeline (LangGraph)[/bold green]",
            border_style="green",
        ))

        # ── 创建所有模块实例 ──────────────────────
        self.doc_processor = DocumentProcessor()
        self.vector_store = MilvusVectorStore()
        self.retriever = HybridRetriever(self.vector_store)  # ★ 共享同一个 vector_store
        self.reranker = Reranker()
        self.transformer = QueryTransformer()
        self.synthesizer = ResponseSynthesizer()
        self.router = AdaptiveRouter()

        # ═══════════════════════════════════════════
        #   将所有实例注册到全局注册表
        #   这样 graph/nodes.py 中的节点函数
        #   通过 _get("vector_store") 拿到的是同一个对象
        # ═══════════════════════════════════════════
        self._register_modules()

        # 构建 LangGraph
        self.graph_executor = RAGGraphExecutor(enable_checkpointing=True)

        # ★ 启动时检查 vector_store 状态
        if self.vector_store.is_ready():
            logger.info(
                f"✅ Milvus 集合已就绪: "
                f"{self.vector_store.get_stats()}"
            )
        else:
            logger.warning(
                "⚠️ Milvus 集合未就绪, 请先调用 build_knowledge_base()"
            )

        logger.info("✅ 所有模块初始化 & 注册完成")

    def _register_modules(self):
        """
        ★ 将 Pipeline 持有的模块实例注册到全局注册表
        保证图节点和 Pipeline 使用完全相同的对象
        """
        registry.register("cache", cache_manager)
        registry.register("vector_store", self.vector_store)
        registry.register("retriever", self.retriever)
        registry.register("reranker", self.reranker)
        registry.register("transformer", self.transformer)
        registry.register("synthesizer", self.synthesizer)
        registry.register("router", self.router)
        registry.register("llm_pool", llm_pool)

        logger.info(f"已注册模块: {registry.registered_modules}")

    # ─── 知识库构建 ─────────────────────────────────
    def build_knowledge_base(
        self,
        source_path: str,
        chunk_strategy: str = "recursive",
        force_recreate: bool = False,
    ):
        logger.info(f"📚 构建知识库: {source_path}")

        # 1. 文档处理
        chunks = self.doc_processor.process(source_path, chunk_strategy)

        # 2. 向量存储
        # ★ create_collection 后, self.vector_store._collection 就有值了
        #   因为注册表中注册的是同一个对象, 节点函数也能看到
        self.vector_store.create_collection(force_recreate)
        self.vector_store.insert_documents(chunks)

        # 3. BM25 索引
        # ★ self.retriever 与注册表中的 retriever 是同一个对象
        #   build_bm25_index 后, 节点函数中的 retriever 也有 BM25 索引了
        lc_docs = [
            Document(page_content=c.page_content, metadata=c.metadata)
            for c in chunks
        ]
        self.retriever.build_bm25_index(lc_docs)

        logger.info(f"✅ 知识库构建完成: {len(chunks)} chunks, "
                     f"{self.vector_store.get_stats()}")

    # ─── 查询 (通过 LangGraph) ──────────────────
    def query(
        self,
        question: str,
        mode: str = None,
        budget_ms: float = 3000,
        thread_id: str = None,
    ) -> RAGResult:
        """
        执行 RAG 查询 — 底层走 LangGraph 状态图
        """
        # ★ 查询前检查就绪状态
        if not self.vector_store.is_ready():
            logger.error("知识库未就绪, 请先调用 build_knowledge_base()")
            return RAGResult(
                query=question,
                answer="系统错误: 知识库未初始化, 请先构建知识库。",
            )

        if thread_id is None:
            thread_id = str(uuid.uuid4())[:8]

        result = self.graph_executor.invoke(
            query=question,
            budget_ms=budget_ms,
            mode=mode,
            thread_id=thread_id,
        )

        return RAGResult(
            query=question,
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            hallucination_check=result.get("hallucination_check", {}),
            pipeline_mode=result.get("pipeline_mode", ""),
            quality_score=result.get("quality_score", 0),
            total_duration_ms=result.get("total_duration_ms", 0),
            cache_hit=result.get("cache_hit", False),
            over_budget=result.get("over_budget", False),
            retry_count=result.get("retry_count", 0),
            step_timings=result.get("step_timings", []),
            metadata=result,
        )

    def stream_query(self, question: str, mode: str = None, budget_ms: float = 3000):
        """
        ★ 流式查询: 实时输出每个节点的处理结果
        适合前端 SSE / WebSocket 场景
        """
        console.print(f"\n[bold]🔍 流式查询: {question}[/bold]\n")
        final = {}
        for event in self.graph_executor.stream(question, budget_ms, mode):
            final = event.get("output", {})
            yield event
        return final

    def fast_query(self, question: str) -> RAGResult:
        return self.query(question, mode="fast", budget_ms=1000)

    def full_query(self, question: str) -> RAGResult:
        return self.query(question, mode="full", budget_ms=8000)

    def export_graph(self, path: str = "rag_graph.png"):
        """导出图可视化"""
        self.graph_executor.get_graph_image(path)

    def get_graph_mermaid(self) -> str:
        """获取 Mermaid 图定义 (用于文档)"""
        return self.graph_executor.get_graph_mermaid()