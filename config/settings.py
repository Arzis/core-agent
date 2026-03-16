import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

# @dataclass
# class EmbeddingConfig:
#     model_name: str = "text-embedding-3-small"
#     dimension: int = 1536
#     batch_size: int = 64
#     use_local: bool = False
#     local_model_path: str = "BAAI/bge-large-zh-v1.5"
#     local_dimension: int = 1024

@dataclass
class EmbeddingConfig:
    model_name: str = "text-embedding-v4" # text-embedding-v4
    dimension: int = 1024
    batch_size: int = 64
    use_local: bool = False
    local_model_path: str = "BAAI/bge-large-zh-v1.5"
    local_dimension: int = 1024

@dataclass
class MilvusConfig:
    host: str = "localhost"
    port: int = 19530
    collection_name: str = "enterprise_rag"
    index_type: str = "HNSW"           # ★ HNSW 比 IVF_FLAT 查询更快
    metric_type: str = "COSINE"
    # HNSW 参数
    M: int = 16
    efConstruction: int = 256
    ef: int = 128                      # 查询时的 ef, 越大越准但越慢
    top_k: int = 20
    # ★ 连接池
    pool_size: int = 5
    warmup_on_init: bool = True

@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: List[str] = field(default_factory=lambda: [
        "\n\n", "\n", "。", "！", "？", ".", "!", "?", " "
    ])
    use_semantic_chunking: bool = False  # ★ 默认关闭, 按需开启
    parent_chunk_size: int = 2048
    child_chunk_size: int = 256

@dataclass
class RetrieverConfig:
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    top_k_dense: int = 15
    top_k_sparse: int = 15
    top_k_final: int = 8
    use_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_k: int = 5

@dataclass
class QueryTransformConfig:
    enable_query_rewrite: bool = True
    enable_hyde: bool = True
    enable_step_back: bool = True
    enable_sub_query: bool = True
    max_sub_queries: int = 3

# @dataclass
# class LLMConfig:
#     model_name: str = "gpt-4o"
#     fast_model: str = "gpt-4o-mini"    # ★ 轻量任务用小模型
#     temperature: float = 0.1
#     max_tokens: int = 2048
#     api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
#     base_url: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL"))
#     # ★ 并发控制
#     max_concurrent_calls: int = 8
#     request_timeout: int = 15          # 单次请求超时(秒)

@dataclass
class LLMConfig:
    model_name: str = "qwen-max"
    fast_model: str = "qwen3-vl-flash"    # ★ 轻量任务用小模型
    temperature: float = 0.1
    max_tokens: int = 2048
    api_key: str = field(default_factory=lambda: os.getenv("DASHSCOPE_API_KEY", ""))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("DASHSCOPE_BASE_URL"))
    # ★ 并发控制
    max_concurrent_calls: int = 8
    request_timeout: int = 60          # 单次请求超时(秒)

@dataclass
class CacheConfig:
    """★ 缓存配置"""
    enable: bool = True
    # 嵌入缓存
    embedding_cache_size: int = 10000
    # 查询结果缓存
    query_cache_size: int = 500
    query_cache_ttl: int = 3600        # 秒
    # LLM 响应缓存
    llm_cache_size: int = 1000
    llm_cache_ttl: int = 1800

@dataclass
class LatencyConfig:
    """★ 延迟预算控制"""
    # 总延迟预算 (ms)
    total_budget_ms: int = 3000
    # 各步骤预算
    query_transform_budget_ms: int = 800
    retrieval_budget_ms: int = 300
    rerank_budget_ms: int = 500
    generation_budget_ms: int = 1500
    # 自适应降级阈值
    degrade_threshold_ms: int = 2000
    # 超时直接返回缓存/降级结果
    hard_timeout_ms: int = 5000


@dataclass
class RAGConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    query_transform: QueryTransformConfig = field(default_factory=QueryTransformConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    latency: LatencyConfig = field(default_factory=LatencyConfig)
    log_level: str = "INFO"


rag_config = RAGConfig()