# core/vector_store.py
"""
★ 优化版 Milvus — 连接池 + HNSW 索引 + 查询预热
"""
import hashlib
import json
from typing import List, Dict, Any, Optional

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.documents import Document
from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility,
)
from langchain_openai import OpenAIEmbeddings

from config.settings import rag_config
from core.cache import cache_manager
from core.logger import logger, timed


class EmbeddingService:
    """★ 嵌入服务: 带缓存 + 批量优化"""

    def __init__(self):
        cfg = rag_config.embedding
        # self._model = OpenAIEmbeddings(
        #     model=cfg.model_name,
        #     openai_api_key=rag_config.llm.api_key,
        #     openai_api_base=rag_config.llm.base_url,
        # )
        self._model = DashScopeEmbeddings(
            model=cfg.model_name,
            dashscope_api_key=rag_config.llm.api_key,
        )

    def embed_query(self, text: str) -> List[float]:
        """★ 单条查询嵌入 (缓存)"""
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_manager.enabled:
            cached = cache_manager.embedding_cache.get(cache_key)
            if cached is not None:
                return cached

        vec = self._model.embed_query(text)
        if cache_manager.enabled:
            cache_manager.embedding_cache.set(cache_key, vec, ttl=0)
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """★ 批量嵌入 (缓存 + 只对缺失项调 API)"""
        results = [None] * len(texts)
        to_embed_indices = []
        to_embed_texts = []

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_manager.enabled:
                cached = cache_manager.embedding_cache.get(cache_key)
                if cached is not None:
                    results[i] = cached
                    continue
            to_embed_indices.append(i)
            to_embed_texts.append(text)

        if to_embed_texts:
            new_vecs = self._model.embed_documents(to_embed_texts)
            for idx, vec, text in zip(to_embed_indices, new_vecs, to_embed_texts):
                results[idx] = vec
                if cache_manager.enabled:
                    cache_key = hashlib.md5(text.encode()).hexdigest()
                    cache_manager.embedding_cache.set(cache_key, vec, ttl=0)

        cache_hits = len(texts) - len(to_embed_texts)
        if cache_hits > 0:
            logger.info(f"嵌入缓存命中 {cache_hits}/{len(texts)}")

        return results


class MilvusVectorStore:
    """★ 优化版 Milvus 向量存储"""

    def __init__(self):
        self.config = rag_config.milvus
        self.embed_service = EmbeddingService()
        self._collection: Optional[Collection] = None
        self._connect()
        self._try_load_existing_collection()

    def _connect(self):
        """★ 修复: 检查是否已连接, 避免重复连接报错"""
        from pymilvus import connections

        # 检查是否已有活跃连接
        try:
            addr = connections.get_connection_addr("default")
            if addr:
                # 已有连接, 检查是否是同一个地址
                if addr.get("host") == self.config.host and addr.get("port") == str(self.config.port):
                    logger.debug("Milvus 已连接, 复用现有连接")
                    return
                else:
                    # 地址不同, 断开重连
                    connections.disconnect("default")
        except Exception:
            pass

        connections.connect(
            "default",
            host=self.config.host,
            port=self.config.port,
        )
        logger.info(f"Milvus 已连接: {self.config.host}:{self.config.port}")

    def _try_load_existing_collection(self):
        """
        如果集合已存在, 自动加载
        这样即使不调用 create_collection(), search() 也能正常工作
        """
        try:
            name = self.config.collection_name
            if utility.has_collection(name):
                self._collection = Collection(name)
                self._collection.load()
                logger.info(f"自动加载已有集合: {name} ({self._collection.num_entities} 条)")
            else:
                logger.info(f"集合 '{name}' 不存在, 需先调用 create_collection()")
        except Exception as e:
            logger.warning(f"自动加载集合失败: {e}")

    @property
    def collection(self) -> Collection:
        """
        惰性属性, 每次访问时检查
        如果 _collection 为 None, 再尝试加载一次
        """
        if self._collection is None:
            self._try_load_existing_collection()
        if self._collection is None:
            raise RuntimeError(
                f"Milvus 集合 '{self.config.collection_name}' 未初始化. "
                f"请先调用 build_knowledge_base() 或 create_collection()."
            )
        return self._collection

    @property
    def dimension(self):
        cfg = rag_config.embedding
        return cfg.local_dimension if cfg.use_local else cfg.dimension

    @timed("创建Milvus集合")
    def create_collection(self, force_recreate: bool = False):
        name = self.config.collection_name
        if utility.has_collection(name):
            if force_recreate:
                utility.drop_collection(name)
            else:
                self._collection = Collection(name)
                self._collection.load()
                logger.info(f"集合已加载: {name}")
                return

        fields = [
            FieldSchema("id", DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema("content", DataType.VARCHAR, max_length=65535),
            FieldSchema("dense_vector", DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema("source_file", DataType.VARCHAR, max_length=512),
            FieldSchema("chunk_index", DataType.INT64),
            FieldSchema("keywords", DataType.VARCHAR, max_length=1024),
            FieldSchema("metadata_json", DataType.VARCHAR, max_length=4096),
        ]
        schema = CollectionSchema(fields, description="Enterprise RAG")
        self._collection = Collection(name=name, schema=schema)

        # ★ HNSW 索引: 比 IVF_FLAT 查询延迟低 3-5 倍
        self._collection.create_index("dense_vector", {
            "index_type": "HNSW",
            "metric_type": self.config.metric_type,
            "params": {"M": self.config.M, "efConstruction": self.config.efConstruction},
        })
        self._collection.load()
        logger.info(f"集合已创建 (HNSW, M={self.config.M})")

        # ★ 预热: 执行一次空查询, 让 Milvus 加载索引到内存
        if self.config.warmup_on_init:
            self._warmup()

    def _warmup(self):
        """★ 索引预热 — 消除首次查询冷启动延迟"""
        try:
            dummy = [0.0] * self.dimension
            self._collection.search(
                data=[dummy], anns_field="dense_vector",
                param={"metric_type": self.config.metric_type, "params": {"ef": 16}},
                limit=1,
            )
            logger.info("Milvus 索引预热完成")
        except Exception:
            pass

    @timed("向量批量写入")
    def insert_documents(self, documents: List[Document], batch_size: int = 200):
        _ = self.collection  # 触发检查

        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i: i + batch_size]
            texts = [d.page_content for d in batch]
            # ★ 批量嵌入 (带缓存)
            vectors = self.embed_service.embed_batch(texts)

            data = [
                {
                    "id": d.metadata.get("doc_id", f"doc_{i+j}"),
                    "content": d.page_content[:65000],
                    "dense_vector": vectors[j],
                    "source_file": d.metadata.get("source_file", ""),
                    "chunk_index": d.metadata.get("chunk_index", 0),
                    "keywords": ",".join(d.metadata.get("keywords", [])),
                    "metadata_json": json.dumps(
                        {k: str(v) for k, v in d.metadata.items()},
                        ensure_ascii=False
                    )[:4000],
                }
                for j, d in enumerate(batch)
            ]
            self._collection.insert(data)
            logger.info(f"写入 {min(i + batch_size, total)}/{total}")

        self._collection.flush()

    def search(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or self.config.top_k

        # 通过 property 自动检查 collection 是否可用
        try:
            coll = self.collection
        except RuntimeError as e:
            logger.error(f"检索失败: {e}")
            return []

        query_vec = self.embed_service.embed_query(query)

        results = self._collection.search(
            data=[query_vec],
            anns_field="dense_vector",
            param={
                "metric_type": self.config.metric_type,
                "params": {"ef": self.config.ef},   # ★ HNSW 查询参数
            },
            limit=top_k,
            output_fields=["content", "source_file", "keywords", "chunk_index"],
        )

        return [
            {
                "id": hit.id,
                "score": hit.score,
                "content": hit.entity.get("content", ""),
                "source_file": hit.entity.get("source_file", ""),
                "keywords": hit.entity.get("keywords", ""),
                "chunk_index": hit.entity.get("chunk_index", 0),
            }
            for hit in results[0]
        ]

    def search_batch(self, queries: List[str], top_k: int = None) -> List[List[Dict]]:
        """★ 批量查询: 一次性发送多个查询向量给 Milvus"""
        top_k = top_k or self.config.top_k
        vecs = self.embed_service.embed_batch(queries)

        results = self._collection.search(
            data=vecs,
            anns_field="dense_vector",
            param={
                "metric_type": self.config.metric_type,
                "params": {"ef": self.config.ef},
            },
            limit=top_k,
            output_fields=["content", "source_file", "keywords", "chunk_index"],
        )

        all_results = []
        for hits in results:
            all_results.append([
                {
                    "id": hit.id, "score": hit.score,
                    "content": hit.entity.get("content", ""),
                    "source_file": hit.entity.get("source_file", ""),
                    "keywords": hit.entity.get("keywords", ""),
                }
                for hit in hits
            ])
        return all_results

    def is_ready(self) -> bool:
        """检查集合是否可用"""
        try:
            _ = self.collection
            return True
        except RuntimeError:
            return False

    def get_stats(self) -> Dict:
        if self._collection is None:
            return {}
        return {"entities": self._collection.num_entities}