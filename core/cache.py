# core/cache.py
"""
★ 多级缓存系统 — 大幅减少重复计算
- L1: 内存 LRU 缓存 (嵌入、LLM 响应)
- L2: 语义缓存 (相似查询命中)
- 自动过期 + 手动失效
"""
import time
import hashlib
import threading
from typing import Any, Optional, Dict, Tuple
from collections import OrderedDict
from dataclasses import dataclass

from config.settings import rag_config
from core.logger import logger


@dataclass
class CacheEntry:
    value: Any
    created_at: float
    ttl: int  # 秒, 0 表示永不过期
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl == 0:
            return False
        return (time.time() - self.created_at) > self.ttl


class LRUCache:
    """线程安全的 LRU 缓存"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.Lock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _make_key(self, *args, **kwargs) -> str:
        raw = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired:
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            # 移到末尾 (最近使用)
            self._cache.move_to_end(key)
            entry.hits += 1
            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int = None):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl if ttl is not None else self._default_ttl,
            )
            # 淘汰
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1

    def invalidate(self, key: str):
        with self._lock:
            self._cache.pop(key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._stats["hits"] + self._stats["misses"]
        return self._stats["hits"] / total if total > 0 else 0.0

    @property
    def stats(self) -> Dict:
        return {**self._stats, "size": len(self._cache), "hit_rate": f"{self.hit_rate:.1%}"}


class SemanticCache:
    """
    ★ 语义缓存 — 相似查询直接命中
    原理: 对查询做嵌入, 在缓存中找相似度 > 阈值的条目
    """

    def __init__(self, similarity_threshold: float = 0.92, max_size: int = 500):
        self._entries: list = []       # [(embedding, key, value, timestamp)]
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._lock = threading.Lock()

    def _cosine_sim(self, a: list, b: list) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot / (norm_a * norm_b + 1e-8)

    def get(self, query_embedding: list) -> Optional[Any]:
        with self._lock:
            best_score = 0.0
            best_value = None
            for emb, key, value, ts in self._entries:
                sim = self._cosine_sim(query_embedding, emb)
                if sim > best_score:
                    best_score = sim
                    best_value = value
            if best_score >= self._threshold:
                logger.info(f"语义缓存命中, 相似度: {best_score:.3f}")
                return best_value
            return None

    def set(self, query_embedding: list, key: str, value: Any):
        with self._lock:
            self._entries.append((query_embedding, key, value, time.time()))
            if len(self._entries) > self._max_size:
                self._entries = self._entries[-self._max_size:]


class CacheManager:
    """统一缓存管理器"""

    def __init__(self):
        cfg = rag_config.cache
        self.enabled = cfg.enable

        self.embedding_cache = LRUCache(
            max_size=cfg.embedding_cache_size, default_ttl=0  # 嵌入永不过期
        )
        self.query_cache = LRUCache(
            max_size=cfg.query_cache_size, default_ttl=cfg.query_cache_ttl
        )
        self.llm_cache = LRUCache(
            max_size=cfg.llm_cache_size, default_ttl=cfg.llm_cache_ttl
        )
        self.semantic_cache = SemanticCache()

    def print_stats(self):
        logger.info(
            f"📦 缓存统计 — "
            f"Embedding: {self.embedding_cache.stats} | "
            f"Query: {self.query_cache.stats} | "
            f"LLM: {self.llm_cache.stats}"
        )


# 全局单例
cache_manager = CacheManager()