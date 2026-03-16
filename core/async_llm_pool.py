# core/async_llm_pool.py
"""
★ LLM 并发调用池 — 多个 LLM 请求并行执行
  将原本串行的 4 次查询变换 LLM 调用压缩到 1 个 RTT
"""
import asyncio
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config.settings import rag_config
from core.cache import cache_manager
from core.logger import logger


class LLMPool:
    """LLM 并发调用池"""

    def __init__(self):
        cfg = rag_config.llm
        # 主模型 (复杂任务: 生成答案)
        self.primary_llm = ChatOpenAI(
            model=cfg.model_name,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            request_timeout=cfg.request_timeout,
        )
        # ★ 快速模型 (轻量任务: 查询改写 / 幻觉检测)
        self.fast_llm = ChatOpenAI(
            model=cfg.fast_model,
            temperature=0.0,
            max_tokens=1024,
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            request_timeout=cfg.request_timeout,
        )
        self._executor = ThreadPoolExecutor(max_workers=cfg.max_concurrent_calls)

    def _cache_key(self, prompt_text: str, model: str) -> str:
        return hashlib.md5(f"{model}:{prompt_text}".encode()).hexdigest()

    def call(
        self,
        prompt: ChatPromptTemplate,
        variables: Dict[str, str],
        use_fast: bool = False,
    ) -> str:
        """单次 LLM 调用 (带缓存)"""
        llm = self.fast_llm if use_fast else self.primary_llm
        model_name = rag_config.llm.fast_model if use_fast else rag_config.llm.model_name

        # 构造缓存 key
        prompt_text = prompt.format(**variables)
        cache_key = self._cache_key(prompt_text, model_name)

        # 查缓存
        if cache_manager.enabled:
            cached = cache_manager.llm_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"LLM 缓存命中: {cache_key[:8]}")
                return cached

        chain = prompt | llm
        result = chain.invoke(variables)
        response = result.content.strip()

        # 写缓存
        if cache_manager.enabled:
            cache_manager.llm_cache.set(cache_key, response)

        return response

    def call_concurrent(
        self,
        tasks: List[Dict[str, Any]],
        timeout: float = None,
    ) -> List[Optional[str]]:
        """
        ★ 并发执行多个 LLM 调用
        tasks: [{"prompt": ..., "variables": ..., "use_fast": True/False}, ...]
        返回: 与 tasks 等长的结果列表, 超时/失败的为 None
        """
        # timeout = timeout or (rag_config.llm.request_timeout + 2)
        timeout = (rag_config.llm.request_timeout + 2)
        results = [None] * len(tasks)

        def _execute(index: int, task: Dict) -> tuple:
            try:
                result = self.call(
                    prompt=task["prompt"],
                    variables=task["variables"],
                    use_fast=task.get("use_fast", True),
                )
                return index, result
            except Exception as e:
                logger.warning(f"LLM 并发调用 [{index}] 失败: {e}")
                return index, None

        start = time.perf_counter()
        futures = {
            self._executor.submit(_execute, i, t): i
            for i, t in enumerate(tasks)
        }

        for future in as_completed(futures, timeout=timeout):
            try:
                idx, result = future.result(timeout=2)
                results[idx] = result
            except (TimeoutError, Exception) as e:
                idx = futures[future]
                logger.warning(f"LLM 并发任务 [{idx}] 超时/失败: {e}")

        elapsed = (time.perf_counter() - start) * 1000
        success_count = sum(1 for r in results if r is not None)
        logger.info(
            f"并发 LLM 调用: {len(tasks)} 个任务, "
            f"{success_count} 成功, 总耗时 {elapsed:.0f}ms "
            f"(串行预估: {elapsed * len(tasks) / max(success_count, 1):.0f}ms)"
        )
        return results

    def shutdown(self):
        self._executor.shutdown(wait=False)


# 全局单例
llm_pool = LLMPool()