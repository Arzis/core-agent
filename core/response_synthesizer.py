# core/response_synthesizer.py
"""
★ 答案合成 — 合并上下文压缩和生成为一步, 省掉一次 LLM 调用
"""
import json
from typing import List, Dict, Any, Optional

from langchain_core.prompts import ChatPromptTemplate

from core.async_llm_pool import llm_pool
from core.logger import timed


# ★ 合并 Prompt: 压缩 + 生成一步到位
GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是专业的企业知识助手。基于参考资料回答问题。

严格规则：
1. **只基于参考资料回答**, 不使用自身知识
2. 在关键信息后标注 [来源X]
3. 资料不足时说明"根据现有资料无法完整回答"
4. 先提取关键信息, 再组织答案, 结构清晰
5. 用中文回答

参考资料:
{context}"""),
    ("human", "{query}"),
])

# 幻觉检测 Prompt
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """检查答案是否完全基于参考资料。
输出格式 (严格 JSON):
{{"has_hallucination": true/false, "confidence": 0.0-1.0, "issues": ["问题1"]}}"""),
    ("human", "资料:\n{context}\n\n答案:\n{answer}\n\n核查结果:"),
])

# 修正 Prompt
REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """之前的答案可能有不准确之处。请严格基于参考资料重新回答, 不确定的内容标注"资料未提及"。

参考资料:
{context}"""),
    ("human", "{query}"),
])


class ResponseSynthesizer:

    @timed("答案生成(合并)")
    def generate(self, query: str, documents: List[Dict]) -> Dict[str, Any]:
        """
        ★ 优化: 压缩 + 生成合并为一次 LLM 调用
        原始: 压缩(~900ms) + 生成(~1500ms) = ~2400ms
        优化: 合并生成(~1500ms) = ~1500ms, 省 900ms
        """
        context = "\n\n".join(
            f"[来源{i+1}] ({d.get('source_file', 'unknown')})\n{d['content']}"
            for i, d in enumerate(documents)
        )

        answer = llm_pool.call(
            GENERATE_PROMPT,
            {"query": query, "context": context},
            use_fast=False,  # 生成用主模型
        )

        return {
            "answer": answer or "抱歉，无法生成答案。",
            "context": context,
            "sources": [
                {
                    "index": i + 1,
                    "file": d.get("source_file", "unknown"),
                    "preview": d["content"][:100],
                    "score": d.get("rerank_score", d.get("score", 0)),
                }
                for i, d in enumerate(documents)
            ],
        }

    @timed("幻觉检测")
    def check_hallucination(self, answer: str, context: str) -> Dict:
        """★ 用快速模型做幻觉检测"""
        result = llm_pool.call(
            HALLUCINATION_PROMPT,
            {"answer": answer, "context": context[:3000]},
            use_fast=True,  # ★ 快速模型
        )
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {"has_hallucination": False, "confidence": 0.5, "issues": []}

    @timed("答案修正")
    def refine(self, query: str, context: str) -> str:
        result = llm_pool.call(
            REFINE_PROMPT,
            {"query": query, "context": context[:3000]},
            use_fast=False,
        )
        return result or ""