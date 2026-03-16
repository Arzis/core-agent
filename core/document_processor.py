# core/document_processor.py
import os
import hashlib
from typing import List
from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import rag_config
from core.logger import logger, timed

import jieba
import jieba.analyse


class DocumentProcessor:
    def __init__(self):
        cfg = rag_config.chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=cfg.separators,
        )

    @timed("文档加载")
    def load_documents(self, source_path: str) -> List[Document]:
        path = Path(source_path)
        docs = []
        loader_map = {".pdf": PyPDFLoader, ".docx": Docx2txtLoader, ".txt": TextLoader, ".md": TextLoader}

        files = [path] if path.is_file() else list(path.rglob("*"))
        for fp in files:
            cls = loader_map.get(fp.suffix.lower())
            if cls:
                try:
                    # [核心修改]：针对 txt 和 md 文件，强制指定 utf-8 编码读取
                    if fp.suffix.lower() in [".txt", ".md"]:
                        # 如果你在 Windows 自带记事本编辑过，建议用 "utf-8-sig" 来兼容 BOM 头
                        loaded = cls(str(fp), encoding="utf-8").load()
                    else:
                        # PDF 和 Docx 是二进制/压缩包格式，不需要传 encoding
                        loaded = cls(str(fp)).load()

                    for d in loaded:
                        d.metadata["source_file"] = fp.name
                        d.metadata["doc_id"] = hashlib.md5(d.page_content[:200].encode()).hexdigest()[:12]
                    docs.extend(loaded)
                except Exception as e:
                    logger.warning(f"加载失败 {fp}: {e}")

        logger.info(f"加载 {len(docs)} 文档")
        return docs

    @timed("文档处理")
    def process(self, source_path: str, chunk_strategy: str = "recursive") -> List[Document]:
        docs = self.load_documents(source_path)

        # 去重 + 过滤
        seen, cleaned = set(), []
        for d in docs:
            h = hashlib.md5(d.page_content.encode()).hexdigest()
            if h not in seen and len(d.page_content.strip()) > 20:
                seen.add(h)
                cleaned.append(d)

        chunks = self.splitter.split_documents(cleaned)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["keywords"] = jieba.analyse.extract_tags(c.page_content, topK=5)

        logger.info(f"{len(docs)} docs → {len(cleaned)} cleaned → {len(chunks)} chunks")
        return chunks