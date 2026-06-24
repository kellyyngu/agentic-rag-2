import os
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    source: str
    page: Optional[int]
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)


def _hash_chunk(text: str, source: str, index: int) -> str:
    raw = f"{source}:{index}:{text[:100]}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def _split_text(text: str, chunk_size: int = 512, overlap: int = 100) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def process_text(
    text: str,
    source: str,
    page: Optional[int] = None,
    chunk_size: int = 512,
    overlap: int = 100,
    metadata: Optional[Dict[str, Any]] = None,
) -> List[DocumentChunk]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    raw_chunks = _split_text(text, chunk_size, overlap)
    chunks = []
    for i, content in enumerate(raw_chunks):
        chunk_id = _hash_chunk(content, source, i)
        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            source=source,
            page=page,
            chunk_index=i,
            total_chunks=len(raw_chunks),
            metadata=metadata or {},
        ))
    return chunks


def process_pdf(file_path: str, chunk_size: int = 512, overlap: int = 100) -> List[DocumentChunk]:
    try:
        import PyPDF2
    except ImportError:
        logger.error("PyPDF2 not installed")
        return []

    source = Path(file_path).name
    chunks = []

    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            page_chunks = process_text(
                text,
                source=source,
                page=page_num,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata={"file_path": file_path, "page": page_num},
            )
            chunks.extend(page_chunks)

    logger.info(f"[processor] pdf={source} pages={len(reader.pages)} chunks={len(chunks)}")
    return chunks


def process_docx(file_path: str, chunk_size: int = 512, overlap: int = 100) -> List[DocumentChunk]:
    try:
        import docx
    except ImportError:
        logger.error("python-docx not installed")
        return []

    source = Path(file_path).name
    doc = docx.Document(file_path)
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    chunks = process_text(
        full_text,
        source=source,
        chunk_size=chunk_size,
        overlap=overlap,
        metadata={"file_path": file_path},
    )
    logger.info(f"[processor] docx={source} chunks={len(chunks)}")
    return chunks


def process_file(
    file_path: str,
    chunk_size: int = 512,
    overlap: int = 100,
) -> List[DocumentChunk]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return process_pdf(file_path, chunk_size, overlap)
    elif ext in (".docx", ".doc"):
        return process_docx(file_path, chunk_size, overlap)
    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return process_text(
            text,
            source=Path(file_path).name,
            chunk_size=chunk_size,
            overlap=overlap,
            metadata={"file_path": file_path},
        )
    else:
        logger.warning(f"[processor] unsupported file type: {ext}")
        return []
