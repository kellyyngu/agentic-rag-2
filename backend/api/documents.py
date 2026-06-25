import asyncio
import os
import tempfile
from typing import List
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from pydantic import BaseModel
from loguru import logger

from config import settings
from retrieval.document_processor import process_file, process_text, DocumentChunk

router = APIRouter()


class IngestTextRequest(BaseModel):
    text: str
    source: str = "manual_input"


class IngestResponse(BaseModel):
    chunks_added: int
    source: str


@router.post("/documents/upload", response_model=IngestResponse)
async def upload_document(request: Request, file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md", ".docx"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed}")

    vector_store = request.app.state.vector_store
    bm25_index = request.app.state.bm25_index

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        chunks: List[DocumentChunk] = process_file(
            tmp_path,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        # Override source name with original filename
        for c in chunks:
            c.source = file.filename or c.source

        if not chunks:
            raise HTTPException(422, "No text could be extracted from the file")

        await vector_store.upsert(chunks)
        await asyncio.to_thread(bm25_index.add, chunks)

        logger.info(f"[documents] uploaded {file.filename}: {len(chunks)} chunks")
        return IngestResponse(chunks_added=len(chunks), source=file.filename or "")
    finally:
        os.unlink(tmp_path)


@router.post("/documents/text", response_model=IngestResponse)
async def ingest_text(request: Request, body: IngestTextRequest):
    vector_store = request.app.state.vector_store
    bm25_index = request.app.state.bm25_index

    chunks = process_text(
        body.text,
        source=body.source,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )
    if not chunks:
        raise HTTPException(422, "Text is empty or too short")

    await vector_store.upsert(chunks)
    await asyncio.to_thread(bm25_index.add, chunks)

    logger.info(f"[documents] ingested text '{body.source}': {len(chunks)} chunks")
    return IngestResponse(chunks_added=len(chunks), source=body.source)


@router.get("/documents/list")
async def list_documents(request: Request):
    vector_store = request.app.state.vector_store
    sources = await vector_store.list_sources()
    return {"documents": sources}


@router.get("/documents/stats")
async def document_stats(request: Request):
    vector_store = request.app.state.vector_store
    bm25_index = request.app.state.bm25_index
    return {
        "vector_count": await vector_store.count(),
        "bm25_count": bm25_index.doc_count,
    }


@router.delete("/documents")
async def clear_documents(request: Request):
    vector_store = request.app.state.vector_store
    bm25_index = request.app.state.bm25_index
    await vector_store.delete_collection()
    bm25_index.clear()
    return {"status": "cleared"}
