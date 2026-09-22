"""TDRAG Web API - FastAPI Tabanlı Asenkron Sunucu."""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cli import _embedding_fonksiyonu_olustur
from .config import Config
from .ingestion import ingest_folder
from .providers import create_provider
from .qa_pipeline import answer_question
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Global durum nesneleri
_config: Config | None = None
_vector_store: VectorStore | None = None
_llm: Any = None
_lock = asyncio.Lock()


def get_services() -> tuple[Config, VectorStore, Any]:
    global _config, _vector_store, _llm
    if _config is None or _vector_store is None or _llm is None:
        _config = Config()
        emb_fn = _embedding_fonksiyonu_olustur(_config)
        _vector_store = VectorStore(
            persist_dir=_config.chroma_persist_dir,
            collection_name=_config.collection_name,
            embedding_function=emb_fn,
            similarity_threshold=_config.similarity_threshold,
            dedup_jaccard_threshold=_config.dedup_jaccard_threshold,
        )
        _llm = create_provider(_config)
    return _config, _vector_store, _llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Başlangıçta servisleri hazırla ve klasörü kontrol et
    cfg, store, _ = get_services()
    cfg.articles_folder.mkdir(parents=True, exist_ok=True)
    logger.info("TDRAG Web API başlatıldı. Koleksiyon parça sayısı: %d", store.count())
    yield


app = FastAPI(
    title="TDRAG Web API",
    description="Türk Devletleri Makaleleri RAG Web Arayüzü ve API'si",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Kullanıcının sorusu")
    source_file: str | None = Field(default=None, description="Opsiyonel odak makale adı")


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    attempts: int
    grounded: bool
    duration_s: float


class ArticleItem(BaseModel):
    filename: str
    size_kb: float
    indexed: bool


class StatsResponse(BaseModel):
    collection_name: str
    total_chunks: int
    total_articles: int
    indexed_articles: int
    model: str
    provider: str
    num_ctx: int
    top_k: int
    max_output_tokens: int


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    cfg, store, _ = get_services()
    files = list(cfg.articles_folder.glob("*.pdf")) if cfg.articles_folder.exists() else []
    indexed_files = await asyncio.to_thread(store.get_indexed_files)
    total_chunks = await asyncio.to_thread(store.count)

    return StatsResponse(
        collection_name=cfg.collection_name,
        total_chunks=total_chunks,
        total_articles=len(files),
        indexed_articles=len(indexed_files),
        model=cfg.llm_model,
        provider=cfg.llm_provider,
        num_ctx=cfg.llm_num_ctx,
        top_k=cfg.top_k,
        max_output_tokens=cfg.llm_max_output_tokens,
    )


@app.get("/api/articles", response_model=list[ArticleItem])
async def list_articles():
    cfg, store, _ = get_services()
    if not cfg.articles_folder.exists():
        return []

    indexed_files = await asyncio.to_thread(store.get_indexed_files)
    articles: list[ArticleItem] = []

    for file_path in sorted(cfg.articles_folder.glob("*.pdf")):
        size_kb = round(file_path.stat().st_size / 1024, 1)
        articles.append(
            ArticleItem(
                filename=file_path.name,
                size_kb=size_kb,
                indexed=file_path.name in indexed_files,
            )
        )
    return articles


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Soru boş olamaz.")

    source_file = req.source_file.strip() if req.source_file else None

    cfg, store, llm = get_services()
    async with _lock:
        t0 = time.time()
        ans = await asyncio.to_thread(
            answer_question, question, store, llm, cfg, source_file=source_file
        )
        elapsed = round(time.time() - t0, 2)

    return ChatResponse(
        answer=ans.text,
        sources=ans.sources,
        attempts=ans.attempts,
        grounded=ans.grounded,
        duration_s=elapsed,
    )


@app.post("/api/upload")
async def upload_articles(files: list[UploadFile] = File(...)):
    cfg, store, _ = get_services()
    cfg.articles_folder.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for f in files:
        if not f.filename or not f.filename.lower().endswith(".pdf"):
            continue
        dest_path = cfg.articles_folder / f.filename
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        saved_files.append(f.filename)

    if not saved_files:
        raise HTTPException(status_code=400, detail="Yüklenecek geçerli PDF dosyası bulunamadı.")

    # Otomatik indekslemeyi çalıştır
    async with _lock:
        sonuc = await asyncio.to_thread(ingest_folder, cfg, store)

    return {
        "message": f"{len(saved_files)} dosya yüklendi.",
        "saved_files": saved_files,
        "ingest_result": sonuc,
    }


@app.post("/api/reindex")
async def reindex():
    cfg, store, _ = get_services()
    async with _lock:
        sonuc = await asyncio.to_thread(ingest_folder, cfg, store)
    return {"message": "İndeksleme tamamlandı.", "result": sonuc}


# Statik web arayüzünü bağla
web_dir = Path(__file__).parent / "web"
if web_dir.exists():
    app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")


@app.get("/")
async def serve_index():
    index_file = web_dir / "index.html"
    if not index_file.exists():
        return {"message": "TDRAG API çalışıyor. Arayüz hazırlanıyor..."}
    return FileResponse(index_file)


def run():
    import uvicorn
    uvicorn.run("tdrag.web_api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
