"""
FastAPI REST API for the RAG Pipeline.
Production-ready API with proper error handling.

Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# App Setup
# ============================================

app = FastAPI(
    title="Insurance Policy Advisor API",
    description="RAG-powered Q&A over insurance policy documents",
    version="1.0.0"
)

# Initialize pipeline on startup
pipeline = None


@app.on_event("startup")
async def startup():
    global pipeline
    api_key = os.environ.get("OPENAI_API_KEY")
    pdf_dir = os.environ.get("PDF_DIRECTORY", "./HDFC_")
    
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        return
    
    pipeline = RAGPipeline(
        api_key=api_key,
        pdf_directory=pdf_dir,
        persist_path="./chroma_db"
    )
    pipeline.ingest()
    logger.info("Pipeline ready")


# ============================================
# Request/Response Models
# ============================================

class QueryRequest(BaseModel):
    question: str
    use_cache: bool = True


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cached: bool = False


class HealthResponse(BaseModel):
    status: str
    documents_count: int
    collection_ready: bool


# ============================================
# Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the pipeline is ready."""
    if pipeline is None or not pipeline.is_ingested:
        return HealthResponse(
            status="not_ready",
            documents_count=0,
            collection_ready=False
        )
    return HealthResponse(
        status="ready",
        documents_count=pipeline.vectorstore.collection.count(),
        collection_ready=True
    )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """Ask a question about insurance policies."""
    if pipeline is None or not pipeline.is_ingested:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Check /health endpoint.")
    
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        answer, sources = pipeline.ask(
            question=request.question,
            use_cache=request.use_cache
        )
        return QueryResponse(
            answer=answer,
            sources=sources,
            cached=False  # TODO: return actual cache status
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ingest")
async def reingest():
    """Re-ingest all PDFs (force recreate collections)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")
    
    try:
        pipeline.ingest(force_recreate=True)
        return {
            "status": "success",
            "documents_count": pipeline.vectorstore.collection.count()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
