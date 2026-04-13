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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Insurance Policy Advisor</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; min-height: 100vh; display: flex; justify-content: center; align-items: center; }
            .container { background: white; border-radius: 16px; padding: 40px; max-width: 700px; width: 90%; box-shadow: 0 4px 24px rgba(0,0,0,0.1); }
            h1 { color: #1a1a2e; margin-bottom: 8px; font-size: 28px; }
            .subtitle { color: #6c757d; margin-bottom: 24px; font-size: 14px; }
            textarea { width: 100%; padding: 14px; border: 2px solid #e0e0e0; border-radius: 10px; font-size: 16px; resize: none; outline: none; transition: border 0.3s; }
            textarea:focus { border-color: #2d6a4f; }
            button { background: #2d6a4f; color: white; border: none; padding: 12px 28px; border-radius: 10px; font-size: 16px; cursor: pointer; margin-top: 12px; width: 100%; }
            button:hover { background: #1b4332; }
            button:disabled { background: #adb5bd; cursor: not-allowed; }
            .answer-box { background: #f8f9fa; border-left: 4px solid #2d6a4f; padding: 20px; border-radius: 0 10px 10px 0; margin-top: 20px; line-height: 1.8; white-space: pre-wrap; }
            .sources { margin-top: 12px; }
            .source-badge { display: inline-block; background: #2d6a4f; color: white; padding: 4px 12px; border-radius: 20px; font-size: 12px; margin: 4px; }
            .loading { display: none; text-align: center; margin-top: 20px; color: #6c757d; }
            .examples { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
            .example { background: #e9ecef; border: none; padding: 8px 14px; border-radius: 20px; font-size: 13px; cursor: pointer; color: #495057; }
            .example:hover { background: #dee2e6; }
            .stats { text-align: center; color: #adb5bd; font-size: 12px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🛡️ Insurance Policy Advisor</h1>
            <p class="subtitle">Ask any question about your HDFC Life insurance policies. Powered by RAG.</p>
            
            <textarea id="question" rows="3" placeholder="Type your question here..."></textarea>
            
            <div class="examples">
                <button class="example" onclick="askExample(this)">How much is covered for hospitalization?</button>
                <button class="example" onclick="askExample(this)">What is the waiting period?</button>
                <button class="example" onclick="askExample(this)">What surgeries are covered?</button>
                <button class="example" onclick="askExample(this)">Can I surrender my policy?</button>
            </div>
            
            <button id="askBtn" onclick="askQuestion()">Ask Question</button>
            
            <div class="loading" id="loading">⏳ Searching policies and generating answer...</div>
            
            <div id="result"></div>
            
            <div class="stats">
                RAG Pipeline: PDF Extraction → Chunking → OpenAI Embeddings → ChromaDB → LLM Reranking → GPT-4o-mini
            </div>
        </div>
        
        <script>
            function askExample(btn) {
                document.getElementById('question').value = btn.textContent;
                askQuestion();
            }
            
            async function askQuestion() {
                const question = document.getElementById('question').value.trim();
                if (!question) return;
                
                const btn = document.getElementById('askBtn');
                const loading = document.getElementById('loading');
                const result = document.getElementById('result');
                
                btn.disabled = true;
                btn.textContent = 'Thinking...';
                loading.style.display = 'block';
                result.innerHTML = '';
                
                try {
                    const response = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question, use_cache: true })
                    });
                    
                    const data = await response.json();
                    
                    let html = '<div class="answer-box">' + data.answer + '</div>';
                    if (data.sources && data.sources.length > 0) {
                        html += '<div class="sources">';
                        data.sources.forEach(s => {
                            html += '<span class="source-badge">📄 ' + s + '</span>';
                        });
                        html += '</div>';
                    }
                    result.innerHTML = html;
                } catch (error) {
                    result.innerHTML = '<div class="answer-box" style="border-color: red;">Error: ' + error.message + '</div>';
                }
                
                btn.disabled = false;
                btn.textContent = 'Ask Question';
                loading.style.display = 'none';
            }
            
            document.getElementById('question').addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
