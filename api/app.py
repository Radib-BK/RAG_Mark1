"""
FastAPI Application for Multilingual RAG System

This module provides:
- REST API endpoints for question answering
- System health and status endpoints
- Configuration management
- Error handling and logging
- Request validation and response formatting
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from embeddings.embed_model import create_embedder, TextChunkEmbedder
    from retrieval.vector_store import create_vector_store, build_vector_store_from_embeddings
    from generation.rag_chain import create_rag_chain, MultilingualRAGChain
    from ingestion.extract_pdf import extract_pdf_content
    from ingestion.preprocess import preprocess_document
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create placeholder classes and functions for testing
    class MultilingualRAGChain:
        def ask(self, question, max_context_chunks=5, similarity_threshold=0.3, include_sources=True): 
            # Mock RAGResponse structure
            class MockResponse:
                def __init__(self):
                    self.answer = "System not fully initialized - dependencies missing. Please fix PyTorch installation."
                    self.source_chunks = []
                    self.query_language = "en"
                    self.model_used = "none"
                    self.confidence_score = 0.0
            return MockResponse()
        def test_components(self): 
            return {"status": "imports_failed"}
        def clear_memory(self):
            pass
        def get_conversation_history(self):
            return []
    
    class TextChunkEmbedder:
        def __init__(self, embedder): pass
        def embed_chunks(self, chunks, batch_size=16): return []
    
    def create_embedder(model_name=None, cache_dir=None):
        return None
    
    def create_vector_store(embedding_dimension=384, store_dir=None):
        return None
        
    def build_vector_store_from_embeddings(embedding_data, store_dir=None):
        return None
        
    def create_rag_chain(vector_store=None, embedder=None, model_name=None, base_url=None):
        return MultilingualRAGChain()
        
    def extract_pdf_content(pdf_path):
        return {"chunks": [], "metadata": {"error": "PDF extraction not available"}}
        
    def preprocess_document(document_data, chunk_size=512):
        return []

# Global variables for the RAG system
rag_chain: Optional[MultilingualRAGChain] = None
system_status = {
    "initialized": False,
    "last_update": None,
    "vector_store_size": 0,
    "models_loaded": False
}

# Simple conversation counter
conversation_counter = 0

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask", min_length=1, max_length=1000)
    max_chunks: int = Field(default=5, description="Maximum number of context chunks", ge=1, le=20)
    threshold: float = Field(default=0.3, description="Similarity threshold", ge=0.0, le=1.0)
    include_sources: bool = Field(default=True, description="Include source information")

class QuestionResponse(BaseModel):
    answer: str
    query_language: str
    model_used: str
    confidence_score: float
    sources: Optional[List[Dict[str, Any]]] = None
    response_time_ms: int
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    system_info: Dict[str, Any]
    component_health: Dict[str, Any]
    timestamp: str

class SystemStatsResponse(BaseModel):
    vector_store_stats: Dict[str, Any]
    system_status: Dict[str, Any]
    conversation_count: int
    uptime_seconds: float

class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: str

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting RAG system...")
    await initialize_rag_system()
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG system...")
    global rag_chain
    if rag_chain:
        rag_chain.clear_memory()

# Create FastAPI app
app = FastAPI(
    title="Multilingual Educational RAG API",
    description="A multilingual RAG system for HSC educational content in Bangla and English",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG = {
    "pdf_path": os.getenv("PDF_PATH", "./data/HSC26-Bangla1st-Paper.pdf"),
    "vector_store_dir": os.getenv("VECTOR_STORE_DIR", "./vector_store"),
    "cache_dir": os.getenv("CACHE_DIR", "./cache"),
    "model_name": os.getenv("MODEL_NAME", "aya-expanse:8b"),
    "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
    "chunk_size": int(os.getenv("CHUNK_SIZE", "512")),
    "embedding_model": os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
}

async def initialize_rag_system():
    """Initialize the RAG system components"""
    global rag_chain, system_status
    
    try:
        logger.info("Initializing embedder...")
        embedder = create_embedder(
            model_name=CONFIG["embedding_model"],
            cache_dir=f"{CONFIG['cache_dir']}/embeddings"
        )
        
        # Check if imports failed (embedder will be None)
        if embedder is None:
            logger.warning("Dependencies not available - running in limited mode")
            rag_chain = create_rag_chain(None, None, CONFIG["model_name"], CONFIG["ollama_url"])
            system_status.update({
                "initialized": True,
                "last_update": datetime.now().isoformat(),
                "vector_store_size": 0,
                "models_loaded": False,
                "mode": "limited"
            })
            logger.info("RAG system initialized in limited mode")
            return
        
        # Check if vector store exists
        vector_store_path = Path(CONFIG["vector_store_dir"]) / "faiss_index.idx"
        
        if vector_store_path.exists():
            logger.info("Loading existing vector store...")
            vector_store = create_vector_store(
                embedding_dimension=embedder.get_embedding_dimension(),
                store_dir=CONFIG["vector_store_dir"]
            )
            vector_store.load_index(str(vector_store_path))
        else:
            logger.info("Creating new vector store from PDF...")
            
            # Check if PDF exists
            pdf_path = Path(CONFIG["pdf_path"])
            if not pdf_path.exists():
                logger.warning(f"PDF not found at {pdf_path}, creating empty vector store")
                vector_store = create_vector_store(
                    embedding_dimension=embedder.get_embedding_dimension(),
                    store_dir=CONFIG["vector_store_dir"]
                )
            else:
                # Process PDF
                document_data = extract_pdf_content(str(pdf_path))
                chunks = preprocess_document(document_data, chunk_size=CONFIG["chunk_size"])
                
                # Create embeddings
                chunk_embedder = TextChunkEmbedder(embedder)
                embedding_data = chunk_embedder.embed_chunks(chunks, batch_size=16)
                
                # Build vector store
                vector_store = build_vector_store_from_embeddings(
                    embedding_data,
                    store_dir=CONFIG["vector_store_dir"]
                )
                
                # Save index only if vector_store is not None
                if vector_store is not None:
                    vector_store.save_index()
                    logger.info("Vector store created and saved")
        
        # Create RAG chain
        logger.info("Initializing RAG chain...")
        rag_chain = create_rag_chain(
            vector_store=vector_store,
            embedder=embedder,
            model_name=CONFIG["model_name"],
            base_url=CONFIG["ollama_url"]
        )
        
        # Update system status
        vector_size = vector_store.index.ntotal if vector_store and hasattr(vector_store, 'index') else 0
        system_status.update({
            "initialized": True,
            "last_update": datetime.now().isoformat(),
            "vector_store_size": vector_size,
            "models_loaded": True
        })
        
        logger.info("RAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {str(e)}")
        # Create fallback RAG chain
        rag_chain = create_rag_chain(None, None, CONFIG["model_name"], CONFIG["ollama_url"])
        system_status.update({
            "initialized": False,
            "last_update": datetime.now().isoformat(),
            "vector_store_size": 0,
            "models_loaded": False,
            "error": str(e)
        })

def get_rag_chain() -> MultilingualRAGChain:
    """Dependency to get RAG chain instance"""
    if not rag_chain or not system_status["initialized"]:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    return rag_chain

@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multilingual Educational RAG API",
        "version": "1.0.0",
        "description": "A multilingual RAG system for HSC educational content",
        "endpoints": {
            "ask": "/ask - Post questions to the RAG system",
            "health": "/health - Check system health",
            "stats": "/stats - Get system statistics",
            "config": "/config - Get configuration",
            "docs": "/docs - API documentation"
        },
        "status": "ready" if system_status["initialized"] else "initializing"
    }

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    chain: MultilingualRAGChain = Depends(get_rag_chain)
):
    """
    Ask a question to the RAG system
    """
    global conversation_counter
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Increment conversation counter
        conversation_counter += 1
        
        # Add timeout protection
        import asyncio
        
        async def process_question():
            # Ask the RAG chain
            response = chain.ask(
                question=request.question,
                max_context_chunks=request.max_chunks,
                similarity_threshold=request.threshold,
                include_sources=request.include_sources
            )
            return response
        
        # Set timeout to 60 seconds
        try:
            response = await asyncio.wait_for(process_question(), timeout=60.0)
        except asyncio.TimeoutError:
            logger.error(f"Question processing timed out after 60 seconds")
            raise HTTPException(
                status_code=408,
                detail="Question processing timed out. Please try a shorter or simpler question."
            )
        
        # Calculate response time
        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return QuestionResponse(
            answer=response.answer,
            query_language=response.query_language,
            model_used=response.model_used,
            confidence_score=response.confidence_score,
            sources=response.source_chunks if request.include_sources else None,
            response_time_ms=response_time,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like timeout)
        raise
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check(chain: MultilingualRAGChain = Depends(get_rag_chain)):
    """
    Health check endpoint
    """
    try:
        # Test system components
        component_health = chain.test_components()
        
        # Determine overall health
        overall_status = "healthy"
        if not system_status["initialized"]:
            overall_status = "unhealthy"
        elif not any(component_health.get("ollama_connection", {}).values()):
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            system_info={
                "initialized": system_status["initialized"],
                "last_update": system_status["last_update"],
                "vector_store_size": system_status["vector_store_size"]
            },
            component_health=component_health,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            system_info=system_status,
            component_health={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )

@app.get("/stats", response_model=SystemStatsResponse)
async def get_stats(chain: MultilingualRAGChain = Depends(get_rag_chain)):
    """
    Get system statistics
    """
    global conversation_counter
    try:
        # Check if we have a real or placeholder chain
        if hasattr(chain, 'vector_store') and chain.vector_store:
            vector_stats = chain.vector_store.get_statistics()
        else:
            vector_stats = {
                "total_vectors": 0, 
                "total_chunks": 0, 
                "embedding_dimension": 384, 
                "index_size_mb": 0.0,
                "status": "not_available"
            }
        
        # Use simple counter for conversation count
        conversation_count = conversation_counter
        
        # Try to get conversation history if available
        if hasattr(chain, 'get_conversation_history'):
            try:
                conversation_history = chain.get_conversation_history()
                if conversation_history and len(conversation_history) > conversation_count:
                    conversation_count = len(conversation_history)
            except:
                pass  # Use simple counter as fallback
        
        return SystemStatsResponse(
            vector_store_stats=vector_stats,
            system_status=system_status,
            conversation_count=conversation_count,
            uptime_seconds=(datetime.now() - datetime.fromisoformat(
                system_status.get("last_update", datetime.now().isoformat())
            )).total_seconds()
        )
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system stats: {str(e)}"
        )

@app.get("/config")
async def get_config():
    """
    Get current configuration
    """
    return {
        "config": CONFIG,
        "system_status": system_status,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/clear-memory")
async def clear_conversation_memory(chain: MultilingualRAGChain = Depends(get_rag_chain)):
    """
    Clear conversation memory
    """
    global conversation_counter
    try:
        # Reset simple counter
        conversation_counter = 0
        
        # Try to clear chain memory if available
        if hasattr(chain, 'clear_memory'):
            chain.clear_memory()
        
        return {
            "message": "Conversation memory cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing memory: {str(e)}"
        )

@app.get("/conversation-history")
async def get_conversation_history(chain: MultilingualRAGChain = Depends(get_rag_chain)):
    """
    Get conversation history
    """
    try:
        history = chain.get_conversation_history()
        return {
            "history": history,
            "count": len(history),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting conversation history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting conversation history: {str(e)}"
        )

@app.post("/rebuild-index")
async def rebuild_vector_index(background_tasks: BackgroundTasks):
    """
    Rebuild vector index from PDF (background task)
    """
    def rebuild_task():
        try:
            logger.info("Starting vector index rebuild...")
            global rag_chain, system_status
            
            # Re-initialize the system
            asyncio.run(initialize_rag_system())
            
            logger.info("Vector index rebuild completed")
        except Exception as e:
            logger.error(f"Vector index rebuild failed: {str(e)}")
            system_status["error"] = str(e)
    
    background_tasks.add_task(rebuild_task)
    
    return {
        "message": "Vector index rebuild started in background",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=f"HTTP {exc.status_code}",
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )

def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI application
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload
    """
    uvicorn.run(
        "api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the RAG API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--config-file", help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    if args.config_file:
        try:
            import json
            with open(args.config_file, 'r') as f:
                file_config = json.load(f)
            CONFIG.update(file_config)
            logger.info(f"Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
    
    logger.info(f"Starting RAG API server on {args.host}:{args.port}")
    logger.info(f"Configuration: {CONFIG}")
    
    run_api(host=args.host, port=args.port, reload=args.reload) 