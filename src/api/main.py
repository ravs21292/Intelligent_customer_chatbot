from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from src.api.chat_endpoints import router as chat_router
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Customer Chatbot API...")
    yield
    # Shutdown
    logger.info("Shutting down Customer Chatbot API...")


app = FastAPI(
    title="Intelligent Customer Support Chatbot API",
    description="Multi-model strategy chatbot with intent classification and RAG",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router, prefix="/api/v1", tags=["chat"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Intelligent Customer Support Chatbot API",
        "version": "0.1.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


def main():
    """Run the API server."""
    import os
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True
    )


if __name__ == "__main__":
    main()

