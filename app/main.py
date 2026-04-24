import logging
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy import text
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from app.models.chat import HealthResponse
from app.database import get_connection,test_connection
from app.services.embeddings import test_embedding_connection


# Logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

async def keep_db_warm():
    """
    Ping the database for every 4 minutes to prevent neon cold starts.
    Neon free tier will be paused after 5 minutes of inactivity.
    This runs as a background task for the lifetime of the server.
    """
    while True:
        try:
            with get_connection() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("DB Keep-alive ping sent!")
        except Exception as e:
            logger.error(f"DB keep-alive failed {e}")
        await asyncio.sleep(240) # 4 minutes



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs on startup and shutdown"""
    # Start background keep-alive task
    task = asyncio.create_task(keep_db_warm())
    logger.info("Database keep-alive task started")
    yield
    # Shutdown - cancel the background task
    task.cancel
    logger.info("Database keep-alive task stopped")



app = FastAPI(
    title="ADHD Chatbot API",
    description=
    """
    An AI-powered ADHD psychoeducation assistant.
    
    Answers questions about ADHD symptoms, terminology and 
    coping strategies — grounded in verified documents.
    
    Note: This system provides educational information only. 
    Always consult a healthcare professional for personal medical decisions
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)



# CORS middleware — allows browser frontends to call this API
# In production you'd restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://adhd-chatbot-ui.vercel.app",
        "https://adhd-chatbot-production.up.railway.app",
        "http://localhost:3000", # local React development
        "http://127.0.0.1:5500", # local HTML file testing via Live Server
        "http://localhost:5500",
        "null" # allows requests from local HTML files opened directly in browser
    ], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Register routers
app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["Chat"]
)

app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["Chat"]
)


# Global exception handler - catches any unhandled error
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exec: Exception):
    logger.error(f"Unhandles exception: {exec}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occured. Please Try again.",
            "status": "error"
        }
    )



@app.get("/", tags=["System"])
async def root():
    return {
        "message": "ADHD Chatbot API is running",
        "docs": "/docs",
        "health": "/health"
    }



@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Verify API is running and DB, AI model is reachable
    Returns 200 if healthy, 503 if database or AI model is unavailable.
    """
    db_status = test_connection()
    ai_status = test_embedding_connection()

    if not db_status or not ai_status:
        # 503 = Service Unavailable
        # Using JSONResponse directly because we need to set status_code
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "version": "0.1.0",
                "database": "connected" if db_status else "unreachable",
                "ai_service": "connected" if ai_status else "unreachable"
            }
        )

    return JSONResponse(
        status_code= 200,
        content={
            "status":"healthy",
            "version": "0.1.0",
            "database":"connected",
            "ai_service":"connected"
        }
    )