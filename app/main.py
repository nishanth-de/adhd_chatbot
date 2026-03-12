import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat
from app.routes import feedback
from app.models.chat import HealthResponse
from app.database import test_connection



# Logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)



app = FastAPI(
    title="ADHD Chatbot API",
    description=
    """
    An AI-powered ADHD psychoeducation assistant.
    
    Answers questions about ADHD symptoms, terminology, diagnosis, 
    and coping strategies — grounded in verified documents.
    
    Note: This system provides educational information only. 
    Always consult a healthcare professional for personal medical decisions
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)



# CORS middleware — allows browser frontends to call this API
# In production you'd restrict this to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    feedback.router,
    prefix="/api/v1",
    tags=["Feedback"]
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
    Verify API is running and DB is reachable
    Returns 200 if healthy, 503 if database is unavailable.
    """
    db_status = test_connection()

    if not db_status:
        # 503 = Service Unavailable
        # Using JSONResponse directly because we need to set status_code
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "version": "0.1.0",
                "database": "unreachable"
            }
        )

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        database="connected"
    )