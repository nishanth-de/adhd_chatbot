import uuid
import logging
from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, SourceReference

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="send a question to the adhd chatbot",
    description="Accepts a question about adhd and retuns a grounded answer with source reference"
)
async def chat_endpoint(request: ChatRequest):
    # Generate session_id if not provided

    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"Chat request received | session={session_id} | question_length={len(request.question)}")

    placeholder_answer = (
        f"You asked: '{request.question}'. "
        f"This is a placeholder response. "
        f"RAG pipeline will be connected in Phase 2."
    )
    
    placeholder_sources = [
        SourceReference(source="placeholder.txt", similarity=0.0)
    ]
    # --- END PLACEHOLDER ---

    logger.info(f"Chat response sent | session={session_id}")

    return ChatResponse(
        answer=placeholder_answer,
        session_id=session_id,
        sources=placeholder_sources,
        status="success"
    )

