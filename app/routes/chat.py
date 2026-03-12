import uuid
import logging
from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, SourceReference
from app.models.chat import FeedbackRequest, FeedbackResponse


logger = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="send a question to the adhd chatbot",
    description="Accepts a question about adhd and retuns a grounded answer with source reference"
)
async def chat_endpoint(chat_request: ChatRequest):
    # Generate session_id if not provided
    session_id = chat_request.session_id or str(uuid.uuid4())

    RAG_available = True

    if not RAG_available:
        raise HTTPException(
            status_code= 503,
            detail="RAG pipeline is currently unavailable. Please try again."
        )

    logger.info(f"Chat request received | session={session_id} | question_length={len(chat_request.question)}")

    placeholder_answer = (
        f"You asked: '{chat_request.question}'. "
        f"This is a placeholder response. "
        f"RAG pipeline will be connected in Phase 2."
    )
    
    placeholder_sources = [
        SourceReference(source="placeholder_document.txt", similarity=0.0)
    ]
    # --- END PLACEHOLDER ---

    logger.info(f"Chat response sent | session={session_id}")

    return ChatResponse(
        answer=placeholder_answer,
        session_id=session_id,
        sources=placeholder_sources,
        status="success"
    )



@router.post(
        "/feedback",
        response_model=FeedbackResponse,
        summary= "Send a feedback to the ADHD chatbot",
        description="Accepts feedback tied to a session — whether the answer was helpful and an optional comment"
)
async def chat_feedback(feedback_request: FeedbackRequest):
    session_id = feedback_request.session_id

    logger.info(f"Feedback request received session={session_id} | was it helpful? = {feedback_request.helpful}")

    return FeedbackResponse(
        status="Received", 
        message="Thank you for your feedback"
    )