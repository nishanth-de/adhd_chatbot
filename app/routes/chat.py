import uuid
import logging
from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest, ChatResponse, SourceCitation
from app.models.chat import FeedbackRequest, FeedbackResponse
from app.services.rag_pipeline import run_rag_pipeline


logger = logging.getLogger(__name__)


router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="send a question to the adhd chatbot",
    description="""
    Accepts a question about adhd and retuns a grounded answer with source reference

    The response includes:
    > answer: grounded in verified ADHD documents
    > confidence: high/medium/low/out_of_scope
    > sources: exact source chunks with excerpts for verification

    This chatbot provides educational information only.
    Always consult a healthcare professional for personal medical decisions.
    """
)
async def chat_endpoint(request: ChatRequest):
    # Generate session_id if not provided
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(
        f"Chat request | session={session_id} | "
        f"question='{request.question[:60]}'"
        )

    try:
        result = run_rag_pipeline(request.question)

        sources = [
            SourceCitation(**src)
            for src in result["sources"]
        ]

        return ChatResponse(
            answer=result["answer"],
            session_id=session_id,
            confidence=result["confidence"],
            sources=sources,
            status="success"
        )

    except Exception as e:
        logger.error(f"RAG pipeline error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred processing your question. Please try again."
        )
    
@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback on a chatbot response"
)
async def chat_feedback(feedback_request: FeedbackRequest):
    session_id = feedback_request.session_id
    comment_length = len(feedback_request.comment) if feedback_request.comment else 0

    logger.info(
        f"Feedback | session={session_id} | "
        f"helpful={feedback_request.helpful} | "
        f"comment_length={comment_length}"
    )

    return FeedbackResponse(
        status="received",
        message="Thank you for your feedback"
    )