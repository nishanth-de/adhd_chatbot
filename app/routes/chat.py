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

@router.get(
    "/demo",
    summary="Get demo questions for the ADHD chatbot",
    description="Returns pre-loaded example questions to demonstrate chatbot capabilities.",
    tags=["Demo"]
)
async def get_demo_questions():
    """
    Returns curated example questions that showcase the chatbot's capabilities.
    Use these question strings directly in POST /chat to see the system in action.
    """
    return {
        "instructions": "Copy any question below and paste it into POST /api/v1/chat",
        "demo_questions": [
            {
                "category": "Core ADHD Understanding",
                "question": "What is ADHD and how does it affect the brain?",
                "demonstrates": "Basic psychoeducation with grounded citations"
            },
            {
                "category": "Specific Concept",
                "question": "What is rejection sensitive dysphoria?",
                "demonstrates": "Specific terminology retrieval — hybrid search strength"
            },
            {
                "category": "Practical Guidance",
                "question": "What coping strategies help with ADHD task initiation?",
                "demonstrates": "Actionable information with source verification"
            },
            {
                "category": "Confidence Gate",
                "question": "What is the capital of Tamil Nadu?",
                "demonstrates": "Out-of-scope detection — system says I don't know"
            }
        ],
        "pipeline": {
            "retrieval": "Hybrid vector + full-text search fused with RRF",
            "reranking": "Cohere cross-encoder rerank-english-v3.0",
            "generation": "Gemini 2.0 Flash with healthcare system prompt",
            "citations": "Exact source chunk with excerpt and relevance score"
        }
    }

from app.database import get_connection
from sqlalchemy import text

@router.get(
    "/stats",
    summary="Knowledge base statistics",
    description="Shows what's in the ADHD knowledge base.",
    tags=["Demo"]
)
async def get_stats():
    """
    Returns statistics about the knowledge base powering this chatbot.
    """
    try:
        with get_connection() as conn:
            # Total chunks
            result = conn.execute(text("SELECT COUNT(*) FROM documents;"))
            total_chunks = result.fetchone()[0]

            # Source breakdown
            result = conn.execute(text("""
                SELECT source_file, COUNT(*) as chunks
                FROM documents
                GROUP BY source_file
                ORDER BY chunks DESC;
            """))
            sources = [
                {"file": row[0], "chunks": row[1]}
                for row in result.fetchall()
            ]

            # Total words
            result = conn.execute(text(
                "SELECT SUM(word_count) FROM documents;"
            ))
            total_words = result.fetchone()[0]

        return {
            "knowledge_base": {
                "total_chunks": total_chunks,
                "total_words": int(total_words or 0),
                "source_documents": len(sources),
                "sources": sources
            },
            "search_capabilities": {
                "vector_search": "Gemini text-embedding-004 (768 dimensions)",
                "keyword_search": "PostgreSQL full-text search with ts_rank_cd",
                "fusion": "Reciprocal Rank Fusion (k=60)",
                "reranking": "Cohere rerank-english-v3.0 cross-encoder"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
