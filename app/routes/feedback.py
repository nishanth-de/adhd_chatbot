import logging
from fastapi import APIRouter
from app.models.chat import FeedbackRequest, FeedbackResponse

logger = logging.getLogger(__name__)

router = APIRouter()

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