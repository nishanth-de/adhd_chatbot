from pydantic import BaseModel, Field, field_validator
from typing import Optional
import uuid

class ChatRequest(BaseModel):
    question: str = Field( 
        ..., # ... means "ellipsis" and this field is required, not default!
        min_length=2, # must be atleast 2 characters.
        max_length=1000, # must be atomost 10 characters.
        description="The user's ADHD-related question"
    )
    session_id: Optional[str] = Field(
        default=None, # optional default to None, if not provided
        description="Session ID for conversation continuity. Auto-generated if not provided."
    ) 

    @field_validator("question")
    @classmethod # classmethods takes cls because they operaton on cls not self(object/instance of a class).
    def question_must_not_be_blank(cls, v): # here cls = ChatRequest
        if not v.strip():
            raise ValueError("Question cannot be blank or white space only")
        return v.strip()

class SourceReference(BaseModel):
    source: str = Field(description="Document filename the answer was drawn from")
    similarity: float = Field(description="Semantic similarity score(0-1) higher is better")

class ChatResponse(BaseModel):
    """
    ChatResponse model for API responses.

    Represents the structured response returned from the chat endpoint, containing
    the AI-generated answer along with metadata about the conversation session and
    source documents used to generate the response.

    Attributes:
        source (str): The AI-generated answer/response text.
        session_id (str): Unique identifier for the conversation session, allowing
            clients to track and manage conversation history.
        sources (list[SourceReference]): List of document references and citations
            used by the AI to generate the answer. Each item is a SourceReference
            object containing structured metadata about the source document.
        status (str): HTTP status indicator for the response (default: "success").
            Indicates whether the request was processed successfully.
    """
    answer: str = Field(description="The AI Generated answer")
    session_id: str = Field(description="The session_id for this conversation")
    sources: list[SourceReference] = Field(description="document used to generate the answer")
    status: str = Field(default="success")


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    ai_service: str = Field(default="unknown")


class FeedbackRequest(BaseModel):
    session_id: str = Field(...)
    helpful: bool = Field(...)
    comment: Optional[str] = Field(default=None)

class FeedbackResponse(BaseModel):
    status: str = Field(default="Received")
    message: str = Field(default="Thank you for your feedback")
