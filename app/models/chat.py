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


class SourceCitation(BaseModel):
    """
    A grounded citation linking an answer back to its exact source chunk.
    Users can use this to verify the answer against the original document.
    """
    source_file: str = Field(description="PDF filename the chunk came from")
    chunk_index: int = Field(description="Position of chunk within source document")
    page_number: int = Field(description="Page number in original PDF")
    excerpt: str = Field(description="First 200 characters of the source chunk")
    relevance_score: float = Field(description="Cohere relevance score (0-1)")
    confidence: str = Field(description="high, medium, or low")

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
    answer: str = Field(description="AI-generated answer grounded in sources")
    session_id: str = Field(description="Session ID for this conversation")
    confidence: str = Field(
        description="Overall confidence: high, medium, low, or out_of_scope"
    )
    sources: list[SourceCitation] = Field(
        description="Source chunks used to generate this answer"
    )
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
