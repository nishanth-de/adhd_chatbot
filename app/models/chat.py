from pydantic import BaseModel, Field, validator
from typing import Optional
import uuid

class ChatRequest(BaseModel):
    question: str = Field(
        ..., # ... means field is required, not default!
        min_length=2,
        max_length=1000,
        description="The user's ADHD-related question"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for conversation continuity. Auto-generated if not provided."
    ) 

    @validator("question")
    def question_must_not_be_blank(cls, v):
        """
        Validates that the 'question' field is not empty or contains only whitespace.

        This is a Pydantic field validator that ensures user input for questions
        meets minimum quality standards.

        Args:
            v (str): The question value to validate.

        Returns:
            str: The validated and stripped question string.

        Raises:
            ValueError: If the question is empty or contains only whitespace.

        Example:
            >>> # Valid input
            >>> msg = ChatMessage(question="  How does AI work?  ")
            >>> msg.question
            'How does AI work?'
            
            >>> # Invalid input
            >>> msg = ChatMessage(question="   ")
            ValueError: Question cannot be blank or white space only

        Why use @field_validator()?
            - Automatic data validation at model instantiation time
            - Ensures data integrity before processing
            - Clean separation of validation logic from business logic
            - Prevents invalid data from entering your application
            - Provides consistent error handling and user feedback
            - Works seamlessly with Pydantic's BaseModel for type safety
        """
        if not v.strip():
            raise ValueError("Question cannot be blank or white space only")
        return v.strip()

class SourceReference(BaseModel):
    source: str = Field(description="Document filename the answer was drawn from")
    similarity: float = Field(description="Semantic simalirty score(0-1) higher is better")

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
    source: str = Field(description="The AI Generated answer")
    session_id: str = Field(description="The session_id for this conversation")
    sources: list[SourceReference] = Field(description="document used to generate the answer")
    status: str = Field(default="success")


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str


class FeedbackRequest(BaseModel):
    session_id: str
    helpful: bool
    comment: Optional[str]

class FeedbackResponse(BaseModel):
    status: str = Field(default="received")
    message: str = Field(default="Thank you for your feedback")
