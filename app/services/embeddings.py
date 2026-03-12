import os
import time
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

logger = logging.getLogger(__name__)

# Initialise the client once at module level
# This is efficient - one client reused across for all calls
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY cannot be found, check .env file!!")

client = genai.Client(api_key=api_key)

# Model configuration - we can change here to switch google/gemini models and it's dimensions!!
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768 # Must match our database vector(768) column

def get_embedding(text: str, retries: int = 3) -> list[float]:
    """
    Convert a document chunks to vector embeddings. We use this function, when 
    ingesting documents and store them in our Database.

    task_type RETREIVEL_DOCUMENT tells the model this embedding is a peice of content 
    that needs to be stored and searched.

    Arguments: 
        text = document chunks to embed,
        retries = number of retry attempts on rate limit errors.

    Returns: list of 768 floats(vectors) representing the text's meaning.
    """
    for attempt in range(retries):
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
                config = types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            embedding = response.embeddings[0].values
            logger.debug(f"Generated document embedding | dimensions = {len(embedding)}")
            return embedding

        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                # Rate limited - wait and retry with exponential backoff
                wait_time = (2 ** attempt) # 2s, 4s
                logger.warning(f"rate limited. waiting{wait_time}s,before retry {attempt + 1}/{retries}")
                time.sleep(wait_time)
            
            else:
                # non rate-limited error
                logger.warning(f"Embedding failed - {e}")
                raise

    raise RuntimeError(f"Failed to generate embedding after {retries} retries")

