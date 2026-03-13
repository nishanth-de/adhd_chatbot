# embeddings.py module responsible for all embedding operations.

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

def test_embedding_connection() -> bool:
    """Tests that the embedding model is reachable."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents="connection test",
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS
            )
        )
        return len(response.embeddings[0].values) == EMBEDDING_DIMENSIONS
    
    except Exception as e:
        logger.error(f"Embedding connection test failed: {e}")
        return False


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
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=EMBEDDING_DIMENSIONS
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

def get_query_embedding(text: str) -> list[float]:
    """
    Converts users question into embeddings.
    Use this when searching relevant chunks in the database.

    task_type is RETRIEVAL_QUERY which tells the model that this embedding 
    represent a search query.

    Args:
        text: the user's question.

    Returns:
        list of 768 floats representing the questions meaning.
    """
    try:
        response = client.models.embed_content(
            model = EMBEDDING_MODEL,
            contents = text,
            config = types.EmbedContentConfig(
                task_type = "RETRIEVAL_QUERY",
                output_dimensionality=EMBEDDING_DIMENSIONS
            )     
        )
        embeddings = response.embeddings[0].values
        logger.info(f"Generated query embedding | dimensions={len(embedding)}")
        return list(embeddings)
    except Exception as e:
        logger.error("Query embedding got failed - {e}")
        raise

if __name__ == "__main__":
    import json

    print("===Embedding service test===\n")

    # Test 1: Basic embedding generation
    test_text = "ADHD affects executive function and task initiation"
    embedding = get_embedding(test_text)
    print(f"Test text: '{test_text}'")
    print(f"Embedding dimensions: {len(embedding)}")
    print(f"First 5 values: {[round(vector, 4) for vector in embedding[:5]]}")
    print(f"Value range: min={round(min(embedding), 4)}, max={round(max(embedding), 4)}")

    # Test 2: Query embedding
    query = "What is executive dysfunction?"
    query_embedding = get_query_embedding(query)
    print(f"\nQuery: '{query}'")
    print(f"Query embedding dimensions: {len(query_embedding)}")

    # Test 3: Semantic similarity intuition
    # These two sentences mean similar things — their embeddings should be close
    sentence_a = "ADHD makes it hard to focus"
    sentence_b = "Attention deficit disorder causes concentration difficulties"
    sentence_c = "The stock market closed higher today"

    embedding_a = get_embedding(sentence_a)
    embedding_b = get_embedding(sentence_b)
    embedding_c = get_embedding(sentence_c)

    # Calculate cosine similarity manually (dot product of normalised vectors)
    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a ** 2 for a in v1) ** 0.5
        mag2 = sum(b ** 2 for b in v2) ** 0.5
        return dot / (mag1 * mag2)

    sim_ab = cosine_similarity(embedding_a, embedding_b)
    sim_ac = cosine_similarity(embedding_a, embedding_c)

    print(f"\n=== Similarity Test ===")
    print(f"'{sentence_a}'")
    print(f"vs '{sentence_b}'")
    print(f"Similarity: {round(sim_ab, 4)} (should be HIGH — similar meaning)")

    print(f"\n'{sentence_a}'")
    print(f"vs '{sentence_c}'")
    print(f"Similarity: {round(sim_ac, 4)} (should be LOW — unrelated meaning)")