import os
import logging
from dotenv import load_dotenv
from google import genai
from google.genai import types
from app.services.citations import format_context_for_llm

load_dotenv()

logger = logging.getLogger(__name__)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

client = genai.Client(api_key=api_key)

CHAT_MODEL = "gemini-3.1-flash-lite-preview"

def test_llm_collection() -> bool:
    """
    Quick connectivity test for the Gemini API.
    only used by health checks.
    Returns True if API is reachable, else False
    """
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents="health check ping",
            config=types.EmbedContentConfig(
                task_type= "RETRIEVAL_QUERY",
                output_dimensionality=768
            )
        )
        return len(response.embeddings[0].values) == 768
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False

# Prompt engineering:
# System prompt — this is the personality and rules of our chatbot
SYSTEM_PROMPT = """

You are a knowledgeable, warm, and empathetic ADHD information assistant.

Your purpose is to help people understand ADHD — whether they are:
- Newly diagnosed and trying to make sense of their condition
- A parent or partner supporting someone with ADHD
- Someone curious about a specific symptom or term they encountered

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY using information provided in the context below. Do not use outside knowledge.
2. If the provided context does not contain enough information to answer, say clearly:
    "I don't have enough information about that in my knowledge base. 
    I'd recommend consulting a healthcare professional or visiting CHADD.org."
3. NEVER suggest specific medications, dosages, or treatment plans.
4. NEVER provide a diagnosis or suggest someone has or doesn't have ADHD.
5. ALWAYS recommend consulting a qualified healthcare professional for personal medical decisions.
6. Be warm and clear. Avoid jargon unless you immediately explain it.
7. Keep answers focused and readable — use short paragraphs, not walls of text.

Remember: you are an educational resource, not a medical professional.
"""

def generate_answer(question:str, context_chunks: list[dict]) -> str:
    """
    Generates a grounded answer using gemini, based only on the retrieval context.

    Argumentss:
        question: The user's question.
        context_chunks: A list of dictionary with 'content' and 'source' keys
                        retrieved from PgVector.

    Returns: string answer from gemini.

    """
    if not context_chunks:
        return("""
        I was not able to find the relevant information in my knowledge base. To answer your question please consult a healthcare professional or visit chadd.org for reliable ADHD information.
        """)
    
    # Using citation formatter for consistent context structure
    context_text = format_context_for_llm(context_chunks)

    # full prompt: context + question
    # The model will only answer based on the provided context
    user_prompt = f"""
    Use the following verified information to answer the question.
    Only use what is provided below, Do not add information from outside this context

    ===CONTEXT===
    {context_text}

    ===QUESTION===
    {question}

    Provide a clear, helpful answer based strictly on the context above.
    """

    try:
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=user_prompt,
            config= types.GenerateContentConfig(
                system_instruction = SYSTEM_PROMPT,
                temperature = 0.1,
                max_output_tokens = 800
            )
        )
        answer = response.text
        logger.info(f"Answer generated | Generated answer length = {len(answer)} in characters")
        return answer
    
    except Exception as e:
        logger.error(f"Answer generation failed - {e}")
        raise

if __name__ == "__main__":
    print("===LLM SERVICE TEST===\n")

        # Simulate what retrieval will provide — fake chunks for now
    fake_chunks = [
        {
            "source": "adhd_basics.txt",
            "content": """
                        ADHD (Attention Deficit Hyperactivity Disorder) is a neurodevelopmental 
                        disorder characterised by persistent patterns of inattention, hyperactivity, and 
                        impulsivity that interfere with functioning and development. ADHD affects approximately 
                        5-7% of children and 2-5% of adults worldwide. It is not caused by bad parenting, 
                        too much screen time, or lack of discipline — it has strong genetic and neurological roots.
                    """
        },
        {
            "source": "executive_function.txt",
            "content": """
                    Executive dysfunction is one of the core challenges of ADHD. Executive 
                    functions are the mental processes that help us plan, focus, remember instructions, 
                    and manage multiple tasks. People with ADHD often struggle with task initiation 
                    (getting started), working memory (holding information in mind), cognitive flexibility 
                    (switching between tasks), and emotional regulation.
                    """
        }
    ]

    question = "What is ADHD and why do people with it struggle to start tasks?"
    print(f"Question: {question}\n")
    print("Generating answer from context...\n")

    answer = generate_answer(question=question, context_chunks=fake_chunks)
    print(f"Answer:\n{answer}")
    print(f"\nAnswer length: {len(answer)} characters")

    # Test Gaurdrails - We ask for medications and it should not generate answer!
    gaurdrail_question = "What medication is should take for ADHD?"
    gaurdrail_answer = generate_answer(question=gaurdrail_question, context_chunks=fake_chunks)
    print("===GAURDRAIL ANSWER===\n")
    print(f"QUESTION: {gaurdrail_question}")
    print(f"ANSWER: {gaurdrail_answer}")

    # Test with empty context
    empty_answer = generate_answer("What is ADHD?", [])
    print(f"\n=== EMPTY CONTEXT TEST ===")
    print(f"A: {empty_answer}")


