import re
import logging
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import functools # For simple in-memory response caching

load_dotenv()

logger = logging.getLogger(__name__)

# CRISIS DETECTION - REGEX ONLY, ALWAYS FIRST
# Never Replaced by a LLM, Must be determenistic!!
CRISIS_PATTERNS = [
    r'\b(suicid|kill\s+myself|end\s+my\s+life|want\s+to\s+die)\b',
    r'\bself.harm|hurt\s+myself',
    r'\b(hurt\s+myself|cutting\s+myself|overdos)\b',
    r'\b(no\s+reason\s+to\s+live|can\'t\s+go\s+on)\b',
]

CRISIS_RESPONSE = """I hear that you're going through something really difficult right now.

Please reach out for immediate support:

**iCall (India):** 9152987821
**Vandrevala Foundation:** 1860-2662-345 (24/7)
**International Association for Suicide Prevention:** https://www.iasp.info/resources/Crisis_Centres/

You don't have to face this alone. A trained counsellor can help. 

This chatbot provides educational information about ADHD and is not equipped
to provide crisis support. Please contact one of the resources above."""

def is_crisis_message(text: str) -> bool:
    """
    Regex based crisis detection, detemenistic and never skipped.
    """
    text_lower = text.lower()
    for pattern in CRISIS_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning("Crisis pattern detected")
            return True

    return False

# Fast Regex Pre-Filter 
# To catch obvious pattern to avoid unnecessary LLM calls!!

OBVIOUS_GREETING_PATTERNS = [
    # Greetings with optional punctuation and extra words
    r'^(hi|hello|hey)[\s!?,.]',        # "Hello!", "Hey there"
    r'^(hi|hello|hey)$',               # "hi" alone
    r'^good\s*(morning|afternoon|evening|day|night)[\s!?,.]?$',
    r'^how\s+are\s+you',
    r'^what\'?s\s+up',
    r'^(howdy|greetings|sup|hiya)',

    # Indian greetings
    r'^(namaste|vanakkam|namaskar)',

    # Thanks
    r'^(thanks|thank\s+you|thank\s+u|thx|ty)[\s!.,]?$',
    r'^(many\s+thanks|much\s+appreciated)',

    # Farewells
    r'^(bye|goodbye|good\s*bye|see\s+you|take\s+care|cya)[\s!.,]?$',
    r'^(have\s+a\s+good|have\s+a\s+great)',

    # Acknowledgements
    r'^(ok|okay|got\s+it|understood|alright|sure|makes\s+sense)[\s!.,]?$',
    r'^(interesting|cool|great|nice|wow)[\s!.,]?$',
    r'^(that\'?s?\s+(helpful|great|interesting|clear|good))',
    r'^(tell\s+me\s+more|can\s+you\s+explain\s+more)',

    # Simple yes/no
    r'^(yes|no|yeah|nope|yep|nah|yup)[\s!.,]?$',
]

# Fix: case-insensitive name extraction
INTRODUCTION_PATTERN = r"i'?m\s+([a-zA-Z]+)|my\s+name\s+is\s+([a-zA-Z]+)"

def is_obviously_conversational(text: str) -> bool:
    """
    Fast Regex check for clearly conversational messages!
    only catch unambiguous changes!
    """
    text_stripped = text.strip().lower()
    for pattern in OBVIOUS_GREETING_PATTERNS:
        if re.search(pattern, text_stripped):
            return True
    
    if re.search(INTRODUCTION_PATTERN, text, re.IGNORECASE):
        return True
    
    return False



# LLM Intent Classifier.
# Only runs when Regex doesn't catch the message. 

CLASSIFICATION_MODEL = "gemini-2.5-flash-lite"

gemini_client = None

def get_gemini_client():
    """
    Lazy initialization - only creates client when needed!!
    """
    global gemini_client
    if gemini_client is None:
        gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    return gemini_client



def classify_intent(text: str) -> str:
    """
    Using LLM to classify message intent for ambiguous cases!!

    Categories:
    > "conversational": greetings, small talk, introductions,
                        thanks, farewells, acknowledgements
    > "adhd_question":  anything about ADHD, mental health,
                        neurodivergence, symptoms, coping
    > "out_of_scope":   unrelated topics, personal diagnosis requests,
                        medication dosages, other medical conditions
    
    Returns one of the 3 category strings.

    Falls back to "adhd_question" on any error - better to run the pipeline 
    unnecssarily than to block a real question.
    """
    
    prompt = f"""
You are a classifier for an ADHD psychoeducation chatbot. 
Your job is to classify the USER'S INTENT, not the topic.

Categories:
- "conversational": greetings, introductions, thanks, farewells, 
small talk, acknowledgements ("ok", "got it", "interesting", 
"tell me more", "makes sense")
- "out_of_scope": requests for personal diagnosis ("do I have ADHD?", 
"am I ADHD?"), specific medication doses or names (Ritalin, Adderall, 
Concerta, dosage, mg), completely unrelated topics
- "adhd_question": genuine questions ABOUT ADHD as a topic — 
symptoms, causes, coping strategies, executive function, research

IMPORTANT: "Do I have ADHD?" is OUT_OF_SCOPE because the user wants 
a diagnosis, not education. "What is ADHD?" is adhd_question because 
they want to learn.

Message: "{text}"

Reply with ONLY one word: conversational, out_of_scope, or adhd_question"""

    
    try:
        client = get_gemini_client()
        response = client.models.generate_content(
            model=CLASSIFICATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0, # Deterministic output
                max_output_tokens = 10 # Category words only
            )
        )


        # Fixed — strip all punctuation and whitespace
        first_word = response.text.strip().lower().split()[0]
        result = re.sub(r'[^\w_]', '', first_word)

        # Validate response in one of our expected categories
        validated_categories = {"conversational", "adhd_question", "out_of_scope"}
        if result not in validated_categories:
            logger.warning(f"Unexpected classification - {result} defaulting to adhd_question")
            return "adhd_question"
        
        logger.info(f"LLM Classified: '{text[:50]}' -> 'result' ")
        return result

    except Exception as e:
        # On any failure, allow through to RAG pipeline.
        # Why? Better to answer a conversational message through RAG, 
        # rather than to block a real adhd_question!
        logger.error(f"Classification failed: {e} - defaulting to adhd_question")
        return "adhd_question"



@functools.lru_cache(maxsize=256)
def classify_intent_cached(text: str) -> str:
    """
    Cached version of classify intent
    Same Input = Same output without an API Call
    LRU Cache holds last 256 unique classification
    """
    return classify_intent(text)


# Conversation Builder
def extract_name(text: str) -> str | None:
    match = re.search(INTRODUCTION_PATTERN, text, re.IGNORECASE)
    if match:
        return match.group(1) or match.group(2)
    return None

def build_conversational_response(text: str) -> str:
    name = extract_name(text)
    text_lower = text.lower().strip()

    if any(word in text_lower for word in["bye", "goodbye", "see you"]):
        return "Take care! Feel free to come back whenever you have questions about ADHD. 👋"
    
    if any(word in text_lower for word in["thank", "thanks", "thx"]):
        return "You're welcome! Let me know if you have any other questions about ADHD."
    
    if name:
        return(
            f"Hello {name}! I'm your ADHD psychoeducation assistant. "
            f"I can help you understand ADHD symptoms, executive dysfunction, "
            f"coping strategies, emotional dysregulation, and more. "
            f"What would you like to know?"
        )
    
    return (
        "Hello! I'm your ADHD psychoeducation assistant. "
        "I can answer questions about ADHD symptoms, diagnosis, "
        "executive dysfunction, coping strategies, and more. "
        "What would you like to know?"
    )

OUT_OF_SCOPE_RESPONSE = (
    "That question is outside what I can reliably help with. "
    "I'm designed to provide educational information about ADHD — "
    "symptoms, terminology, coping strategies, and lived experiences. "
    "For personal medical decisions, medication questions, or diagnosis, "
    "please consult a qualified healthcare professional. "
    "You can find ADHD specialists through CHADD at chadd.org."
)

# RESPONSE SANITISATION 
UNSAFE_RESPONSE_PATTERNS = [
    r'\byou should take\b',
    r'\bi (diagnose|recommend you take|prescribe)\b',
    r'\bstop (taking|your) (medication|medicine|pills)\b',
    r'\btake \d+\s*mg\b',
    r'\byou (have|likely have|probably have) ADHD\b',
]

MANDATORY_DISCLAIMER = (
    "\n\n*This information is educational only. "
    "Always consult a qualified healthcare professional "
    "for personal medical decisions.*"
)

def sanitise_response(answer: str) -> str:
    answer_lower = answer.lower()
    for pattern in UNSAFE_RESPONSE_PATTERNS:
        if re.search(pattern, answer_lower, re.IGNORECASE):
            logger.warning(f"Unsafe response pattern detected, appending discalimer")
            if MANDATORY_DISCLAIMER not in answer:
                answer = answer + MANDATORY_DISCLAIMER
            break
    return answer



CLEAR_OUT_OF_SCOPE_PATTERNS = [
    r'\b(do\s+i\s+have|have\s+i\s+got|am\s+i\s+(adhd|autistic))\b',
    r'\b(diagnos\s+me|can\s+you\s+diagnose)\b',
    r'\b(dosage|how\s+much.*mg|milligram)\b',
    r'\b(ritalin|adderall|concerta|strattera|vyvanse|dexamphetamine)\b',
]

def is_clearly_out_of_scope(text: str) -> bool:
    """
    Regex check for unambiguously out-of-scope requests.
    Medication names and diagnosis requests don't need LLM classification.
    """
    text_lower = text.lower()
    for pattern in CLEAR_OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def check_input(text: str) -> dict:
    # Layer 1 — Crisis (regex, always runs)
    if is_crisis_message(text):
        return {
            "safe": False,
            "blocked_response": CRISIS_RESPONSE,
            "reason": "crisis_detected"
        }

    # Layer 2 — Obvious conversational (regex, free)
    if is_obviously_conversational(text):
        return {
            "safe": False,
            "blocked_response": build_conversational_response(text),
            "reason": "conversational"
        }

    # Layer 3 — Clear out-of-scope (regex, free)
    # Handles medication names and diagnosis requests reliably
    if is_clearly_out_of_scope(text):
        return {
            "safe": False,
            "blocked_response": OUT_OF_SCOPE_RESPONSE,
            "reason": "out_of_scope"
        }

    # Layer 4 — LLM classification (ambiguous cases only)
    intent = classify_intent_cached(text)

    if intent == "conversational":
        return {
            "safe": False,
            "blocked_response": build_conversational_response(text),
            "reason": "conversational"
        }

    if intent == "out_of_scope":
        return {
            "safe": False,
            "blocked_response": OUT_OF_SCOPE_RESPONSE,
            "reason": "out_of_scope"
        }

    return {"safe": True, "blocked_response": None, "reason": None}