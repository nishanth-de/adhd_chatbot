import re
import logging

logger = logging.getLogger(__name__)

# CRISIS DETECTION
CRISIS_PATTERNS = [
    r'\b(suicid|kill\s+myself|end\s+my\s+life|want\s+to\s+die|self.?harm)\b',
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