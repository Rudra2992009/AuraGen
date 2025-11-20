import re

# List of prohibited phrases and patterns for deepfake/explicit/sexual content
PROHIBITED_PHRASES = [
    'deepfake', 'sex video', 'blowjob', 'nude', 'full cloth opening',
    'porn', 'explicit', 'nsfw', 'masturbation', 'erotic', 'strip', 'striptease',
    'penetration', 'cumshot', 'orgasm', 'genital', 'breast', 'vulva', 'penis', 'sperm',
    'rape', 'molestation', 'sexual assault', 'incest', 'bestiality'
]
PROHIBITED_PATTERNS = [
    r'\b(?i)deepfake(s)?\b', r'(?i)sex(ual)? video', r'(?i)blowjob', r'(?i)nude', r'(?i)porn', r'(?i)explicit',
    r'(?i)nsfw', r'(?i)masturbat(e|ion)', r'(?i)erotic', r'(?i)strip(tease)?', r'(?i)penetration',
    r'(?i)cumshot', r'(?i)orgasm', r'(?i)genital(s)?', r'(?i)breast(s)?', r'(?i)vulva', r'(?i)penis', r'(?i)sperm',
    r'(?i)rape', r'(?i)molest(ation|ed)', r'(?i)sexual assault', r'(?i)incest', r'(?i)bestiality'
]

# Safety check for prompt text

def is_prompt_safe(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    for phrase in PROHIBITED_PHRASES:
        if phrase in prompt_lower:
            return False
    for pattern in PROHIBITED_PATTERNS:
        if re.search(pattern, prompt_lower):
            return False
    return True

# Usage: Call is_prompt_safe(prompt_str) before generating
# If False, reject the request
