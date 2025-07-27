import re
import emoji
from typing import List, Dict

def clean_text(text: str) -> str:
    text = text.lower()
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_last_n_messages(messages: List[Dict[str, str]], n: int = 6) -> str:
    recent = messages[-n:] if len(messages) >= n else messages
    return " ".join([f"{msg['sender']}: {clean_text(msg['text'])}" for msg in recent])

def preprocess_conversations(data: List[Dict], n_messages: int = 6) -> List[Dict]:
    processed = []
    for convo in data:
        convo_id = convo['conversation_id']
        combined_text = get_last_n_messages(convo['messages'], n_messages)
        processed.append({
            "conversation_id": convo_id,
            "text": combined_text
        })
    return processed
