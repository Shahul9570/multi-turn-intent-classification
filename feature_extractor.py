from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict
from tqdm import tqdm

# Load tokenizer and model from Hugging Face
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Put model in eval mode (we don’t train it here)
model.eval()

def get_embeddings(conversations: List[Dict]) -> List[Dict]:
    """Generate BERT embeddings for each conversation's text"""
    embeddings = []

    for item in tqdm(conversations, desc="Generating embeddings"):
        convo_id = item['conversation_id']
        text = item['text']

        # Tokenize the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # Get model output
        with torch.no_grad():
            outputs = model(**inputs)
            # Use the [CLS] token embedding (first token)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        embeddings.append({
            "conversation_id": convo_id,
            "embedding": cls_embedding
        })

    return embeddings
