# src/predict.py

import joblib
import numpy as np
import pandas as pd
from src.feature_extractor import get_embeddings
from src.preprocess import preprocess_conversations

def generate_rationale(text: str, predicted_intent: str) -> str:
    """Simple rule-based rationale generator (can improve later)"""
    if "site visit" in text or "schedule" in text:
        return "User mentioned a schedule or visit, indicating appointment booking."
    elif "price" in text or "discount" in text:
        return "User asked about pricing or negotiation."
    elif "class" in text or "product" in text:
        return "User is inquiring about product or service."
    elif "help" in text or "issue" in text:
        return "User is asking for support."
    else:
        return f"User message context suggests a {predicted_intent.lower()}."

def predict_intents(conversations, model_path="model/classifier.pkl"):
    # Load model
    bundle = joblib.load(model_path)
    clf = bundle["model"]
    encoder = bundle["encoder"]

    # Preprocess and embed
    processed = preprocess_conversations(conversations)
    embedded = get_embeddings(processed)

    results = []

    for i, item in enumerate(embedded):
        convo_id = item["conversation_id"]
        vector = item["embedding"].reshape(1, -1)
        pred_label_index = clf.predict(vector)[0]
        intent = encoder.inverse_transform([pred_label_index])[0]

        rationale = generate_rationale(processed[i]["text"], intent)

        results.append({
            "conversation_id": convo_id,
            "predicted_intent": intent,
            "rationale": rationale
        })

    return results

def save_outputs(results, json_path="outputs/predictions.json", csv_path="outputs/predictions.csv"):
    # Save as JSON
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save as CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
