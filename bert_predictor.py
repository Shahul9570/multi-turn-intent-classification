from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import joblib
import numpy as np
import pandas as pd

def predict_with_finetuned_bert(processed_data, model_path="model/bert"):
    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    encoder = joblib.load("model/bert_encoder.pkl")
    
    model.eval()
    
    results = []

    for item in processed_data:
        convo_id = item["conversation_id"]
        text = item["text"]

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            label = encoder.inverse_transform([prediction])[0]

        rationale = generate_rationale(text, label)
        results.append({
            "conversation_id": convo_id,
            "predicted_intent": label,
            "rationale": rationale
        })

    return results

def generate_rationale(text, predicted_intent):
    """Basic rule-based rationale generation"""
    if "schedule" in text or "book" in text or "meeting" in text:
        return "User mentioned scheduling or booking, indicating appointment intent."
    if "price" in text or "discount" in text:
        return "User discussed pricing, indicating negotiation."
    if "class" in text or "product" in text or "compatible" in text:
        return "User asked about product or service, suggesting inquiry."
    if "not arrived" in text or "damaged" in text or "issue" in text:
        return "User mentioned a problem or delay, indicating support request."
    if "follow" in text or "waiting" in text:
        return "User is following up on a previous interaction."
    return f"User message context suggests a {predicted_intent.lower()}."

def save_outputs(results, json_path="outputs/predictions.json", csv_path="outputs/predictions.csv"):
    import json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
