import json
import numpy as np
from src.preprocess import preprocess_conversations

# Step 1: Load raw data
with open("data/conversations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Preprocess
processed = preprocess_conversations(data)

# Step 3: Load labels
with open("data/labels.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

# Step 4: Fine-tune a transformer model
from src.fine_tune import fine_tune_bert
fine_tune_bert(processed, labels)

# Step 5: Predict using fine-tuned model
from src.bert_predictor import predict_with_finetuned_bert, save_outputs
predictions = predict_with_finetuned_bert(processed)

# Step 6: Save to JSON & CSV
save_outputs(predictions)

print("✅ Predictions saved to outputs/predictions.json and .csv")
