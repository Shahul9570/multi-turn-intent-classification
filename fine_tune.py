# src/fine_tune.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch

def fine_tune_bert(processed, labels_json, model_name="distilbert-base-uncased", save_path="model/bert"):
    # Map conversation_id to label
    label_map = {item["conversation_id"]: item["intent"] for item in labels_json}

    # Filter only labeled conversations
    texts, labels = [], []
    for convo in processed:
        if convo["conversation_id"] in label_map:
            texts.append(convo["text"])
            labels.append(label_map[convo["conversation_id"]])

    # Encode labels
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Save encoder
    import joblib
    joblib.dump(encoder, f"{save_path}_encoder.pkl")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True)

    data = Dataset.from_dict({"text": texts, "label": y})
    data = data.map(tokenize)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(set(y)))

    # Training config
    args = TrainingArguments(
        output_dir=save_path,
        evaluation_strategy="epoch",
        logging_steps=10,
        num_train_epochs=10,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=data,
        eval_dataset=data,
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)  # ✅ Save tokenizer too

    print(f"✅ Fine-tuned model saved at: {save_path}")
