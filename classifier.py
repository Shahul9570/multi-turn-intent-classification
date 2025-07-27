import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

INTENT_LABELS = [
    "Book Appointment",
    "Product Inquiry",
    "Pricing Negotiation",
    "Support Request",
    "Follow-Up"
]

def train_classifier(embeddings, labels_json, save_path="model/classifier.pkl"):
    """Train classifier on embeddings + labels, and save model"""
    # Create mapping of conversation_id to intent
    label_map = {item["conversation_id"]: item["intent"] for item in labels_json}

    X, y = [], []
    for item in embeddings:
        convo_id = item["conversation_id"]
        if convo_id in label_map:
            X.append(item["embedding"])
            y.append(label_map[convo_id])

    X = np.array(X)
    y = np.array(y)

    # Convert intent labels to numbers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train simple logistic regression classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y_encoded)

    # Save model + label encoder
    joblib.dump({"model": clf, "encoder": label_encoder}, save_path)
    print(f"✅ Model saved to {save_path}")
