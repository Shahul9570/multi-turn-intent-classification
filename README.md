# Multi-Turn Intent Classification System

This project classifies user intent in multi-turn conversations using a fine-tuned open-source transformer model (DistilBERT). The system is capable of understanding full conversations, reasoning context, and predicting intent such as booking appointments, pricing negotiations, or product inquiries.

---

## 🚀 Setup Instructions

### 1. Clone the Repository & Set Up Environment


git clone <your-repo-url>
cd multi_turn_intent_classification
python -m venv venv
# For Windows:
venv\Scripts\activate
# For Unix/macOS:
source venv/bin/activate
pip install -r requirements.txt


### 2. Run the Training Pipeline
python main.py

### 3. Predict on New Test Conversations

python test.py

## 🧠 Model Choice

* **Model:** DistilBERT (`distilbert-base-uncased`)
* **Why?** Lightweight, open-source, efficient for small to mid-scale datasets.
* **Framework:** Hugging Face Transformers
* **Trained Locally:** Using Hugging Face `Trainer` (no external API used)

This model was fine-tuned on labeled multi-turn conversations where user intent was tagged manually.

---

## 📊 Sample Predictions

Sample output from `outputs/test_predictions.json`:

```json
{
  "conversation_id": "conv_012",
  "predicted_intent": "Pricing Negotiation",
  "rationale": "User discussed pricing, indicating negotiation."
}
```

Each prediction includes both the predicted intent and the reasoning behind it.

---

## ⚠️ Limitations

* **Intent Overlap:** Booking vs Pricing can overlap in phrasing.
* **Training Size:** Limited training data can restrict performance.
* **Languages:** Only English supported in current version.
* **Confidence:** Model does not reject uncertain predictions (can be extended with thresholding).

## 📂 Project Structure

multi_turn_intent_classification/
|
├── data/
│   ├── conversations.json
│   ├── labels.json
│   ├── new_conversations.json
│
├── src/
│   ├── preprocess.py
│   ├── feature_extractor.py
│   ├── classifier.py
│   ├── fine_tune.py
│   ├── bert_predictor.py
│   └── predict.py
│
├── outputs/
│   ├── predictions.json
│   └── test_predictions.json
│
├── main.py
├── test.py
├── requirements.txt
└── README.md
```

---

## ✅ Compliance with Task Requirements

* ✅ **Model:** Used only open-source model `distilbert-base-uncased` from Hugging Face (Apache 2.0 license)
* ✅ **Language:** Python
* ✅ **Frameworks:** Hugging Face Transformers, Datasets, Scikit-learn, Pandas
* ✅ **No commercial APIs:** No OpenAI, Claude, etc. used
* ✅ **Output Format:** Predictions saved to both `.json` and `.csv`
* ✅ **Scalability:** Code designed to handle thousands of conversations

---

## 🙏 Credits

Built by Muhammed Shahul as part of a Machine Learning Intern Task.
