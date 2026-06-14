# 🏥 Healthcare AI Chatbot — Intelligent Medical Symptom Analyzer

> An NLP-powered chatbot that analyzes user-reported symptoms and provides intelligent preliminary medical guidance using machine learning classification and a curated disease-symptom knowledge base.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Model Pipeline](#model-pipeline)
- [API Endpoints](#api-endpoints)
- [Future Work](#future-work)

---

## Overview

Healthcare accessibility remains a major challenge in developing countries. This project builds an **intelligent conversational agent** that:

1. **Parses natural language symptom descriptions** using NLP preprocessing (tokenization, lemmatization, stopword removal)
2. **Classifies symptoms** into disease categories using a trained ML model
3. **Retrieves relevant medical guidance** from a curated knowledge base of 5,000+ disease-symptom mappings
4. **Provides structured responses** with confidence scores and recommended next steps

> ⚠️ **Disclaimer:** This is an educational project. It does not replace professional medical advice.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     User Interface                       │
│              (Web Chat / REST API Client)                │
└─────────────────┬───────────────────────────────────────┘
                  │ HTTP Request (symptom text)
                  ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                         │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ NLP Pipeline  │  │ ML Classifier │  │ Knowledge Base│  │
│  │              │  │              │  │               │  │
│  │ • Tokenizer  │──▶│ • TF-IDF     │──▶│ • 5000+      │  │
│  │ • Lemmatizer │  │ • SVM/RF     │  │   disease-    │  │
│  │ • Stopwords  │  │ • Ensemble   │  │   symptom     │  │
│  │ • Spell Fix  │  │              │  │   mappings    │  │
│  └──────────────┘  └──────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────┘
                  │
                  ▼ JSON Response
┌─────────────────────────────────────────────────────────┐
│ { "disease": "...", "confidence": 0.87,                  │
│   "description": "...", "precautions": [...] }           │
└─────────────────────────────────────────────────────────┘
```

---

## Key Results

| Metric | Score |
|--------|-------|
| **Intent Classification Accuracy** | 87.2% |
| **Symptom-Disease Mapping F1** | 0.83 |
| **Precision (macro avg)** | 0.85 |
| **Recall (macro avg)** | 0.81 |
| **Average Response Latency** | < 2 seconds |
| **Knowledge Base Coverage** | 5,000+ disease-symptom pairs |

### Classification Report (Top Categories)

```
                    precision    recall  f1-score   support
     Common Cold       0.91      0.89      0.90       142
        Diabetes       0.88      0.85      0.86       128
    Hypertension       0.86      0.82      0.84       115
       Migraine        0.84      0.88      0.86       103
     Skin Allergy      0.82      0.79      0.80        97
    ...
    macro avg          0.85      0.81      0.83      1250
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **ML Framework** | scikit-learn, NLTK, spaCy |
| **Feature Extraction** | TF-IDF Vectorizer |
| **Models** | SVM, Random Forest, Voting Ensemble |
| **Backend** | FastAPI |
| **Data** | Medical Q&A Dataset, Custom Disease-Symptom KB |
| **Evaluation** | cross_val_score, classification_report, confusion_matrix |

---

## Dataset

- **Medical Q&A Dataset (MedQuAD)** — 47K+ curated medical question-answer pairs
- **Custom Disease-Symptom KB** — 5,000+ manually verified disease-symptom-precaution mappings
- **Preprocessing:** Lowercasing → Tokenization → Lemmatization → Stopword removal → TF-IDF

---

## Getting Started

### Prerequisites
```bash
python >= 3.9
pip install -r requirements.txt
```

### Installation
```bash
git clone https://github.com/diwakar453t/Heathcare_Chatbot.git
cd Heathcare_Chatbot
pip install -r requirements.txt
```

### Run
```bash
# Train model
python train_model.py

# Start API server
uvicorn app:app --reload --port 8000
```

### Usage
```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "symptoms": "I have headache, fever, and body pain for 3 days"
})
print(response.json())
# {"disease": "Common Cold", "confidence": 0.89, "precautions": [...]}
```

---

## Model Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('clf', VotingClassifier(estimators=[
        ('svm', SVC(kernel='rbf', probability=True, C=10)),
        ('rf', RandomForestClassifier(n_estimators=200, max_depth=20))
    ], voting='soft'))
])
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/predict` | Analyze symptoms and return diagnosis |
| `GET` | `/diseases` | List all supported diseases |
| `GET` | `/health` | API health check |

---

## Future Work

- [ ] Fine-tune a BioBERT model for medical NER
- [ ] Add multi-turn conversation memory
- [ ] Integrate drug interaction database
- [ ] Deploy on AWS Lambda with API Gateway
- [ ] Add voice input support (speech-to-text)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Built by [Diwakar Patel](https://github.com/diwakar453t)** | Google Student Ambassador | AI/ML Engineer
