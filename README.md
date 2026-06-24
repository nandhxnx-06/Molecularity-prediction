# Molecular Property Prediction using Deep Learning

## Overview
This project focuses on predicting molecular properties, specifically **compound solubility**, from molecular SMILES representations using deep learning techniques. Accurate prediction of molecular properties plays a crucial role in drug discovery, pharmaceutical research, and chemical analysis.

The project explores and compares multiple sequence-based deep learning models including **LSTM**, **GRU**, and **Transformer** architectures to learn meaningful patterns from molecular structures and estimate solubility levels.

---

## Problem Statement
Determining molecular properties through laboratory experiments can be expensive and time-consuming. This project aims to automate the prediction process by leveraging deep learning models trained on molecular SMILES data.

---

## Features
- Molecular property prediction from SMILES strings
- Data preprocessing and tokenization
- Deep learning-based sequence modeling
- Comparison of LSTM, GRU, and Transformer models
- Performance evaluation using regression metrics
- Visualization of training and validation results

---

## Dataset
The dataset contains molecular compounds represented as **SMILES (Simplified Molecular Input Line Entry System)** strings along with their corresponding solubility values.

### Sample Data

| SMILES | Solubility |
|--------|------------|
| CCO    | 0.85 |
| CCN    | 0.72 |
| CCC    | 0.41 |

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Deep Learning
- Natural Language Processing Concepts
- Google Colab

---

## Model Architecture

### LSTM Model
- Embedding Layer
- LSTM Layer
- Dense Layers
- Output Layer

### GRU Model
- Embedding Layer
- GRU Layer
- Dense Layers
- Output Layer

### Transformer Model
- Token Embeddings
- Multi-Head Attention
- Feed Forward Network
- Regression Output Layer

---

## Workflow

1. Load Molecular Dataset
2. Preprocess SMILES Strings
3. Tokenize Molecular Sequences
4. Train Deep Learning Models
5. Evaluate Model Performance
6. Compare Results
7. Predict Solubility for New Molecules

---

## Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Project Structure

```
Molecular-Property-Prediction/
│
├── app.py
├── model.pth
├── vocab.pkl
├── stats.pkl
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

---

## Applications

- Drug Discovery
- Pharmaceutical Research
- Chemical Property Analysis
- Molecular Screening
- Computational Chemistry

---

## Future Enhancements

- Integration of Graph Neural Networks (GNNs)
- Prediction of multiple molecular properties
- Deployment as a web application
- Real-time molecular analysis dashboard

---

## Results

The deep learning models successfully learned molecular patterns from SMILES representations and provided reliable solubility predictions. Comparative analysis helped identify the most effective architecture for molecular property prediction.

