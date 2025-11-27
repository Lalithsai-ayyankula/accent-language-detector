Team AOT – Accent-Aware Cuisine Recommendation System
NLP Mini Project | HuBERT + MFCC-CNN | Child Accent Inference + Dish Recommender
📝 1. Project Overview

This project builds an Accent Detection System using:

MFCC + CNN Baseline

HuBERT-based Deep Embeddings

Layer-wise analysis for HuBERT

Child speech accent inference

Accent-aware Cuisine Recommendation Application

The model predicts the accent of a speaker from speech audio and recommends regional dishes based on the detected accent.

📂 2. Folder Structure
Team_AOT/
│── code/
│   ├── app_cuisine_recommender.py
│   ├── datasets.py
│   ├── evaluate.py
│   ├── inference_child_audio.py
│   ├── layerwise_analysis.py
│   ├── models.py
│   ├── preprocess.py
│   ├── random_split_baseline.py
│   ├── train_hubert.py
│   ├── train_mfcc.py
│   └── word_sentence_analysis.py
│
│── models/
│   ├── best_cnn_mfcc.pth
│   ├── best_hubert_clf.pth
│   ├── hubert_classifier.joblib
│   └── mfcc_cnn.pth
│
│── outputs/
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── layerwise_results.json
│   └── mfcc_vs_hubert_results.csv
│
│── Team_AOT_project.ipynb
│── README.md
└── requirements.txt

🎯 3. Tasks Completed (as per project description)
✔ 1. Baseline MFCC Approach

Extract MFCCs

Train CNN classifier

Achieved high validation accuracy

Saved model + predictions

✔ 2. HuBERT-based Accent Detection

Extract HuBERT embeddings

Train classifier

Compare multiple layers

Layer-wise accuracy analysis

✔ 3. Word-/sentence-wise analysis

Evaluate accent prediction consistency across different speech segments

✔ 4. Child Accent Inference

Run trained models on child speech dataset

Save predictions to CSV

✔ 5. Accent-Aware Cuisine Recommender Application

Python app that detects accent and recommends dishes

Uses saved model

✔ 6. Outputs & Artifacts

Classification report

Confusion matrix

Predictions

JSON layer-wise results

MFCC vs HuBERT comparison

📦 4. How to Install Requirements

You can install everything using:

pip install -r requirements.txt


Or manually:

pip install torch torchaudio librosa numpy pandas scikit-learn matplotlib seaborn transformers datasets joblib gradio

▶️ 5. How to Run the Models (from .py files)
A) Train MFCC Model
python code/train_mfcc.py

B) Train HuBERT Model
python code/train_hubert.py

C) Generate Evaluation Outputs
python code/evaluate.py

D) Run Layer-wise Analysis
python code/layerwise_analysis.py

E) Predict Accent for Child Audio

Put child WAV files in:

child_audio/


Run:

python code/inference_child_audio.py --model-type hubert


Outputs saved to:

outputs/child_predictions.csv

F) Launch Cuisine Recommendation App
python code/app_cuisine_recommender.py

🍲 6. Mapping: Detected Accent → Recommended Dishes
Accent	Region	Recommended Dishes
Tamil	Tamil Nadu	Dosa, Pongal, Chettinad Chicken
Telugu	Andhra Pradesh	Pesarattu, Biryani, Gongura
Hindi	North India	Rajma Chawal, Chole, Aloo Paratha
Bengali	West Bengal	Fish Curry, Mishti Doi
Kannada	Karnataka	Bisi Bele Bath, Ragi Mudde

(You can extend the list in the app.)

📈 7. Results Summary

Best MFCC-CNN Accuracy: ~99%

Best HuBERT Accuracy: ~92%

Best HuBERT Layer: -10

Layer-wise analysis included

Confusion matrix, F1 score, and CSVs saved

All metrics are available in /outputs/.

🔗 8. Drive/GitHub Link



🙌 9. Team Members

AYYANKULA LALITH SAI KUMAR
GARIKIPATI BABY DHANUSHA
MAASA KEERTHI

