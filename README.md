# Team AOT – Accent-Aware Cuisine Recommendation System
NLP Mini Project | MFCC-CNN + HuBERT | Accent Detection & Cuisine Recommendation

---

## 1. Project Overview
This project focuses on detecting the accent of a speaker using speech audio. Two approaches were implemented:

1. MFCC + CNN baseline classifier  
2. HuBERT-based deep embedding classifier

Based on the predicted accent, the system recommends regional cuisine.  
The project also includes:
- Accent detection for child audio
- Word/sentence-level analysis
- HuBERT layer-wise performance analysis
- Output reports and visualizations
- A cuisine recommendation application

The entire workflow is documented in the Jupyter Notebook.

---

## 2. Folder Structure

<pre>
Team_AOT/
│
├── code/
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
├── models/
│   ├── best_cnn_mfcc.pth
│   ├── best_hubert_clf.pth
│   ├── hubert_classifier.joblib
│   └── mfcc_cnn.pth
│
├── outputs/
│   ├── predictions.csv
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   ├── layerwise_results.json
│   └── mfcc_vs_hubert_results.csv
│
├── Team_AOT_project.ipynb
├── README.md
└── requirements.txt
</pre>


---

## 3. Tasks Completed (As per Project Description)

### MFCC + CNN Baseline
- Extracted MFCC features
- Built CNN classifier
- Achieved high validation accuracy
- Saved model and predictions

### HuBERT-Based Classifier
- Extracted HuBERT embeddings
- Trained classifier (MLP/SVM)
- Performed layer-wise accuracy comparison
- Saved model and best layer results

### Additional Tasks
- Word & sentence-level accent analysis
- Child audio inference pipeline
- Cuisine recommendation application
- Detailed outputs saved in `outputs/`

---

## 4. Installation

Install required packages:
pip install -r requirements.txt


Or manually:
pip install torch torchaudio librosa numpy pandas scikit-learn matplotlib seaborn transformers datasets joblib gradio

---

## 5. How to Run the Code

### 5.1 Train MFCC Model
python code/train_mfcc.py

### 5.2 Train HuBERT Model
python code/train_hubert.py

### 5.3 Evaluate Models
python code/evaluate.py

### 5.4 Run HuBERT Layer-wise Analysis
python code/layerwise_analysis.py

### 5.5 Run Child Accent Inference
Place child wav files into:
child_audio/

Run:
python code/inference_child_audio.py --model-type hubert

Output:
outputs/child_predictions.csv

### 5.6 Run Cuisine Recommendation Application
python code/app_cuisine_recommender.py

---

## 6. Outputs Generated

| File | Description |
|------|-------------|
| predictions.csv | Model predictions for test data |
| classification_report.txt | Precision, recall, F1-score |
| confusion_matrix.png | Confusion matrix heatmap |
| layerwise_results.json | Accuracy per HuBERT layer |
| mfcc_vs_hubert_results.csv | Comparison of MFCC vs HuBERT |

---

## 7. Accent → Cuisine Mapping

| Accent | Region | Recommended Dishes |
|--------|--------|--------------------|
| Tamil | Tamil Nadu | Dosa, Idli, Pongal, Chettinad Chicken |
| Telugu | Andhra Pradesh | Pesarattu, Biryani, Gongura Pachadi |
| Bengali | West Bengal | Fish Curry, Roshogolla, Mishti Doi |
| Hindi | North India | Rajma, Chole Bhature, Paratha |
| Kannada | Karnataka | Bisi Bele Bath, Ragi Mudde |

---

## 8. Notebook Description
`Team_AOT_project.ipynb` includes:

- Data loading
- MFCC feature extraction
- CNN model training & evaluation
- HuBERT embeddings extraction
- HuBERT classifier training
- Layer-wise analysis
- Child accent inference
- Recommendation demo

---

## 9. How to Cite Models and Files
All trained models are stored in the `models/` directory.  
All experiment outputs are stored in the `outputs/` directory.

---

## 10. Contributors
Team AOT  
AYYANKULA LALITH SAI KUMAR
GARIKIPATI BABY DHANUSHA
MAASA KEERTHI

---





