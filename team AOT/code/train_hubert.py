# train_hubert.py
import os, argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import librosa

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--layer", type=int, default=-1)
parser.add_argument("--out-dir", type=str, default=".")
args = parser.parse_args()

os.makedirs(os.path.join(args.out_dir, "models"), exist_ok=True)
os.makedirs(os.path.join(args.out_dir, "outputs"), exist_ok=True)

print("Loading dataset...")
ds = load_dataset("DarshanaS/IndicAccentDb", split="train")
rows = []
for i, ex in enumerate(ds):
    if args.max_samples and i >= args.max_samples:
        break
    audio = ex.get("audio")
    if audio is None:
        continue
    rows.append({
        "audio": np.array(audio["array"], dtype=np.float32),
        "sr": int(audio["sampling_rate"]),
        "label": ex["label"]
    })
df = pd.DataFrame(rows)
label_names = ds.features["label"].names
label_map = {name: idx for idx, name in enumerate(label_names)}
df["label_id"] = df["label"].map(label_map)

fe = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True).eval()

def extract_emb(wav, sr, layer=-1):
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    inputs = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = hubert(**inputs)
        hs = out.hidden_states
        vec = hs[layer].mean(dim=1).cpu().numpy()[0]
    return vec

print("Extracting embeddings (this may take time)...")
embs, labels = [], []
for i, row in df.iterrows():
    if i % 50 == 0:
        print(f"Processed {i}/{len(df)}")
    emb = extract_emb(row["audio"], row["sr"], layer=args.layer)
    embs.append(emb)
    labels.append(int(row["label_id"]))

X = np.vstack(embs)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=400)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print("HuBERT classifier accuracy:", acc)

joblib.dump(clf, os.path.join(args.out_dir, "models", "hubert_classifier.joblib"))
with open(os.path.join(args.out_dir, "outputs", "classification_report_hubert.txt"), "w") as f:
    f.write(classification_report(y_test, preds, target_names=label_names, zero_division=0))
with open(os.path.join(args.out_dir, "outputs", "mfcc_vs_hubert_results.csv"), "a") as f:
    f.write(f"hubert,{acc:.4f}\n")
