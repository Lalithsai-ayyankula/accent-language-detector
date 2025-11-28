# word_sentence_analysis.py
# This script splits dataset into word-level and sentence-level (based on 'text' length)
# then extracts HuBERT features or MFCC and compares accuracies.

import os, argparse
import numpy as np, pandas as pd, librosa
import torch
from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["hubert","mfcc"], default="hubert")
parser.add_argument("--max-samples", type=int, default=1000)
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

os.makedirs(os.path.join(args.out_dir,"outputs"), exist_ok=True)
ds = load_dataset("DarshanaS/IndicAccentDb", split="train")
rows=[]
for i,ex in enumerate(ds):
    if i>=args.max_samples: break
    audio = ex.get("audio")
    if audio is None: continue
    rows.append({"audio": np.array(audio["array"],dtype=np.float32), "sr": int(audio["sampling_rate"]), "label": ex["label"], "text": ex.get("text","")})
df = pd.DataFrame(rows)
label_map={name:i for i,name in enumerate(ds.features["label"].names)}
df["label_id"]=df["label"].map(label_map)

# split into word-level vs sentence-level
df["is_word"] = df["text"].apply(lambda t: len(t.split())==1)
word_df = df[df["is_word"]].reset_index(drop=True)
sent_df = df[~df["is_word"]].reset_index(drop=True)

def eval_subset(df_subset):
    # extract hubert emb or mfcc and train a quick classifier
    if args.mode=="hubert":
        fe = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
        hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True).eval()
        X=[]; y=[]
        for _,row in df_subset.iterrows():
            wav=row["audio"]; sr=row["sr"]
            inputs = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                out = hubert(**{k:v for k,v in inputs.items()})
                emb = out.hidden_states[-1].mean(dim=1).cpu().numpy()[0]
            X.append(emb); y.append(int(row["label_id"]))
    else:
        # mfcc path: simple mfcc extraction & flatten/aggregate
        X=[]; y=[]
        for _,row in df_subset.iterrows():
            wav=row["audio"]; sr=row["sr"]
            if sr!=16000:
                wav=librosa.resample(wav, orig_sr=sr, target_sr=16000)
            mf = librosa.feature.mfcc(y=wav, sr=16000, n_mfcc=40).mean(axis=1)
            X.append(mf); y.append(int(row["label_id"]))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(X), np.array(y), test_size=0.2, stratify=y, random_state=42)
    clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400).fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc

word_acc = eval_subset(word_df) if len(word_df)>10 else None
sent_acc = eval_subset(sent_df) if len(sent_df)>10 else None

out = {"word_acc": word_acc, "sentence_acc": sent_acc}
import json
with open(os.path.join(args.out_dir,"outputs","word_sentence_results.json"),"w") as f:
    json.dump(out, f, indent=2)
print("Saved word/sentence results:", out)
