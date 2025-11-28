# layerwise_analysis.py
import os, argparse, numpy as np, pandas as pd
from datasets import load_dataset
from transformers import AutoFeatureExtractor, HubertModel
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import librosa

parser = argparse.ArgumentParser()
parser.add_argument("--layers", type=int, nargs="+", default=list(range(-1, -13, -1)))
parser.add_argument("--max-samples", type=int, default=500)
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

os.makedirs(os.path.join(args.out_dir, "outputs"), exist_ok=True)
ds = load_dataset("DarshanaS/IndicAccentDb", split="train")
rows=[]
for i,ex in enumerate(ds):
    if i>=args.max_samples: break
    audio = ex.get("audio")
    if audio is None: continue
    rows.append({"audio": np.array(audio["array"], dtype=np.float32), "sr": int(audio["sampling_rate"]), "label": ex["label"]})
import pandas as pd
df = pd.DataFrame(rows)
label_map = {name:i for i,name in enumerate(ds.features["label"].names)}
df["label_id"] = df["label"].map(label_map)

fe = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True).eval()

def extract_layer_emb(wav,sr,layer):
    if sr!=16000:
        wav = librosa.resample(wav,orig_sr=sr,target_sr=16000)
    inputs = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        out = hubert(**{k: v for k,v in inputs.items()})
        emb = out.hidden_states[layer].mean(dim=1).cpu().numpy()[0]
    return emb

layer_acc = {}
for layer in args.layers:
    X=[]; y=[]
    for _,row in df.iterrows():
        X.append(extract_layer_emb(row["audio"], row["sr"], layer))
        y.append(int(row["label_id"]))
    X = np.vstack(X); y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=400).fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    layer_acc[str(layer)] = float(acc)
    print("Layer", layer, "acc:", acc)

import json
with open(os.path.join(args.out_dir, "outputs", "layerwise_results.json"), "w") as f:
    json.dump(layer_acc, f, indent=2)
print("Saved layerwise results.")
