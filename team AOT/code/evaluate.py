# evaluate.py
import os, argparse
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib
import torch
import librosa
from transformers import AutoFeatureExtractor, HubertModel
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["mfcc","hubert"], default="mfcc")
parser.add_argument("--out-dir", default=".")
parser.add_argument("--max-samples", type=int, default=None)
args = parser.parse_args()

os.makedirs(os.path.join(args.out_dir, "outputs"), exist_ok=True)

# load dataset
ds = load_dataset("DarshanaS/IndicAccentDb", split="train")
rows = []
for i, ex in enumerate(ds):
    if args.max_samples and i >= args.max_samples:
        break
    audio = ex.get("audio")
    if audio is None:
        continue
    rows.append({"audio": np.array(audio["array"], dtype=np.float32), "sr": int(audio["sampling_rate"]), "label": ex["label"], "age": ex.get("age", None)})
df = pd.DataFrame(rows)
label_names = ds.features["label"].names
label_map = {name: idx for idx, name in enumerate(label_names)}
df["label_id"] = df["label"].map(label_map)

# choose test set (child set for mfcc mode)
def is_child(age):
    if age is None: return False
    try:
        a = int(age); return a < 16
    except: return ("child" in str(age).lower()) or ("kid" in str(age).lower())

df["is_child"] = df["age"].apply(is_child)
child_df = df[df["is_child"]].reset_index(drop=True)

if args.mode == "mfcc":
    # load model
    model_path = os.path.join(args.out_dir, "models", "mfcc_cnn.pth")
    if not os.path.exists(model_path):
        raise SystemExit("mfcc model not found")
    from models import SimpleMFCCCNN
    model = SimpleMFCCCNN(num_classes=len(label_names)).eval()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    preds, trues = [], []
    for _, row in child_df.iterrows():
        wav = row["audio"]
        sr = row["sr"]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        from preprocess import compute_mfcc_from_array
        mf = compute_mfcc_from_array(wav, sr=16000)
        x = np.expand_dims(np.transpose(mf), axis=0)  # 1 x n_mfcc x frames
        import torch
        xb = torch.tensor(np.expand_dims(x, axis=0)).float()  # batch 1
        out = model(xb)
        preds.append(int(out.argmax(dim=1).cpu().numpy()[0]))
        trues.append(int(row["label_id"]))
    acc = accuracy_score(trues, preds)
    cm = confusion_matrix(trues, preds)
    cr = classification_report(trues, preds, target_names=label_names, zero_division=0)
    pd.DataFrame({"true": trues, "pred": preds}).to_csv(os.path.join(args.out_dir, "outputs", "child_predictions_mfcc.csv"), index=False)
    np.save(os.path.join(args.out_dir, "outputs", "confusion_matrix_mfcc.npy"), cm)
    with open(os.path.join(args.out_dir, "outputs", "classification_report_mfcc.txt"), "w") as f: f.write(cr)
    print("MFCC adult->child acc:", acc)

else:
    # hubert mode: load sklearn joblib
    model_path = os.path.join(args.out_dir, "models", "hubert_classifier.joblib")
    if not os.path.exists(model_path):
        raise SystemExit("hubert model not found")
    clf = joblib.load(model_path)
    fe = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960", output_hidden_states=True).eval()
    X, y = [], []
    for _, row in df.iterrows():
        wav = row["audio"]
        sr = row["sr"]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        inputs = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = hubert(**{k: v for k, v in inputs.items()})
            emb = out.hidden_states[-1].mean(dim=1).cpu().numpy()[0]
        X.append(emb); y.append(int(row["label_id"]))
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.vstack(X), np.array(y), test_size=0.2, stratify=y, random_state=42)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    cr = classification_report(y_test, preds, target_names=label_names, zero_division=0)
    pd.DataFrame({"true": list(y_test), "pred": list(preds)}).to_csv(os.path.join(args.out_dir, "outputs", "predictions_hubert.csv"), index=False)
    np.save(os.path.join(args.out_dir, "outputs", "confusion_matrix_hubert.npy"), cm)
    with open(os.path.join(args.out_dir, "outputs", "classification_report_hubert.txt"), "w") as f: f.write(cr)
    print("HuBERT random-split acc:", acc)
