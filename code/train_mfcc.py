# train_mfcc.py
import os, argparse
import numpy as np
import pandas as pd
import librosa
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from preprocess import compute_mfcc_from_array
from datasets import TorchMFCCDataset
from models import SimpleMFCCCNN

parser = argparse.ArgumentParser()
parser.add_argument("--max-samples", type=int, default=None)
parser.add_argument("--n-mfcc", type=int, default=40)
parser.add_argument("--max-frames", type=int, default=200)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--out-dir", type=str, default=".")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
        "label": ex["label"],
        "age": ex.get("age", None)
    })

df = pd.DataFrame(rows)
label_names = ds.features["label"].names
label_map = {name: idx for idx, name in enumerate(label_names)}
df["label_id"] = df["label"].map(label_map)

# simple child detection
def is_child(age):
    if age is None: return False
    try:
        a = int(age)
        return a < 16
    except:
        return ("child" in str(age).lower()) or ("kid" in str(age).lower())

df["is_child"] = df["age"].apply(is_child)
adult_df = df[~df["is_child"]].reset_index(drop=True)
child_df = df[df["is_child"]].reset_index(drop=True)

train_df, val_df = train_test_split(adult_df, test_size=0.15, stratify=adult_df["label_id"], random_state=42)

def prepare_mfcc(df_subset):
    X, y = [], []
    for _, row in df_subset.iterrows():
        wav = row["audio"]
        sr = row["sr"]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        mf = compute_mfcc_from_array(wav, sr=16000, n_mfcc=args.n_mfcc, max_frames=args.max_frames)
        X.append(mf)
        y.append(int(row["label_id"]))
    return X, y

print("Preparing MFCCs ...")
X_train, y_train = prepare_mfcc(train_df)
X_val, y_val = prepare_mfcc(val_df)
X_test, y_test = prepare_mfcc(child_df)

train_ds = TorchMFCCDataset(X_train, y_train)
val_ds = TorchMFCCDataset(X_val, y_val)
test_ds = TorchMFCCDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=args.batch_size)

model = SimpleMFCCCNN(num_classes=len(label_names), n_mfcc=args.n_mfcc, max_frames=args.max_frames).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.CrossEntropyLoss()

best_val = 0.0
for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item() * yb.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    train_acc = correct / total if total else 0.0

    # validation
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            val_preds.extend(out.argmax(dim=1).cpu().numpy().tolist())
            val_trues.extend(yb.cpu().numpy().tolist())
    from sklearn.metrics import accuracy_score
    val_acc = accuracy_score(val_trues, val_preds) if val_trues else 0.0

    print(f"Epoch {epoch+1}/{args.epochs} train_acc={train_acc:.4f} val_acc={val_acc:.4f} loss={total_loss/total:.4f}")

    if val_acc > best_val:
        best_val = val_acc
        torch.save(model.state_dict(), os.path.join(args.out_dir, "models", "mfcc_cnn.pth"))

print("Best validation acc:", best_val)
# Evaluate on child set (adult->child)
model.load_state_dict(torch.load(os.path.join(args.out_dir, "models", "mfcc_cnn.pth"), map_location=DEVICE))
model.eval()
test_preds, test_trues = [], []
with torch.no_grad():
    for xb, yb in test_ds:
        xb = xb.unsqueeze(0).to(DEVICE)
        out = model(xb)
        test_preds.append(int(out.argmax(dim=1).cpu().numpy()[0]))
        test_trues.append(int(yb))
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(test_trues, test_preds) if test_trues else 0.0
print("Adult->Child accuracy (MFCC):", acc)

# save outputs
import json, pandas as pd
os.makedirs(os.path.join(args.out_dir, "outputs"), exist_ok=True)
pd.DataFrame({"true": test_trues, "pred": test_preds}).to_csv(os.path.join(args.out_dir, "outputs", "child_predictions_mfcc.csv"), index=False)
with open(os.path.join(args.out_dir, "outputs", "classification_report_mfcc.txt"), "w") as f:
    f.write(classification_report(test_trues, test_preds, target_names=label_names, zero_division=0))
import numpy as np
np.save(os.path.join(args.out_dir, "outputs", "confusion_matrix_mfcc.npy"), confusion_matrix(test_trues, test_preds))
