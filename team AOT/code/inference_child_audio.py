import os
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
from transformers import HubertModel, Wav2Vec2FeatureExtractor

# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
CHILD_DIR = r"C:\Users\saiay\Downloads\team AOT\child samples"
MODEL_DIR = r"C:\Users\saiay\Downloads\team AOT\models"
OUT_DIR = r"C:\Users\saiay\Downloads\team AOT\outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SUPPORTED_AUDIO = (".wav", ".mp3", ".flac", ".ogg")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
MAX_DURATION = 5.0
MAX_LEN = int(SAMPLE_RATE * MAX_DURATION)

# ---------------------------------------------------
# ACCENT LABEL MAPPING
# ---------------------------------------------------
ACCENT_LABELS = {
    0: "andhra_pradesh",
    1: "gujarat",
    2: "jharkhand",
    3: "karnataka",
    4: "kerala",
    5: "tamil"
}

# ---------------------------------------------------
# DEFINE CLASSIFIER ARCHITECTURE (MATCHES TRAINING)
# ---------------------------------------------------
class HubertClassifier(nn.Module):
    def __init__(self, embedding_dim=768, num_classes=6):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
print("🔹 Loading HuBERT backbone and classifier...")

try:
    # Load HuBERT backbone
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/hubert-base-ls960",
        cache_dir=MODEL_DIR
    )
    hubert = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960",
        cache_dir=MODEL_DIR
    ).to(DEVICE)
    hubert.eval()
    
    # Load trained classifier
    clf_path = os.path.join(MODEL_DIR, "best_hubert_clf.pth")
    
    checkpoint = torch.load(clf_path, map_location=DEVICE)
    
    hubert_clf = HubertClassifier(
        embedding_dim=768,
        num_classes=6
    ).to(DEVICE)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            hubert_clf.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            hubert_clf.load_state_dict(checkpoint['state_dict'])
        else:
            hubert_clf.load_state_dict(checkpoint)
    else:
        hubert_clf.load_state_dict(checkpoint)
    
    hubert_clf.eval()
    print("✅ Models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def extract_hubert_embeddings(inputs, hubert_model, layer=-1):
    with torch.no_grad():
        outputs = hubert_model(inputs.to(DEVICE), output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer]
        embeddings = hidden_states.mean(dim=1)
    return embeddings

# ---------------------------------------------------
# FIND AUDIO FILES
# ---------------------------------------------------
audio_files = []
for root, _, files in os.walk(CHILD_DIR):
    for f in files:
        if f.lower().endswith(SUPPORTED_AUDIO):
            audio_files.append(os.path.join(root, f))

if len(audio_files) == 0:
    raise ValueError(f"No audio files found in {CHILD_DIR}!")

print(f"🔍 Found {len(audio_files)} audio files.")

# ---------------------------------------------------
# PROCESS AUDIO FILES
# ---------------------------------------------------
results = []
failed = 0
softmax = nn.Softmax(dim=1)

for i, audio_path in enumerate(audio_files, 1):
    try:
        print(f"Processing {i}/{len(audio_files)}: {os.path.basename(audio_path)}...", end=" ")
        
        wav, sr = librosa.load(audio_path, sr=None)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        wav = wav.astype(np.float32)

        if len(wav) > MAX_LEN:
            wav = wav[:MAX_LEN]
        else:
            wav = np.pad(wav, (0, MAX_LEN - len(wav)))

        inputs = feature_extractor(
            wav, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt"
        )["input_values"]
        
        embeddings = extract_hubert_embeddings(inputs, hubert, layer=-1)
        
        with torch.no_grad():
            logits = hubert_clf(embeddings)
            probs = softmax(logits)
            pred_id = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_id].item()

        accent_name = ACCENT_LABELS.get(pred_id, "unknown")

        results.append((audio_path, accent_name, confidence))
        print(f"✔ Accent: {accent_name}, Confidence: {confidence:.2%}")

    except Exception as e:
        print(f"❌ Error: {e}")
        results.append((audio_path, "ERROR", 0.0))
        failed += 1

## ---------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------
df = pd.DataFrame(results, columns=["audio_path", "predicted_accent", "confidence"])
df["confidence_percentage"] = df["confidence"].apply(
    lambda x: round(x*100, 2) if isinstance(x, (int, float)) else None
)

csv_path = os.path.join(OUT_DIR, "child_results.csv")
df.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(f"📁 Results saved to: {csv_path}")
print(f"✅ Successfully processed: {len(audio_files) - failed}/{len(audio_files)}")
print(f"❌ Failed files: {failed}")
print(f"{'='*60}")
