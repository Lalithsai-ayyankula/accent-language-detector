"""
Accent-Based Cuisine Recommender
Detects Indian regional accents from audio and recommends traditional dishes
"""

import gradio as gr
import torch
import torch.nn as nn
import os
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, HubertModel

# -------------------------------
# Configuration
# -------------------------------
MODEL_DIR = "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000

# Accent labels (6 Indian state accents)
LABEL_NAMES = ["andhra_pradesh", "gujrat", "jharkhand", "karnataka", "kerala", "tamil"]

# Cuisine recommendations for each state
CUISINE_MAP = {
    "andhra_pradesh": ["Hyderabadi Biryani", "Pesarattu", "Gongura Pachadi", "Pulihora", "Gutti Vankaya"],
    "gujrat": ["Dhokla", "Khandvi", "Thepla", "Undhiyu", "Fafda"],
    "jharkhand": ["Litti Chokha", "Dhuska", "Pitha", "Rugra", "Bamboo Shoot Curry"],
    "karnataka": ["Bisi Bele Bath", "Mysore Pak", "Ragi Mudde", "Masala Dosa", "Obbattu"],
    "kerala": ["Appam with Stew", "Puttu with Kadala", "Avial", "Fish Moilee", "Kerala Parotta"],
    "tamil": ["Masala Dosa", "Idli Sambar", "Chettinad Chicken", "Pongal", "Filter Coffee"]
}

print(f"✅ Loaded {len(LABEL_NAMES)} accent labels: {LABEL_NAMES}")

# -------------------------------
# HuBERT Classifier Architecture
# -------------------------------
class HubertClassifier(nn.Module):
    """Classifier for accent detection from HuBERT embeddings"""
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

# -------------------------------
# Load Models
# -------------------------------
print("🔹 Loading HuBERT models...")

try:
    # Load feature extractor and HuBERT backbone
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained(
        "facebook/hubert-base-ls960", 
        output_hidden_states=True
    ).to(DEVICE).eval()

    # Load trained classifier
    clf_path = os.path.join(MODEL_DIR, "best_hubert_clf.pth")
    classifier = HubertClassifier(embedding_dim=768, num_classes=6).to(DEVICE)

    checkpoint = torch.load(clf_path, map_location=DEVICE)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            classifier.load_state_dict(checkpoint['state_dict'])
        else:
            classifier.load_state_dict(checkpoint)
    else:
        classifier.load_state_dict(checkpoint)

    classifier.eval()
    print(f"✅ Models loaded successfully on {DEVICE}!")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise

# -------------------------------
# Helper Functions
# -------------------------------
def get_cuisine_recommendations(accent_label):
    """Get dish recommendations for a given accent"""
    return CUISINE_MAP.get(accent_label.lower(), ["No recommendations available"])

def predict_accent(audio_path):
    """Predict accent from audio file and return recommendations"""
    if audio_path is None:
        return "❌ No audio uploaded", [], None

    try:
        # Load and preprocess audio
        wav, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Validate audio length
        if len(wav) < 1600:  # Less than 0.1 seconds
            return "❌ Audio too short (minimum 0.1 seconds required)", [], None
        
        # Extract features
        inputs = feature_extractor(
            wav, 
            sampling_rate=SAMPLE_RATE, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Get HuBERT embeddings
        with torch.no_grad():
            outputs = hubert_model(**inputs)
            embeddings = outputs.hidden_states[-1].mean(dim=1)  # [1, 768]
            
            # Classify
            logits = classifier(embeddings)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(logits, dim=1).item()
            confidence = probs[0, pred_idx].item()

        # Get predictions
        accent_label = LABEL_NAMES[pred_idx]
        dishes = get_cuisine_recommendations(accent_label)
        
        # Format state name for display
        display_name = accent_label.replace("_", " ").title()
        
        # Format result
        result_text = f"🎯 **Detected State Accent:** {display_name}\n📊 **Confidence:** {confidence:.1%}"
        
        # Add confidence warning if low
        if confidence < 0.5:
            result_text += "\n\n⚠️ **Low confidence** - Consider using a clearer audio sample"
        
        # Create confidence breakdown with formatted names
        confidence_breakdown = {
            LABEL_NAMES[i].replace("_", " ").title(): f"{probs[0, i].item():.1%}" 
            for i in range(len(LABEL_NAMES))
        }
        
        return result_text, dishes, confidence_breakdown

    except Exception as e:
        return f"❌ Error processing audio: {str(e)}", [], None

# -------------------------------
# Gradio Interface
# -------------------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Accent Cuisine Recommender") as app:
    
    gr.Markdown(
        """
        # 🎤 Indian State Accent-Based Cuisine Recommender
        
        Upload an audio recording or speak into your microphone to detect your Indian state accent 
        and get personalized traditional dish recommendations!
        
        **Supported States:** Andhra Pradesh, Gujarat, Jharkhand, Karnataka, Kerala, Tamil Nadu
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                type="filepath", 
                label="🎙️ Upload or Record Audio",
                sources=["upload", "microphone"]
            )
            
            with gr.Row():
                predict_btn = gr.Button("🔍 Analyze Accent", variant="primary", size="lg")
                clear_btn = gr.Button("🔄 Clear", variant="secondary")
        
        with gr.Column(scale=1):
            accent_result = gr.Textbox(
                label="📍 Accent Detection Result",
                lines=3
            )
            
            dishes_output = gr.JSON(
                label="🍽️ Recommended Traditional Dishes"
            )
            
            confidence_output = gr.JSON(
                label="📊 Confidence Breakdown (All Accents)"
            )
    
    gr.Markdown(
        """
        ---
        ### 💡 Tips for Best Results:
        - **Audio Length:** Record at least 2-3 seconds of clear speech
        - **Audio Quality:** Use a quiet environment with minimal background noise
        - **Speaking:** Speak naturally in your native language or accent
        - **Microphone:** For best results, speak directly into the microphone
        
        ### 🔧 How It Works:
        1. Your audio is processed using HuBERT (Hidden-Unit BERT) model
        2. The model extracts acoustic features and detects accent patterns
        3. Based on the detected accent, traditional dishes from that region are recommended
        """
    )
    
    # Event handlers
    predict_btn.click(
        fn=predict_accent,
        inputs=audio_input,
        outputs=[accent_result, dishes_output, confidence_output]
    )
    
    clear_btn.click(
        fn=lambda: (None, "", [], None),
        outputs=[audio_input, accent_result, dishes_output, confidence_output]
    )
    
    # Auto-predict when audio changes
    audio_input.change(
        fn=predict_accent,
        inputs=audio_input,
        outputs=[accent_result, dishes_output, confidence_output]
    )

# -------------------------------
# Launch Application
# -------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Launching Accent Cuisine Recommender...")
    print("="*60 + "\n")
    
    app.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Local only
        server_port=7860,  # Default Gradio port
        show_error=True
    )