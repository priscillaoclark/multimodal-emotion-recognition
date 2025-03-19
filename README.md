# Multimodal Emotion Classification Using Deep Learning 🎭🔊📖  

## Overview  
This project explores multimodal deep learning for emotion recognition, integrating speech and text features using Wav2Vec2, HuBERT, and RoBERTa, along with handcrafted acoustic features. We evaluate early vs. late fusion strategies and analyze how feature scaling and data partitioning impact model generalization. The study is based on RAVDESS and TESS datasets and achieves 98.53% accuracy on test data after applying optimal preprocessing techniques.  

📄 **Final Project Paper:** [15.773 HODL Final Project](https://github.com/priscillaoclark/multimodal-emotion-recognition/blob/main/15_773_Final_Project.pdf)  

## Key Findings  
✅ **Feature Scaling Matters** – Improves early fusion models dramatically (65.51% → 98.35%).  
✅ **Multimodal vs. Unimodal** – Text embeddings did not significantly boost accuracy in neutral-content datasets.  
✅ **Early vs. Late Fusion** – After scaling, early fusion performed as well as structured fusion models.  
✅ **Train-Test Leakage Risks** – Text-based leakage led to an overestimated accuracy of 97.88%, corrected to 93.89%.  

## Methodology  
1️⃣ **Data Preprocessing** – Extracted MFCCs, chroma, prosody features + speech-to-text transcription using Whisper.  
2️⃣ **Deep Learning Models** – Built early-fusion, late-fusion, and hybrid Transformer-based models**.  
3️⃣ **Feature Scaling** – Applied z-score normalization to prevent magnitude imbalances.  
4️⃣ **Evaluation** – Tested models under grouped train-test split to reduce text-based leakage bias.  

## Installation & Setup 🚀  
### Prerequisites  
- Python 3.8+  
- TensorFlow/Keras  
- PyTorch (for Wav2Vec2 & HuBERT)  
- Hugging Face Transformers (for RoBERTa)  
- Librosa (for audio processing)  
- OpenAI Whisper (for speech-to-text)  

### Installation  
```bash
git clone https://github.com/priscillaoclark/multimodal-emotion-recognition.git  
cd multimodal-emotion-recognition  
pip install -r requirements.txt  
