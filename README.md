# MFCC-Based Speaker Recognition System
*Python, NumPy, SciPy, Librosa, Streamlit, Signal Processing*

---

## 🎯 Overview
This project implements a **speaker recognition system** that differentiates individuals based on their voice using **MFCC (Mel-Frequency Cepstral Coefficients)** features.  
It includes a **Streamlit dashboard** for real-time visualization of audio signals, spectrograms, and MFCC heatmaps. This system can be used for voice authentication, speaker verification, or educational purposes in signal processing.

---

## ⚡ Key Features
- ✅ **MFCC Feature Extraction Pipeline:**  
  Extracts key audio features using pre-emphasis, framing, FFT, Mel filter bank, log-energy, and DCT to distinguish speakers accurately.
- 📊 **Interactive Streamlit Dashboard:**  
  Visualizes:
  - Waveforms of uploaded audio samples  
  - Spectrograms showing frequency over time  
  - MFCC heatmaps for feature comparison  
- ⚙️ **Parameter Optimization:**  
  Fine-tuned frame size, overlap, and number of Mel filters to improve recognition precision.  
- 🧩 **Flexible Input:**  
  Supports `.wav` audio files and allows comparison of multiple samples in real-time.

---

## 🚀 Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Anshul-Rajpoot/Python-projects.git
