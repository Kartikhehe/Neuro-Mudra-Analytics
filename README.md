# 🧠 Neuro Mudra Analytics  
**Mentor:** Prof. Rajesh M. Hegde, IIT Kanpur  
**Objective:** To classify diverse *Yogic Mudras* using EEG and EMG biosignals, bridging neuroscience, signal processing, and machine learning.

---

## 📂 [**Project Presentation**](https://drive.google.com/drive/folders/1WXu2cwPSaPsVOqpJqYN8k49scG7Vc7RL?usp=sharing)  
Explore research insights, results, and visuals in our detailed presentation.

---

## 🔬 **Project Summary**  
This is a **first-of-its-kind study** to classify 10 distinct **yogic hand mudras** using **self-recorded EEG and EMG biosignals**. The project integrates neuroscience, bio-signal analytics, and machine learning to study the neurological and physiological effects of meditation practices.

---

## 🎯 **Objectives**
- Develop an **accurate classifier** for 10 yogic mudras using biosignal data.  
- Study **neurological and muscular impacts** of meditation mudras via EEG & EMG.  
- Validate **mudra recognition using machine learning and domain adaptation**.

---

## 🧪 **Approach & Methodology**

### 1️⃣ Data Collection
- Biosignals recorded from **4 participants** performing **10 meditation mudras**.  
- Data collected from **3 channels**, repeated trials per mudra.  
- **Simultaneous EEG and EMG capture** at suitable sampling rates.

### 2️⃣ Data Preprocessing
- **Baseline correction** and **normalization**  
- **Epoch segmentation** into time windows  
- **Butterworth/Bandpass filtering** to remove noise

### 3️⃣ Feature Extraction
- Computed **10+ statistical & spectral features** per channel:  
  - *Mean, Variance, RMS, Entropy, Zero-Crossings*  
  - *Delta to Gamma band powers* via **Welch’s method**

---

## 🧠 **Modeling Techniques**

### 🔁 LSTM & Deep Learning
- Trained **LSTM** models on raw EEG/EMG time-series  
- Implemented **CNN-LSTM** hybrids and **CRNN** architectures  
- Tuned hyperparameters, mitigated **overfitting**

### 🌐 Domain-Adversarial Neural Network (DANN)
- Applied **adversarial training** to reduce domain shift  
- Learned **domain-invariant features** for better generalization  
- **Accuracy:** *81.82%* (cross-participant)

### 🌲 Traditional ML on Extracted Features
- Models used:
  - **Random Forest** – *95% accuracy*
  - **Decision Tree** – *92% accuracy*
  - **SVM**, **Logistic Regression**
- Trained on statistical + spectral features per channel

---

## 📊 **Results**

| Model             | Accuracy   |
|------------------|------------|
| **Random Forest** | **95%**     |
| **Decision Tree** | **92%**     |
| **DANN**          | **81.82%**  |

- **Correlation matrices** plotted to analyze EEG–EMG inter-channel relationships  
- Verified **model robustness** and generalization capabilities  

---

## 🌱 **Impact**
- Developed a **multi-channel biosignal dataset** for Yogic Mudras  
- Demonstrated potential for **biosignal-based gesture recognition**  
- **Paved the way for neuro-yogic computing research**

---
