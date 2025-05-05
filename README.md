# NotUrFace-AI: Deepfake Detection Model

NotUrFace-AI is a deepfake detection model designed to classify video content as real or fake. Leveraging **TensorFlow** for development, it processes video frames and applies advanced machine learning techniques to identify synthetic or manipulated media.<br><br>
Click the link to try the deployed project:
https://huggingface.co/spaces/sarvansh/NotUrFace-AI <br><br>
**Datasets used for training the model: FaceForensics++ and CelebDF**

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Datasets](#datasets)
4. [Model Architecture](#model-architecture)
5. [Setup Instructions](#setup-instructions)
6. [How to Use](#how-to-use)
7. [Training and Evaluation](#training-and-evaluation)
8. [Results](#results)
<br>(some data in the readme needs to be updated which will be completed soon..)
---

## Project Overview
NotUrFace-AI is aimed at combating the proliferation of deepfake videos using machine learning and computer vision. The model analyzes video files, extracts frames, and identifies manipulations using features learned from real and fake datasets.

It is particularly useful for:
- Social media content moderation
- Digital forensics
- Research in deepfake detection and AI ethics

---
## Features
- **Real-time video classification**: Detects whether a video is real or synthetic.
- **Frame-based analysis**: Processes video content frame by frame to improve accuracy.
- **Preprocessing pipeline**: Includes face cropping, frame skipping, and augmentations.
- **TensorFlow-based development**: Efficient training and inference using GPU acceleration.

---

## Datasets
The model was trained and evaluated on:
- **FaceForensics++ (FF++) Dataset**: Benchmark dataset for deepfake detection.
- **Celeb-DF Dataset**: Dataset with challenging real and deepfake videos.

Each video is preprocessed to extract **30 frames** for classification purposes.

---

## Model Architecture
The NotUrFace-AI architecture includes:
1. **Feature Extraction**: XceptionNet (pretrained) extracts spatial features from input frames.
2. **Temporal Analysis**: Long Short-Term Memory (LSTM) layers analyze temporal dependencies between frames.
3. **Dense Layers**: Fully connected layers for final classification.

The model predicts probabilities for two classes:
- **Real** (Label: 0)
- **Fake** (Label: 1)

---

## Setup Instructions
### Prerequisites
- check requirements.txt
- Google Colab/ Kaggle/ LightningAI (optional for training with GPUs)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/NotUrFace-AI.git
   cd NotUrFace-AI
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure GPU acceleration is enabled (if available).

---

## How to Use

---

## Results
| Metric          | Value       |
|-----------------|-------------|
| Training Accuracy | 98.44%       |
| Validation Accuracy | 97.05%     |
| Test Accuracy   | 95.93%         |


---

**Author**: Sarvansh Pachori  
**Project Name**: NotUrFace-AI
