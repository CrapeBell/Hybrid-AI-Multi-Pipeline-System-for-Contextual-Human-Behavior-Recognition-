# Hybrid-AI-Multi-Pipeline-System-for-Contextual-Human-Behavior-Recognition-
A real-time, context-aware safety and surveillance monitoring system that integrates concurrent computer vision pipelines and deep learning architectures to holistically analyze human physical actions and emotional states, issuing rapid distress alerts.
# System Architecture
Traditional vision systems often look at isolated actions or standalone facial expressions in a vacuum. This system overcomes contextual blindness by running multiple specialized neural networks concurrently to assess an individual’s physical and emotional state via two parallel analytical pipelines:
              ┌──────────────┐
              │ Webcam Input │
              └──────┬───────┘
                     │
     ┌───────────────┴───────────────┐
┌──────────────────┐           ┌───────────────────┐
│  Face Detection  │           │  Pose Estimation  │
└────────┬─────────┘           └─────────┬─────────┘
┌──────────────────┐           ┌───────────────────┐
│   Emotion CNN    │           │ Action Autoencoder│
│   (7 Classes)    │           │    (6 Classes)    │
└────────┬─────────┘           └─────────┬─────────┘
┌──────────────────┐           ┌───────────────────┐
│  Distress LSTM   │           │ Anomaly Detection │
│ (Sequence Logic) │           │ (Recon. Error)    │
└────────┬─────────┘           └─────────┬─────────┘
         │                               │
         └───────────────┬───────────────┘
              ┌──────────────────────┐
              │ Weighted Alert Engine│
              └──────────┬───────────┘
             ┌──────────────────────┐
             │ Web UI / Audio Alert │
             └──────────────────────┘

**Pipeline A (Emotion & Temporal Distress):** Processes facial crops through a custom **Convolutional Neural Network (CNN)** trained on the **FER2013** dataset to identify instantaneous emotional vectors. These spatial probabilities are fed sequentially into a **Long Short-Term Memory (LSTM)** network to analyze temporal distress patterns over time.

**Pipeline B (Spatio-Temporal Action & Anomaly):** Utilizes **MediaPipe Pose** to track skeletal keypoints, feeding joint embeddings into an **Autoencoder** to classify physical actions and compute Reconstruction Error (MSE) for zero-day anomaly detection.

# Features Implemented

| Category | Feature | Status | Technical Description |

| **Vision & DL** | Face Detection | ✅ | Localizes and tracks human faces with dynamic bounding boxes. |
| | Emotion Recognition | ✅ | Classifies 7 distinct emotional expressions using a spatial CNN. |
| | Distress Detection | ✅ | Evaluates sequence variations over time via LSTM networks. |
| | Action Recognition | ✅ | Maps and identifies 6 physical baseline activities. |
| | Anomaly Detection | ✅ | Flags unusual movements mathematically via Autoencoder reconstruction loss. |
| **Alerting** | Audio Alerts | ✅ | Instant low-latency multi-threaded audio beeps upon distress detection. |
| | Visual Alerts | ✅ | Persistent dynamic red banner warning indicators on the UI overlay. |
| **Interface** | Web Dashboard | ✅ | Real-time interactive UI streaming low-latency processed frames. |
| | Tuning Controls | ✅ | Live threshold configuration sliders to adjust model sensitivity dynamically. |

# Tech Stack

* **Backend:** Python, Flask, TensorFlow, Keras
* **Computer Vision:** OpenCV, MediaPipe
* **Frontend:** HTML5, CSS3, JavaScript (Real-time DOM manipulation)
* **Database:** SQLite3

# Performance Metrics

* **Emotion Model Accuracy:** 59% (Validated on competitive FER-2013 benchmark)
* **Distress Detection Accuracy:** 98% (Evaluated on contextual synthetic sequence dataset)
* **Processing Throughput:** 15–30 FPS (Optimized for standard webcams)
* **End-to-End Alert Latency:** $< 100\text{ms}$

# Getting Started & Installation

Prerequisites
Ensure you have **Python 3.8+** installed on your system.

### Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/CrapeBell/Hybrid-AI-Multi-Pipeline-System-for-Contextual-Human-Behavior-Recognition.git](https://github.com/CrapeBell/Hybrid-AI-Multi-Pipeline-System-for-Contextual-Human-Behavior-Recognition.git)
2. **Move into the project folder (Crucial step! You are now inside the cloned repository):**
   ```bash
   cd Hybrid-AI-Multi-Pipeline-System-for-Contextual-Human-Behavior-Recognition
3. **Install the dependencies (While remaining inside the main repository folder where requirements.txt is located):**
   ```bash
   pip install -r requirements.txt
4. **Navigate to the backend folder (Moving one level deeper into the project structure):**
   ```bash
   cd backend
5. **Run the script (Inside the backend folder where app_integrated.py lives):**
   ```bash
   python app_integrated.py
