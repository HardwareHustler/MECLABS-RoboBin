🗑️ Smart Waste Segregation Bin (MECLabs Project)
📌 Overview

This project aims to build a Smart Waste Segregation Bin that automatically classifies waste into biodegradable and non-biodegradable categories using computer vision and deep learning.

The system is designed with social relevance in mind, targeting applications in:

Hospitals

Public spaces

Elderly care facilities

Assisted-living environments

The goal is to reduce manual waste segregation, improve hygiene, and support sustainable waste management practices.

🎯 Problem Statement

Improper waste segregation leads to:

Increased health risks

Inefficient recycling

Higher landfill usage

Manual segregation is often inconvenient or inaccessible for elderly and specially-abled individuals. This project addresses that gap using automation and AI.

💡 Proposed Solution

The bin uses a camera-based AI system to identify the type of waste dropped into it and automatically directs the waste into the appropriate compartment.

Core idea:

Waste is dropped into the bin

Camera captures an image

AI model classifies the waste

Mechanical mechanism diverts waste accordingly

🧠 System Architecture
Hardware

Jetson Nano – AI inference and image processing

Arduino (UNO / Nano) – Motor and actuator control

Camera Module – Waste image capture

High-torque DC motors – Movement / internal mechanisms

Servo motors – Lid and segregation flaps

Motor driver module – Safe motor control

Software

Python

PyTorch

OpenCV

Arduino IDE (C/C++)

🧪 AI Model Details

Model: ResNet-18 (Transfer Learning)

Framework: PyTorch

Datasets Used:

TrashNet

Kaggle Waste Classification Dataset

Classes:

Cardboard

Glass

Metal

Paper

Plastic

Trash

The model is trained on resized images and achieves ~80% validation accuracy during initial experimentation.

📂 Project Structure
smart_waste_ai/
│
├── dataset/
│   ├── train/
│   └── val/
│
├── train.py              # Model training script
├── infer.py              # Inference script
├── split_dataset.py      # Dataset splitting utility
├── requirements.txt      # Python dependencies
├── waste_classifier.pth  # Trained model weights
└── README.md

🚀 Current Status

✅ Dataset preparation
✅ Model training and validation
✅ Inference pipeline
🛠️ Hardware integration (in progress)
🛠️ Mechanical segregation mechanism (planned)
