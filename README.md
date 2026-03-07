🚗 AI Car Damage Detection & Insurance Assistant

An AI-powered web application that detects multiple types of car damages from uploaded images and provides repair suggestions, severity assessment, estimated repair time & cost, and insurance claim guidance. Built with YOLOv8, Streamlit, OpenCV, and Groq AI.

Features

Detects multiple damages in a single car image.

Supports 22 damage classes (e.g., dents, scratches, bumper damage, windscreen damage, etc.).

Provides AI-generated repair suggestions, severity assessment, estimated repair time & cost, and insurance claim possibilities.

Generates a downloadable PDF report with damage details and AI assessment.

Rule-based check for large damage areas to flag severe cases.

User-friendly Streamlit interface with annotated images.

Demo



Tech Stack

Python 3.12

Streamlit – Frontend interface

YOLOv8 (Ultralytics) – Object detection model for car damage detection

OpenCV – Image annotation

Groq API (LLaMA 3) – AI-generated repair and insurance assessment

ReportLab – Generate PDF reports

Pandas & NumPy – Data handling

Dataset

Trained on a Roboflow dataset with 22 car damage classes:
['Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent', 'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent', 'doorouter-scratch', 'fender-dent', 'front-bumper-dent', 'front-bumper-scratch', 'medium-Bodypanel-Dent', 'paint-chip', 'paint-trace', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'rear-bumper-scratch', 'roof-dent']


Usage

Run the Streamlit app:

streamlit run app.py

Upload a car image (jpg, jpeg, png).

Click "🔍 Detect Damage".

View:

Annotated image with detected damages

AI-generated repair and insurance assessment

Downloadable PDF report
