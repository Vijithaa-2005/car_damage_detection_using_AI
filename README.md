# 🚗 AI Car Damage Detection & Insurance Assistant

- **AI-powered web application** that detects multiple car damages
- Provides:
  - **Repair suggestions**
  - **Severity assessment**
  - **Estimated repair time & cost**
  - **Insurance claim guidance**
- Built with **YOLOv8, Streamlit, OpenCV, and Groq AI**

# ✨ Features

- Detects **multiple damages** in a single car image
- Supports **22 damage classes** (dents, scratches, bumper damage, windscreen damage)
- Provides **AI-generated repair suggestions, severity, estimated repair time & cost, and insurance claim feasibility**
- Generates **downloadable PDF reports**
- Includes **rule-based severity check** for large damage areas
- **User-friendly Streamlit interface** with annotated images

#  Tech Stack 

- Python 3.12 – Core programming language

- Streamlit – Interactive web interface

- YOLOv8 (Ultralytics) – Object detection for car damages

- OpenCV – Image processing and annotation

- Groq API (LLaMA 3) – AI-based repair and insurance assessment

- ReportLab – PDF report generation

- Pandas & NumPy – Data handling and processing

#  📊 Dataset 

Trained on 22 damage classes

['Front-Windscreen-Damage', 'Headlight-Damage', 'Major-Rear-Bumper-Dent', 'Rear-windscreen-Damage', 'RunningBoard-Dent', 'Sidemirror-Damage', 'Signlight-Damage', 'Taillight-Damage', 'bonnet-dent', 'doorouter-dent', 'doorouter-scratch', 'fender-dent', 'front-bumper-dent', 'front-bumper-scratch', 'medium-Bodypanel-Dent', 'paint-chip', 'paint-trace', 'pillar-dent', 'quaterpanel-dent', 'rear-bumper-dent', 'rear-bumper-scratch', 'roof-dent']

#  📂 Project Structure 

car-damage-ai/
├─ app.py                  # Main Streamlit app
├─ requirements.txt        # Python dependencies
├─ runs/detect/train8/     # YOLOv8 trained weights folder
├─ assets/                 # Optional demo images/screenshots

#  💡 Future Improvements 

Combine all detected damages in a single AI prompt for more accurate overall severity

Add real-time webcam damage detection

Integrate insurance APIs to generate instant claim estimates

Add confidence threshold sliders in Streamlit for low-confidence detection filterin



