import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime
from groq import Groq

# -------------------------
# Page Setup (FIXED)
# -------------------------
st.set_page_config(page_title="AI Car Damage Detection", layout="centered")
st.title("🚗 AI Car Damage Detection & Insurance Assistant")

# -------------------------
# 🔥 FORCE SMALL IMAGE USING CSS
# -------------------------
st.markdown("""
    <style>
    img {
        max-width: 300px !important;
        height: auto !important;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------
# Load YOLO Model
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

# -------------------------
# Detect Damage
# -------------------------
def detect_damage(model, image):
    img = np.array(image)
    results = model(img)
    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = map(int, box)
            damage_type = model.names[int(cls)]

            detections.append({
                "Damage Type": damage_type,
                "Confidence": round(float(conf), 2),
                "Bounding Box": [x1, y1, x2, y2],
                "Width": x2 - x1,
                "Height": y2 - y1
            })

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{damage_type} ({round(conf,2)})",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

    return detections, img

# -------------------------
# AI Assessment
# -------------------------
def get_ai_assessment(detections):
    if not detections:
        return "No damage detected."

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        ai_results = []

        total_area = sum(d['Width'] * d['Height'] for d in detections)
        size_threshold = 50000

        for d in detections:
            prompt = f"""
            Damage: {d['Damage Type']} ({d['Width']}x{d['Height']} px), confidence {d['Confidence']}

            Give:
            1. Repair suggestion
            2. Severity
            3. Repair time
            4. Cost in INR
            5. Insurance claim possibility
            """

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=500
            )

            text = response.choices[0].message.content

            if d['Width'] * d['Height'] > size_threshold:
                text += "\n⚠️ Large damage → SEVERE"

            ai_results.append(f"🔹 {d['Damage Type']}:\n{text}")

        if total_area > size_threshold * len(detections):
            ai_results.append("\n⚠️ Overall damage is SEVERE")

        return "\n\n".join(ai_results)

    except Exception as e:
        st.warning(f"AI Error: {e}")
        return "AI assessment failed."

# -------------------------
# PDF Report
# -------------------------
def generate_pdf(detections, ai_report):
    styles = getSampleStyleSheet()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    story = []

    story.append(Paragraph("Car Damage Report", styles["Title"]))
    story.append(Spacer(1, 20))

    if detections:
        data = [["Damage", "Confidence", "Box"]]
        for d in detections:
            data.append([d["Damage Type"], str(d["Confidence"]), str(d["Bounding Box"])])
        story.append(Table(data))

        story.append(Spacer(1, 20))
        story.append(Paragraph("AI Assessment", styles["Heading2"]))
        story.append(Paragraph(ai_report.replace("\n", "<br/>"), styles["Normal"]))
    else:
        story.append(Paragraph("No damage detected.", styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated: {datetime.now()}", styles["Normal"]))

    pdf = SimpleDocTemplate(temp.name, pagesize=letter)
    pdf.build(story)

    return temp.name

# -------------------------
# Upload Image
# -------------------------
uploaded = st.file_uploader("Upload Car Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    # Optional resize for performance
    image = image.resize((400, 300))

    # ✅ SMALL IMAGE (CSS controlled)
    st.image(image, caption="Uploaded Image")

    if st.button("🔍 Detect Damage"):
        model = load_model()
        detections, annotated = detect_damage(model, image)

        st.subheader("Detected Damage")

        # ✅ SMALL IMAGE (CSS controlled)
        st.image(annotated, caption="Detected Damage Areas")

        if not detections:
            st.success("No damage detected")
        else:
            df = pd.DataFrame(detections)
            st.dataframe(df)

            st.subheader("🤖 AI Assessment")
            with st.spinner("Analyzing..."):
                ai_report = get_ai_assessment(detections)

            st.write(ai_report)

            pdf_file = generate_pdf(detections, ai_report)
            with open(pdf_file, "rb") as f:
                st.download_button("📄 Download Report", f, "car_damage_report.pdf")
