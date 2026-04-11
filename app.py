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
# Page Setup
# -------------------------
st.set_page_config(page_title="AI Car Damage Detection", layout="wide")
st.title("🚗 AI Car Damage Detection & Insurance Assistant")

# -------------------------
# Load YOLO Model
# -------------------------
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

# -------------------------
# Detect Damage Function
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
            label = f"{damage_type} ({round(conf,2)})"
            cv2.putText(img, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 0, 0), 2)

    return detections, img

# -------------------------
# AI Assessment
# -------------------------
def get_ai_assessment(detections):
    if len(detections) == 0:
        return "No damage detected."

    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        ai_results = []

        total_area = sum([d['Width'] * d['Height'] for d in detections])
        size_threshold = 50000

        for d in detections:
            damage_desc = f"{d['Damage Type']} ({d['Width']}x{d['Height']} px), confidence {d['Confidence']}"
            prompt = (
                f"A car image inspection detected the following damage: {damage_desc}\n\n"
                "Provide:\n"
                "1. Repair suggestion\n"
                "2. Estimated severity (minor/moderate/severe)\n"
                "3. Estimated repair time\n"
                "4. Estimated cost in INR\n"
                "5. Insurance claim possibility with reason.\n"
            )

            completion = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_completion_tokens=500
            )

            result_text = completion.choices[0].message.content

            if (d['Width'] * d['Height']) > size_threshold:
                result_text += "\n⚠️ Large damage → Consider SEVERE."

            ai_results.append(f"🔹 {d['Damage Type']}:\n{result_text}")

        if total_area > (size_threshold * len(detections)):
            ai_results.append(f"\n⚠️ Overall damage is large → SEVERE.")

        return "\n\n".join(ai_results)

    except Exception as e:
        st.warning(f"⚠️ AI error: {str(e)}")
        return "AI assessment failed."

# -------------------------
# PDF Report
# -------------------------
def generate_pdf(detections, ai_report):
    styles = getSampleStyleSheet()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    story = []

    story.append(Paragraph("Car Damage Assessment Report", styles["Title"]))
    story.append(Spacer(1, 20))

    if len(detections) == 0:
        story.append(Paragraph("No damages detected.", styles["Normal"]))
    else:
        data = [["Damage Type", "Confidence", "Bounding Box"]]
        for d in detections:
            data.append([d["Damage Type"], str(d["Confidence"]), str(d["Bounding Box"])])
        story.append(Table(data))

        story.append(Spacer(1, 20))
        story.append(Paragraph("AI Assessment", styles["Heading2"]))
        story.append(Paragraph(ai_report.replace("\n", "<br/>"), styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated On: {datetime.now()}", styles["Normal"]))

    pdf = SimpleDocTemplate(temp.name, pagesize=letter)
    pdf.build(story)

    return temp.name

# -------------------------
# Upload Image
# -------------------------
uploaded = st.file_uploader("Upload Car Image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    # ✅ Resize image (optional but recommended)
    image = image.resize((400, 300))

    # ✅ Center + Smaller Image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, caption="Uploaded Image", width=300)

    if st.button("🔍 Detect Damage"):
        model = load_model()
        detections, annotated = detect_damage(model, image)

        st.subheader("Detected Damage")

        # ✅ Center + Smaller Annotated Image
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.image(annotated, caption="Detected Damage Areas", width=400)

        if len(detections) == 0:
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
                st.download_button(
                    "📄 Download Report",
                    f,
                    file_name="car_damage_report.pdf"
                )
