import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from datetime import datetime

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train8/weights/best.pt")

# -------------------------
# DAMAGE ANALYSIS
# -------------------------
def damage_analysis(detections):

    repair_data = []
    total_cost = 0

    for d in detections:
        damage = d["Damage Type"].lower()

        if "scratch" in damage:
            cost = 1500
            suggestion = "Polishing or repaint required"

        elif "dent" in damage:
            cost = 3000
            suggestion = "Dent removal and repaint"

        elif "bumper" in damage or "crack" in damage:
            cost = 5000
            suggestion = "Part replacement or repair"

        elif "glass" in damage or "headlight" in damage:
            cost = 4000
            suggestion = "Replace damaged part"

        else:
            cost = 2000
            suggestion = "General inspection required"

        total_cost += cost
        repair_data.append([damage, suggestion, f"₹{cost}"])

    insurance = "Eligible for Insurance Claim" if total_cost > 4000 else "No Insurance Required"

    return repair_data, total_cost, insurance

# -------------------------
# DETECTION (BALANCED)
# -------------------------
def detect_damage(model, image):

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    detections = []
    annotated = img.copy()

    results = model.predict(
        img,
        conf=0.4,   # 🔥 balanced
        iou=0.5,
        max_det=10
    )

    for r in results:
        if r.boxes is None:
            continue

        for box, conf, cls in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy()
        ):

            if conf < 0.4:
                continue

            x1, y1, x2, y2 = box
            area = (x2 - x1) * (y2 - y1)

            if area < 600:
                continue

            label = model.names[int(cls)]

            detections.append({
                "Damage Type": label,
                "Confidence": round(float(conf), 2),
                "Bounding Box": box.tolist()
            })

    # 🔥 REMOVE DUPLICATES
    final_detections = []
    seen = set()

    for d in detections:
        key = (d["Damage Type"], tuple(map(int, d["Bounding Box"])))
        if key not in seen:
            seen.add(key)
            final_detections.append(d)

    # DRAW BOXES
    for d in final_detections:
        x1, y1, x2, y2 = map(int, d["Bounding Box"])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(
            annotated,
            f"{d['Damage Type']} ({d['Confidence']})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return final_detections, annotated

# -------------------------
# PDF GENERATION
# -------------------------
def generate_pdf(detections, repair_data, total_cost, insurance):

    styles = getSampleStyleSheet()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    story = []

    story.append(Paragraph("Car Damage Report", styles["Title"]))
    story.append(Spacer(1, 20))

    if detections:
        data = [["Damage Type", "Confidence"]]
        for d in detections:
            data.append([d["Damage Type"], str(d["Confidence"])])

        story.append(Paragraph("Detected Damages", styles["Heading2"]))
        story.append(Table(data))
        story.append(Spacer(1, 20))

        repair_table = [["Damage", "Suggestion", "Cost"]] + repair_data
        story.append(Paragraph("Repair Suggestions", styles["Heading2"]))
        story.append(Table(repair_table))
        story.append(Spacer(1, 20))

        story.append(Paragraph(f"Total Cost: ₹{total_cost}", styles["Normal"]))
        story.append(Paragraph(f"Insurance: {insurance}", styles["Normal"]))
    else:
        story.append(Paragraph("No damage detected.", styles["Normal"]))

    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Generated On: {datetime.now()}", styles["Normal"]))

    pdf = SimpleDocTemplate(temp.name, pagesize=letter)
    pdf.build(story)

    return temp.name

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Car Damage Detection", layout="wide")
st.title("🚗 AI Damage Detection System")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("🔍 Detect Damage"):

        model = load_model()
        detections, annotated = detect_damage(model, image)

        st.image(annotated, caption="Detection Output")

        # ✅ FINAL OUTPUT LOGIC
        if len(detections) == 0:
            st.error("❌ No damage detected")

        elif len(detections) == 1:
            st.success("✅ 1 damage detected")

        else:
            st.success(f"✅ {len(detections)} damages detected")

        if detections:
            df = pd.DataFrame(detections)
            st.dataframe(df)

            repair_data, total_cost, insurance = damage_analysis(detections)

            st.subheader("🔧 Repair Suggestions")
            st.table(repair_data)

            st.subheader("💰 Total Cost")
            st.write(f"₹{total_cost}")

            st.subheader("🛡 Insurance Decision")
            st.write(insurance)
        else:
            repair_data, total_cost, insurance = [], 0, "No Damage"

        # PDF DOWNLOAD
        pdf_file = generate_pdf(detections, repair_data, total_cost, insurance)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name="damage_report.pdf"
            )
