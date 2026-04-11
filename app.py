import streamlit as st
from PIL import Image
import cv2
import numpy as np
from datetime import datetime

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="AI Car Damage Detection", layout="wide")

st.title("🚗 AI Car Damage Detection System")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Create 2 columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(image, use_column_width=True)

    if st.button("🔍 Analyze Image"):

        # Convert to OpenCV format
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        output = img.copy()

        # -----------------------------
        # Dummy Detection (Replace with YOLO later)
        # -----------------------------
        # Scratch box
        cv2.rectangle(output, (150, 80), (350, 230), (0, 255, 0), 2)
        cv2.putText(output, "Scratch ($120)", (150, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Dent box
        cv2.rectangle(output, (300, 280), (520, 450), (0, 165, 255), 2)
        cv2.putText(output, "Dent ($450)", (300, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # Convert back to RGB
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

        # -----------------------------
        # Show Output Image
        # -----------------------------
        with col2:
            st.subheader("🧠 Analyzed Result")
            st.image(output, use_column_width=True)

        # -----------------------------
        # Summary Table
        # -----------------------------
        st.markdown("### 📊 Analysis Summary")

        st.table({
            "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Image Width": [output.shape[1]],
            "Image Height": [output.shape[0]],
            "Total Damages": [2],
            "Damage Area %": [9]
        })

        # -----------------------------
        # Suggestions
        # -----------------------------
        st.markdown("## 🛠 Detected Damage & Suggestions")

        st.success("✅ Dent detected → Estimated repair cost: $450")
        st.success("✅ Paint scratches detected → Estimated cost: $120")
        st.warning("⚠ Interior may be exposed → Check for dust/water damage")

        # -----------------------------
        # CSV Download
        # -----------------------------
        import pandas as pd

        df = pd.DataFrame({
            "Damage Type": ["Dent", "Scratch"],
            "Estimated Cost": ["$450", "$120"]
        })

        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇ Download CSV Report",
            data=csv,
            file_name="damage_report.csv",
            mime='text/csv'
        )

        # -----------------------------
        # Final Status
        # -----------------------------
        st.success("✅ Auto-Approve: Repair ticket ready (demo)")
