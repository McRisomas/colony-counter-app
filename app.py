import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

# 1. Konfigurasi Halaman UI
st.set_page_config(page_title="Colony Counter App", page_icon="🧫")
st.title("🧫 Automatic Bacterial Colony Counter")
st.write("Upload a petri dish image to detect and count the colonies automatically")

# 2. Load Model YOLO (Gunakan cache agar model tidak dimuat ulang tiap ada interaksi)
@st.cache_resource
def load_colony_model():
    return YOLO('best(1).pt')

model = load_colony_model()

# 3. Sidebar Pengaturan
st.sidebar.header("Detection Settings")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold ", 
    min_value=0.05, 
    max_value=0.90, 
    value=0.17, 
    step=0.01,
    help=(
        "Adjust how strictly the model identifies colonies:\n\n"
        "• High Value (> 0.4): High precision. Reduces false detections "
        "(bubbles/glare), but faint colonies may be missed.\n\n"
        "• Low Value (< 0.2): High sensitivity. Captures faint colonies, "
        "but may mistakenly detect bubbles or artifacts as colonies."
    ),
)

# 4. Area Upload Gambar
uploaded_files = st.file_uploader("Choose a petri dish image ...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.info(f"Total {len(uploaded_files)} image is ready for analysis..")

    for uploaded_file in uploaded_files:
        # Tampilkan preview gambar yang diupload terlebih dahulu
        st.image(uploaded_file, caption=(f"Uploaded {uploaded_file.name}"), width=400)

    # Jalankan prediksi dengan YOLO
    if st.button("🚀 Count Colonies", type="primary"):
        with st.spinner("Counting colonies..."):
            for uploaded_file in uploaded_files:
                # Konversi file unggahan menjadi format numpy/array
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                results = model.predict(img_bgr, imgsz=1280, conf=conf_thresh)

                # Ambil hasil plot bounding box
                res_plotted = results[0].plot(labels=False, conf=False, line_width=2)
                total_cfu = len(results[0].boxes)

                # Menggunakan expander sehingga satu per satu gambar dapat dilihat dan ditampilkan jumla CFU
                with st.expander(f"📄 {uploaded_file.name} — Total: {total_cfu} CFU"):
                    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                    st.image(res_rgb, caption="Colony Count Result", use_container_width=True)
                    
        st.success("**Colonies Have Been Counted**")
