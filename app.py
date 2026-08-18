import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2

# 1. Konfigurasi Halaman UI
st.set_page_config(page_title="Colony Counter App", page_icon="🧫")
st.title("🧫 Automatic Bacterial Colony Counter")
st.write("Unggah foto cawan petri untuk menghitung jumlah koloni secara otomatis.")

# 2. Load Model YOLO (Gunakan cache agar model tidak dimuat ulang tiap ada interaksi)
@st.cache_resource
def load_colony_model():
    return YOLO('best(1).pt')

model = load_colony_model()

# 3. Sidebar Pengaturan
st.sidebar.header("Pengaturan Deteksi")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.05, 
    max_value=0.90, 
    value=0.17, 
    step=0.01
)

# 4. Area Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar cawan petri...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Konversi file unggahan menjadi format numpy/array
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Jalankan prediksi dengan YOLO
    with st.spinner("Menghitung koloni..."):
        results = model.predict(img_bgr, imgsz=1280, conf=conf_thresh)
        
        # Ambil hasil plot bounding box
        res_plotted = results[0].plot(labels=False, conf=False, line_width=2)
        total_cfu = len(results[0].boxes)

    # Tampilkan Hasil
    res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    st.image(res_rgb, caption="Hasil Deteksi Koloni", use_container_width=True)
    st.success(f"**Total Koloni Terdeteksi:** {total_cfu} CFU")