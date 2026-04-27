import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Tomat", layout="centered")

# --- CUSTOM CSS (UI MOBILE MERAH & PERBAIKAN STYLING) ---
st.markdown("""
    <style>
    /* Mengubah background utama menjadi merah */
    .stApp {
        background-color: #A53A3A;
        color: white;
    }
    
    /* Mengubah warna teks agar terlihat jelas di background merah */
    h1, h2, h3, p, label, .stMarkdown {
        color: white !important;
    }

    /* Styling tombol 'Mulai Deteksi' (Warna Hijau) */
    div.stButton > button:first-child {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 25px;
        border: none;
        height: 3.5em;
        width: 100%;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
    }

    /* Input Kamera & File Uploader */
    .stCameraInput > label, .stFileUploader > label {
        color: white !important;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #7A2E2E !important;
    }
    
    /* Styling Tabel agar bersih */
    .styled-table {
        background-color: white;
        color: black;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best15.pt ada di folder yang sama dengan app.py
    return YOLO('best15.pt')

try:
    model = load_model()
except Exception as e:
    st.error("Model 'best15.pt' tidak ditemukan! Pastikan file model ada di folder skripsi.")

# --- SIDEBAR: TIPS PENGGUNAAN & PENGATURAN ---
with st.sidebar:
    st.markdown("### 💡 Tips Penggunaan")
    st.info("""
    1. **Pencahayaan**: Pastikan objek tomat terkena cahaya terang.
    2. **Jarak**: Jangan terlalu dekat atau terlalu jauh.
    3. **Sensitivitas**: Jika tomat tidak terdeteksi, turunkan 'Confidence' di bawah.
    """)
    
    st.markdown("---")
    st.header("⚙️ Pengaturan Model")
    conf_val = st.slider("Confidence (Sensitivitas)", 0.05, 1.0, 0.25, help="Semakin rendah semakin sensitif.")
    iou_val = st.slider("IoU (Pembersih Kotak)", 0.1, 1.0, 0.45, help="Mengatur tumpang tindih kotak deteksi.")

# --- TAMPILAN UTAMA ---
st.markdown("<h2 style='text-align: center; font-weight: bold;'>Klasifikasi Kerusakan Buah Tomat</h2>", unsafe_allow_html=True)

# Pilihan input gambar (seperti tombol di UI)
pilihan = st.radio("Pilih Sumber Gambar:", ("Ambil Foto (Kamera)", "Upload dari Galeri"), horizontal=True)

foto = None
if pilihan == "Ambil Foto (Kamera)":
    foto = st.camera_input("Ambil foto tomat secara langsung")
else:
    foto = st.file_uploader("Pilih foto dari galeri...", type=["jpg", "png", "jpeg"])

st.markdown("---")

# PROSES DETEKSI
if foto is not None:
    # Tombol Mulai (Warna Hijau sesuai gambar UI)
    if st.button("🚀 Mulai Deteksi"):
        with st.spinner('Sedang menganalisis kualitas tomat...'):
            gambar = Image.open(foto)
            img_array = np.array(gambar)
            
            # Jalankan YOLOv8
            results = model.predict(source=img_array, conf=conf_val, iou=iou_val)
            
            # Layout Hasil (Default vs Hasil)
            col_def, col_res = st.columns(2)
            
            with col_def:
                st.markdown("<p style='text-align: center; font-weight: bold;'>Default (Asli)</p>", unsafe_allow_html=True)
                st.image(gambar, use_container_width=True)
            
            with col_res:
                st.markdown("<p style='text-align: center; font-weight: bold;'>Hasil Deteksi</p>", unsafe_allow_html=True)
                res_plotted = results[0].plot()
                st.image(res_plotted, use_container_width=True)

            # --- STATISTIK HASIL ---
            st.markdown("### 📊 Ringkasan Deteksi")
            counts = results[0].boxes.cls.tolist()
            names = results[0].names
            
            if len(counts) > 0:
                df_counts = pd.DataFrame(counts, columns=['class_id'])
                df_counts['Kondisi Tomat'] = df_counts['class_id'].apply(lambda x: names[int(x)])
                rekap = df_counts['Kondisi Tomat'].value_counts().reset_index()
                rekap.columns = ['Kondisi', 'Jumlah']
                
                # Tampilkan Tabel dan Metrik Total
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.dataframe(rekap, use_container_width=True)
                with c2:
                    st.metric("Total Buah", len(counts))
            else:
                st.warning("Tomat tidak terdeteksi. Silakan coba foto kembali dengan sudut yang berbeda.")