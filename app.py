import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Wonderful Tomato Sorting", layout="wide")

# --- CUSTOM CSS (WONDERFUL INDONESIA STYLE) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');
    
    /* Background & Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
    }

    /* Hero Section (Hitam Eksklusif) */
    .hero-section {
        background-color: #111111;
        padding: 60px 40px;
        border-radius: 0 0 40px 40px;
        color: white;
        text-align: left;
        margin-bottom: 40px;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 50px;
        line-height: 1.1;
        margin-bottom: 20px;
    }
    
    .hero-subtitle {
        font-size: 18px;
        opacity: 0.8;
        max-width: 600px;
        border-left: 3px solid #D4AF37; /* Aksen Emas */
        padding-left: 20px;
    }

    /* Card Section (Seperti Spotlight) */
    .spotlight-card {
        background-color: #F9F9F9;
        padding: 30px;
        border-radius: 25px;
        margin-bottom: 30px;
        border: 1px solid #EEEEEE;
    }

    /* Tombol Utama (Elegan & Tajam) */
    div.stButton > button:first-child {
        background-color: #111111 !important;
        color: white !important;
        border-radius: 5px;
        height: 60px;
        width: 100%;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: none;
        transition: 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #D4AF37 !important; /* Berubah jadi emas saat hover */
    }

    /* Panduan Tindakan (Horizontal Scroll Style) */
    .guide-item {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border-bottom: 4px solid #111;
    }
    
    /* Input Camera Customization */
    .stCameraInput { border-radius: 20px; }

    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO('best157.pt')

try:
    model = load_model()
except:
    st.error("Model tidak ditemukan.")

# --- HERO SECTION (SPOTLIGHT) ---
st.markdown("""
    <div class="hero-section">
        <p style="letter-spacing: 3px; color: #D4AF37; margin-bottom: 10px;">PROYEK SKRIPSI</p>
        <h1 class="hero-title">Karakteristik<br>Kualitas Tomat</h1>
        <p class="hero-subtitle">
            Temukan standar kualitas terbaik melalui teknologi AI. Identifikasi kerusakan secara akurat untuk manajemen stok yang lebih efisien di Toko Iwan.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- AREA UTAMA ---
col_input, col_tips = st.columns([2, 1])

with col_input:
    st.markdown("### Ambil Citra Tomat")
    metode = st.segmented_control("Metode", ["Kamera Langsung", "Galeri File"], default="Kamera Langsung")
    
    foto = None
    if metode == "Kamera Langsung":
        foto = st.camera_input("Scanner")
    else:
        foto = st.file_uploader("Pilih Berkas", type=["jpg", "png", "jpeg"])

with col_tips:
    st.markdown("""
        <div class="spotlight-card">
            <h4 style="margin-top:0;">Tips Spotlight</h4>
            <p style="font-size: 14px; color: #666;">
                Pastikan objek berada di tengah bingkai dengan pencahayaan yang cukup. AI akan menganalisis tekstur kulit untuk menentukan klasifikasi.
            </p>
            <hr>
            <p style="font-size: 12px; font-style: italic;">Standar Akurasi: YOLOv8m (94%) [cite: 26, 257, 262]</p>
        </div>
    """, unsafe_allow_html=True)

# --- PROSES DETEKSI ---
if foto is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("MULAI ANALISIS CITRA"):
        with st.spinner('Menganalisis karakteristik objek...'):
            gambar = Image.open(foto)
            img_array = np.array(gambar)
            results = model.predict(source=img_array, conf=0.25, iou=0.45)
            
            st.markdown("---")
            
            # Display Results
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(gambar, caption="Citra Default", use_container_width=True)
            with col_res2:
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Visualisasi AI", use_container_width=True)

            # --- STATISTIK ---
            counts = results[0].boxes.cls.tolist()
            if len(counts) > 0:
                st.markdown(f"#### Hasil: {len(counts)} Objek Teridentifikasi")
            else:
                st.warning("Objek tidak ditemukan.")

# --- PANDUAN (WONDERFUL STYLE) ---
st.markdown("### Panduan Klasifikasi")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""<div class="guide-item" style="border-color: #2E8B57;">
        <b style="color: #2E8B57;">SEHAT</b><br>
        <small>Pajang di etalase utama. Kualitas ekspor.</small>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="guide-item" style="border-color: #FFA500;">
        <b style="color: #FFA500;">RUSAK SEDANG</b><br>
        <small>Segera distribusikan atau jadikan bahan olahan.</small>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class="guide-item" style="border-color: #B22222;">
        <b style="color: #B22222;">RUSAK BERAT</b><br>
        <small>Pisahkan segera. Potensi penularan bakteri tinggi.</small>
    </div>""", unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #CCC; font-size: 11px; letter-spacing: 2px;'>WONDERFUL TOMATO • SKRIPSI 2026</p>", unsafe_allow_html=True)