import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Wonderful Tomato Sorting", layout="wide")

# --- CUSTOM CSS (WONDERFUL INDONESIA - ELDERLY FRIENDLY) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;600&display=swap');
    
    /* Font Global diperbesar untuk keterbacaan orang tua */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FFFFFF;
        font-size: 20px;
    }

    /* Hero Section (Hitam Eksklusif) */
    .hero-section {
        background-color: #111111;
        padding: 50px 40px;
        border-radius: 0 0 40px 40px;
        color: white;
        text-align: left;
        margin-bottom: 30px;
    }
    
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 60px;
        line-height: 1.1;
        margin-bottom: 20px;
    }
    
    /* Tips Pengambilan poto */
    .spotlight-text {
        font-size: 22px;
        color: #D4AF37; /* Warna Emas */
        max-width: 800px;
        border-left: 4px solid #D4AF37;
        padding-left: 20px;
        line-height: 1.5;
        font-weight: 600;
    }

    /* Tombol Utama Besar & Jelas */
    div.stButton > button:first-child {
        background-color: #111111 !important;
        color: white !important;
        border-radius: 10px;
        height: 80px;
        width: 100%;
        font-weight: 700;
        font-size: 24px;
        letter-spacing: 2px;
        text-transform: uppercase;
        border: 2px solid #D4AF37;
    }
    
    /* Kartu Panduan */
    .guide-item {
        background-color: white;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border-bottom: 6px solid #111;
        margin-bottom: 20px;
    }

    /* Input Styling */
    .stCameraInput { border-radius: 20px; }
    label { font-size: 24px !important; font-weight: bold !important; }

    /* Tabel Styling agar kontras */
    .stTable {
        background-color: #f9f9f9;
        border-radius: 10px;
    }

    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model():
    # Memuat model YOLOv8 yang telah dilatih
    return YOLO('best157.pt')

try:
    model = load_model()
except:
    st.error("Model tidak ditemukan di folder aplikasi.")

# --- HERO SECTION (SPOTLIGHT) ---
st.markdown("""
    <div class="hero-section">
        <p style="letter-spacing: 4px; color: #D4AF37; margin-bottom: 10px; font-weight: bold;">TOKO IWAN</p>
        <h1 class="hero-title">Periksa<br>Kerusakan Tomat</h1>
        <div class="spotlight-text">
            TIPS PENGAMBILAN FOTO: Foto buah tomat dari atas, pastikan buah tomat berada di cahaya yang cukup terang. 
            pastikan buah tomat dapat terfoto semua dari atas.
        </div>
    </div>
    """, unsafe_allow_html=True)

# --- AREA INPUT ---
st.markdown("### Ambil FotoTomat")

# Opsi ditukar: Galeri Foto HP menjadi urutan pertama dan default terpilih
metode = st.radio("Pilih Cara:", ("Galeri Foto HP", "Kamera Langsung"), horizontal=True)

foto = None
if metode == "Galeri Foto HP":
    foto = st.file_uploader("Pilih Berkas Gambar", type=["jpg", "png", "jpeg"])
else:
    foto = st.camera_input("Scanner AI")

# --- PROSES DETEKSI ---
if foto is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("MULAI PERIKSA"):
        # Parameter deteksi dikunci untuk stabilitas hasil pada dataset terbatas[cite: 2]
        CONF_LIMIT = 0.25
        IOU_LIMIT = 0.45

        with st.spinner('Sedang menganalisis citra...'):
            gambar = Image.open(foto)
            img_array = np.array(gambar)
            
            # Melakukan prediksi menggunakan model YOLOv8[cite: 1, 2]
            results = model.predict(source=img_array, conf=CONF_LIMIT, iou=IOU_LIMIT)
            
            st.markdown("---")
            
            # Menampilkan hasil deteksi secara visual
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.image(gambar, caption="Foto Asli", use_container_width=True)
            with col_res2:
                res_plotted = results[0].plot(conf=False)
                st.image(res_plotted, caption="Hasil Identifikasi AI", use_container_width=True)

            # --- BAGIAN STATISTIK (HANYA TABEL) ---
            st.markdown("---")
            st.markdown("### Rincian Kerusakan")
            
            counts = results[0].boxes.cls.tolist()
            names = results[0].names 
            
            if len(counts) > 0:
                # Mengolah data rincian[cite: 1, 2]
                df_counts = pd.DataFrame(counts, columns=['class_id'])
                df_counts['Kondisi'] = df_counts['class_id'].apply(lambda x: names[int(x)])
                rekap = df_counts['Kondisi'].value_counts().reset_index()
                rekap.columns = ['Kategori Kerusakan', 'Jumlah (Butir)']
                
                # Tampilan Header Ringkasan
                st.markdown(f"<div style='text-align:center; padding:15px; background:#111; color:#D4AF37; border-radius:15px; font-size:28px; font-weight:bold; margin-bottom:20px;'>TOTAL TERPERIKSA: {len(counts)} BUAH</div>", unsafe_allow_html=True)
                
                # Menampilkan tabel rincian data[cite: 1, 2]
                st.table(rekap) 
            else:
                st.warning("Tomat tidak terbaca jelas. Mohon dekati objek dan pastikan cahaya cukup.")

# --- PANDUAN TINDAKAN (TAMPIL PERMANEN) ---
st.markdown("---")
st.markdown("### Saran Tindakan")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""<div class="guide-item" style="border-color: #2E8B57;">
        <b style="color: #2E8B57; font-size: 26px;">TIDAK RUSAK</b><br>
        <span style="font-size: 18px;">Kondisi Bagus. Masukkan ke rak pajangan utama.</span>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown("""<div class="guide-item" style="border-color: #FFA500;">
        <b style="color: #FFA500; font-size: 26px;">KERUSAKAN SEDANG</b><br>
        <span style="font-size: 18px;">agar tidak menular, segera di pisahkan jika kerusakan sudah semakin parah.</span>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown("""<div class="guide-item" style="border-color: #B22222;">
        <b style="color: #B22222; font-size: 26px;">KERUSAKAN BERAT</b><br>
        <span style="font-size: 18px;">Rusak Parah. Pisahkan segera agar tidak semakin menular.</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<br><p style='text-align: center; color: #BBB; font-size: 14px; letter-spacing: 3px;'>SKRIPSI 2026</p>", unsafe_allow_html=True)
