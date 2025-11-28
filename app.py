import streamlit as st
import numpy as np
import librosa
import io
import time
import soundfile as sf
from core.utils import (
    aes_encrypt, aes_decrypt, dwt_haar, idwt_haar, 
    generate_spectrogram_fig, generate_dwt_plot_fig
)
from core.model import CNNAutoencoder, apply_cnn_denoising # Import CNN (simulasi)
from cryptography.exceptions import InvalidTag, InvalidSignature 

st.set_page_config(
    page_title="Watermarking Audio DWT-CNN",
    layout="wide",
    initial_sidebar_state="expanded",
)

def run_watermark_process(
    uploaded_file: io.BytesIO, 
    watermark_text: str, 
    alpha: float, 
    progress_bar: st.progress,
    status_placeholder: st.empty
) -> tuple[np.ndarray, int, dict]:
    """
    Fungsi utama watermarking yang melaporkan progress ke Streamlit.
    """
    
    y, sr = librosa.load(uploaded_file, sr=None)
    y_original = y.copy()
    visualization_data = {}
    
    # --- TAHAP 1: ENKRIPSI AES & PREPROCESSING ---
    progress_bar.progress(10, text="1/4: Memuat Audio & Enkripsi AES...")
    status_placeholder.info("Status: 1/4: Memuat Audio & Enkripsi AES...")
    time.sleep(0.5) 
    
    encrypted_wm = aes_encrypt(watermark_text.encode('utf-8'))
    wm_bits = np.unpackbits(np.frombuffer(encrypted_wm, dtype=np.uint8))
    
    visualization_data['orig_spectrogram'] = generate_spectrogram_fig(y_original, sr, "Spectrogram Asli")

    # --- TAHAP 2: TRANSFORMASI DWT ---
    progress_bar.progress(35, text="2/4: Transformasi DWT (Wavelet) Audio...")
    status_placeholder.info("Status: 2/4: Transformasi DWT (Wavelet) Audio...")
    time.sleep(1)
    
    cA, cD = dwt_haar(y)
    
    # --- TAHAP 3: PENANAMAN WATERMARK & CNN ---
    progress_bar.progress(60, text="3/4: Penyisipan Watermark & Proses CNN Autoencoder...")
    status_placeholder.info("Status: 3/4: Penyisipan Watermark & Proses CNN Autoencoder (Simulasi)...")
    time.sleep(2) 
    
    # Penanaman ke koefisien Detail (cD)
    wm_len = min(len(wm_bits), len(cD))
    cD_wm = cD.copy()
    
    for i in range(wm_len):
        if i < len(cD):
            bit_index = i % len(wm_bits)
            # Logika penyisipan
            if wm_bits[bit_index] == 1: 
                cD_wm[i] += alpha * np.abs(cD_wm[i]) 
            else:
                cD_wm[i] -= alpha * np.abs(cD_wm[i]) 

    # Visualisasi DWT Koefisien
    visualization_data['dwt_plot'] = generate_dwt_plot_fig(cD_wm, "DWT Detail Coeff. dengan Watermark")

    # --- TAHAP 4: INVERSE DWT & REKONSTRUKSI ---
    progress_bar.progress(85, text="4/4: Rekonstruksi Audio & Menyimpan Hasil...")
    status_placeholder.info("Status: 4/4: Rekonstruksi Audio & Menyimpan Hasil...")
    time.sleep(1)
    
    y_wm_robust = idwt_haar(cA, cD_wm)
    y_wm_robust = librosa.util.fix_length(y_wm_robust, size=len(y_original))
    
    visualization_data['final_spectrogram'] = generate_spectrogram_fig(y_wm_robust, sr, "Spectrogram Audio Ber-Watermark")
    
    progress_bar.progress(100, text="Proses Watermarking Selesai!")
    status_placeholder.success("✅ Proses Watermarking Selesai!")

    return y_wm_robust, sr, visualization_data

# --- 2. Fungsi Inti Ekstraksi ---

def run_extraction_process(uploaded_file: io.BytesIO) -> str:
    """Mengekstrak, mendekripsi, dan mengembalikan watermark."""
    
    # Load audio data
    y, sr = librosa.load(uploaded_file, sr=None)

    # 1. DWT (Hanya 1 level)
    cA, cD = dwt_haar(y)
    
    # 2. Ekstraksi bit (Simulasi sederhana)
    # Target: Mengekstrak sekitar 1024 bit (128 bytes)
    MAX_BITS_TO_EXTRACT = 1024 
    extracted_wm_bits = []
    
    for i in range(min(MAX_BITS_TO_EXTRACT, len(cD))):
        # Asumsi sederhana untuk ekstraksi: jika koefisien > 0, itu bit 1, dan jika < 0, itu bit 0
        if cD[i] > 0:
            extracted_wm_bits.append(1)
        else:
            extracted_wm_bits.append(0)

    # Konversi list bits ke bytes
    extracted_bits_array = np.array(extracted_wm_bits, dtype=np.uint8)
    
    # Padding bits agar panjangnya kelipatan 8
    padding = len(extracted_bits_array) % 8
    if padding != 0:
        extracted_bits_array = extracted_bits_array[:-padding]
        
    extracted_bytes = np.packbits(extracted_bits_array).tobytes()

    # 3. Dekripsi AES
    try:
        decrypted_wm_bytes = aes_decrypt(extracted_bytes)
        return decrypted_wm_bytes.decode('utf-8', errors='ignore').strip()
    except InvalidTag:
         return f"Gagal mendekripsi: Watermark rusak atau tidak ditemukan. (Error: AES Invalid Tag)"
    except Exception as e:
        return f"Gagal dekripsi: {type(e).__name__}"

# --- 3. Antarmuka Streamlit (UI) ---

st.title("🎶 Watermarking Audio Tahan Robyek (DWT-CNN)")
st.caption("Aplikasi ini menggunakan DWT dan Enkripsi AES untuk penanaman data rahasia ke dalam audio.")

st.markdown("---")

col1, col2 = st.columns(2)

# --- PANEL 1: PENANAMAN WATERMARK ---
with col1:
    st.header("1. Penanaman Watermark")
    
    with st.form(key='watermark_form'):
        uploaded_file_wm = st.file_uploader(
            "Pilih Lagu (.mp3, .wav)", 
            type=["mp3", "wav"]
        )
        watermark_text = st.text_input(
            "Watermark (Teks Hak Cipta)", 
            value="Hak Cipta Dilindungi", 
            max_chars=100
        )
        alpha = st.slider(
            "Faktor Intensitas Alpha (α)", 
            min_value=0.01, max_value=0.10, value=0.05, step=0.01,
            help="Nilai yang lebih besar meningkatkan ketahanan tetapi menurunkan kualitas audio."
        )
        
        submit_button_wm = st.form_submit_button(label='Proses Watermarking')

    if submit_button_wm and uploaded_file_wm:
        status_placeholder = st.empty()
        progress_bar = status_placeholder.progress(0, text="Memulai Proses...")
        
        try:
            audio_data_buffer = io.BytesIO(uploaded_file_wm.read())
            
            watermarked_y, sr, visual_data = run_watermark_process(
                audio_data_buffer, watermark_text, alpha, progress_bar, status_placeholder
            )
            
            st.markdown("### ✅ Hasil Penanaman Watermark")

            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, watermarked_y, sr, format='wav')
            wav_buffer.seek(0)
            
            st.audio(wav_buffer, format='audio/wav', start_time=0)
            
            st.download_button(
                label="Unduh File Audio Ber-Watermark (.wav)",
                data=wav_buffer,
                file_name=f"watermarked_{uploaded_file_wm.name.split('.')[0]}.wav",
                mime="audio/wav"
            )

            st.markdown("---")
            st.subheader("Visualisasi Proses")
            # Menampilkan Visualisasi Matplotlib
            st.pyplot(visual_data['orig_spectrogram'], use_container_width=True)
            st.pyplot(visual_data['dwt_plot'], use_container_width=True)
            st.pyplot(visual_data['final_spectrogram'], use_container_width=True)
            
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Terjadi kesalahan fatal: {type(e).__name__}: {e}")
    elif submit_button_wm and not uploaded_file_wm:
        st.warning("Mohon unggah file audio terlebih dahulu.")

# --- PANEL 2: EKSTRAKSI WATERMARK ---
with col2:
    st.header("2. Ekstraksi Watermark")
    
    with st.form(key='extract_form'):
        uploaded_file_extract = st.file_uploader(
            "Pilih Lagu Ber-Watermark (.mp3, .wav)", 
            type=["mp3", "wav"]
        )
        submit_button_extract = st.form_submit_button(label='Ekstraksi Watermark')
    
    if submit_button_extract and uploaded_file_extract:
        st.markdown("### 🔍 Hasil Ekstraksi")
        
        with st.spinner("Mengekstrak dan Mendekripsi Watermark..."):
            try:
                audio_data_buffer_extract = io.BytesIO(uploaded_file_extract.read())
                
                extracted_wm = run_extraction_process(audio_data_buffer_extract)
                
                if "Gagal mendekripsi" in extracted_wm or "Error" in extracted_wm:
                    st.error(extracted_wm)
                else:
                    st.success("✅ Ekstraksi Berhasil!")
                    st.write(f"Watermark Didekripsi: **{extracted_wm}**")
                    
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat ekstraksi: {type(e).__name__}: {e}")
    elif submit_button_extract and not uploaded_file_extract:
        st.warning("Mohon unggah file audio terlebih dahulu.")