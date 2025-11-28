
# core/utils.py

import numpy as np
import librosa
import pywt # PyWavelets untuk DWT
import hashlib
import matplotlib.pyplot as plt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding

KEY = hashlib.sha256("Aplikasi Watermarking Audio".encode('utf-8')).digest()[:32] # 256-bit key
IV = hashlib.md5("Inisialisasi Vektor".encode('utf-8')).digest()[:16] # 128-bit IV

def aes_encrypt(data: bytes) -> bytes:
    """Mengenkripsi data menggunakan AES-256 (dengan PKCS7 Padding)."""
    cipher = Cipher(algorithms.AES(KEY), modes.CBC(IV), backend=default_backend())
    encryptor = cipher.encryptor()
    
    # PKCS7 Padding
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(data) + padder.finalize()
    
    ct = encryptor.update(padded_data) + encryptor.finalize()
    return ct

def aes_decrypt(data: bytes) -> bytes:
    """Mendekripsi data menggunakan AES-256."""
    cipher = Cipher(algorithms.AES(KEY), modes.CBC(IV), backend=default_backend())
    decryptor = cipher.decryptor()
    
    dt = decryptor.update(data) + decryptor.finalize()
    
    # Unpadding
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    data = unpadder.update(dt) + unpadder.finalize()
    return data

def dwt_haar(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Melakukan DWT 1-level menggunakan Haar wavelet."""
    return pywt.dwt(signal, 'haar')

def idwt_haar(cA: np.ndarray, cD: np.ndarray) -> np.ndarray:
    """Melakukan IDWT 1-level menggunakan Haar wavelet."""
    return pywt.idwt(cA, cD, 'haar')

def generate_spectrogram_fig(y: np.ndarray, sr: int, title: str) -> plt.Figure:
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    
    fig, ax = plt.subplots(figsize=(8, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.set(title=title)
    fig.colorbar(ax.collections[0], ax=ax, format="%+2.0f dB")
    
    return fig

def generate_dwt_plot_fig(cD: np.ndarray, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sample_size = min(len(cD), 5000)
    ax.plot(cD[:sample_size], color='blue')
    ax.set_title(title)
    ax.set_xlabel('Koefisien Index (Sampel)')
    ax.set_ylabel('Amplitudo Detail')

    return fig
