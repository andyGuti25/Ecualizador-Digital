import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

os.makedirs('results', exist_ok=True)

audio_path = 'data/audio_prueba.wav'

if not os.path.exists(audio_path):
    print(f"⚠️ Advertencia: No se encontró el archivo en '{audio_path}'.")
    print("Coloque un archivo .wav en la carpeta 'data/' para ejecutar el análisis.")
else:
    fs, data = wavfile.read(audio_path)
    
    if len(data.shape) > 1:
        signal = data[:, 0]
    else:
        signal = data

    signal = signal.astype(np.float32)
    signal = signal / np.max(np.abs(signal))

    num_samples = len(signal)
    Ts = 1 / fs
    duration = num_samples / fs
    val_max = np.max(signal)
    val_min = np.min(signal)
    val_mean = np.mean(signal)
    rms_val = np.sqrt(np.mean(signal**2))

    print("==================================================")
    print("           PARÁMETROS DE TEORÍA DE SEÑALES            ")
    print("==================================================")
    print(f"-> Frecuencia de muestreo (fs): {fs} Hz")
    print(f"-> Período de muestreo (Ts):     {Ts:.6f} s")
    print(f"-> Número total de muestras:   {num_samples}")
    print(f"-> Duración de la señal:        {duration:.2f} s")
    print(f"-> Amplitud Máxima / Mínima:    {val_max:.2f} / {val_min:.2f}")
    print(f"-> Valor medio (Offset DC):     {val_mean:.4f}")
    print(f"-> Valor RMS (Potencia aprox):  {rms_val:.4f}")
    print("==================================================")

    time = np.linspace(0, duration, num_samples)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, color='#1f77b4', alpha=0.8, label='Señal limpia')
    plt.title('Proyecto 14: Representación Temporal del Audio Original', fontsize=12, fontweight='bold')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud Normalizada')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, duration)
    plt.tight_layout()
    plt.savefig('results/dominio_temporal.png', dpi=300)
    plt.show()
    print("✅ Gráfica temporal guardada en 'results/dominio_temporal.png'")

    n = len(signal)
    fft_spectrum = np.fft.fft(signal)
    fft_frequencies = np.fft.fftfreq(n, 1/fs)

    positive_mask = fft_frequencies >= 0
    f_positive = fft_frequencies[positive_mask]
    amplitude_spectrum = np.abs(fft_spectrum[positive_mask]) / n

    plt.figure(figsize=(10, 5))
    plt.plot(f_positive, amplitude_spectrum, color='crimson', lw=1.2, label='Espectro FFT')
    
    plt.axvspan(0, 250, color='blue', alpha=0.1, label='Graves (<250 Hz)')
    plt.axvspan(250, 4000, color='green', alpha=0.1, label='Medios (250-4000 Hz)')
    plt.axvspan(4000, fs/2, color='orange', alpha=0.1, label='Agudos (>4000 Hz)')
    
    plt.axvline(fs/2, color='black', linestyle=':', label=f'Frecuencia Nyquist ({fs/2:.0f} Hz)')

    plt.title('Proyecto 14: Análisis Frecuencial e Identificación de Bandas', fontsize=12, fontweight='bold')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud Normalizada')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, fs / 2)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('results/dominio_frecuencial.png', dpi=300)
    plt.show()
    print("✅ Gráfica frecuencial guardada en 'results/dominio_frecuencial.png'")

    snr_db = 20
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise

    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, color='blue', alpha=0.5, label='Señal Original')
    plt.plot(time, noisy_signal, color='red', alpha=0.4, label='Señal con Ruido (SNR 20 dB)')
    plt.title('Comparación Temporal: Señal Original vs Señal con Ruido', fontsize=12, fontweight='bold')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/comparacion_ruido.png', dpi=300)
    plt.show()
    print("✅ Comparación de ruido guardada en 'results/comparacion_ruido.png'")

    print("\n--- ETAPA DE PROCESAMIENTO INICIAL Y ARQUITECTURA ---")
    print("Estructura de filtrado multibanda configurada correctamente:")
    print("  1. Banda de Graves  : 20 Hz  a 250 Hz")
    print("  2. Banda de Medios  : 250 Hz a 4000 Hz")
    print("  3. Banda de Agudos  : 4000 Hz a 20000 Hz (o Nyquist)")
    print("Procesamiento en desarrollo: Filtros Butterworth IIR / FIR en la Entrega Final.")