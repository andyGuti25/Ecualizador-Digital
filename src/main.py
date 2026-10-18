import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Crear la carpeta 'results' si no existe, para guardar las gráficas automáticamente
os.makedirs('results', exist_ok=True)

# 1. Carga, Adquisición o Generación de la Señal
# Ruta del archivo de audio de prueba ubicado en la carpeta data/
audio_path = 'data/audio_prueba.wav'

if not os.path.exists(audio_path):
    print(f"⚠️ Advertencia: No se encontró el archivo en '{audio_path}'.")
    print("Por favor, coloca un archivo de audio en formato .wav dentro de la carpeta 'data/' con ese nombre para ejecutar la prueba.")
else:
    # Lectura del archivo de audio utilizando scipy
    fs, data = wavfile.read(audio_path)
    
    # Si el archivo es estéreo (2 canales), seleccionamos únicamente el canal izquierdo
    if len(data.shape) > 1:
        signal = data[:, 0]
    else:
        signal = data

    # Normalización de la señal en amplitud (valores entre -1 y 1)
    signal = signal / np.max(np.abs(signal))

    # 2. Documentación de parámetros fundamentales de Teoría de Señales
    num_samples = len(signal)
    Ts = 1 / fs               # Período de muestreo en segundos
    duration = num_samples / fs # Duración total de la señal en segundos

    print("==================================================")
    print("       PARÁMETROS DE TEORÍA DE SEÑALES           ")
    print("==================================================")
    print(f"-> Frecuencia de muestreo (fs): {fs} Hz")
    print(f"-> Período de muestreo (Ts): {Ts:.6f} s")
    print(f"-> Número total de muestras: {num_samples}")
    print(f"-> Duración de la señal: {duration:.2f} s")
    print("==================================================")

    # 3. Representación Temporal
    # Creación del vector de tiempo basado en la frecuencia de muestreo
    time = np.linspace(0, duration, num_samples)
    
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, color='blue', alpha=0.7)
    plt.title('Proyecto 14: Representación Temporal del Audio')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud Normalizada')
    plt.grid(True)
    plt.savefig('results/dominio_temporal.png')
    plt.show()
    print("✅ Gráfica de dominio temporal guardada en 'results/dominio_temporal.png'")

    # 4. Análisis Frecuencial Inicial (Transformada Rápida de Fourier - FFT)
    n = len(signal)
    fft_spectrum = np.fft.fft(signal)
    fft_frequencies = np.fft.fftfreq(n, 1/fs)

    # Filtrar únicamente la mitad positiva del espectro por simetría de señales reales
    positive_mask = fft_frequencies >= 0
    f_positive = fft_frequencies[positive_mask]
    amplitude_spectrum = np.abs(fft_spectrum[positive_mask]) / n

    plt.figure(figsize=(10, 4))
    plt.plot(f_positive, amplitude_spectrum, color='red')
    plt.title('Proyecto 14: Espectro de Amplitud Inicial (FFT)')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud Normalizada')
    plt.grid(True)
    plt.xlim(0, fs / 2) # Límite superior definido por la frecuencia de Nyquist
    plt.savefig('results/dominio_frecuencial.png')
    plt.show()
    print("✅ Gráfica de dominio frecuencial guardada en 'results/dominio_frecuencial.png'")

    # 5. Estructura base de procesamiento inicial (Ecualizador por bandas)
    print("\n--- ETAPA DE PROCESAMIENTO INICIAL (ECUALIZADOR) ---")
    print("Arquitectura de bandas planteada para el sistema:")
    print("  • Banda de Graves (< 250 Hz)")
    print("  • Banda de Medios (250 Hz - 4000 Hz)")
    print("  • Banda de Agudos (> 4000 Hz)")
    print("Nota: Los filtros digitales específicos (IIR/FIR) se integrarán conforme avancen las clases teóricas.")