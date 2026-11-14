import argparse
import os
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

if os.environ.get('DISPLAY', '') == '' and sys.platform != 'win32':
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover
    tk = None
    filedialog = None
    messagebox = None
    ttk = None

RESULTS_DIR = Path('results')
AUDIO_PATH = Path('data/audio_prueba.wav')


def normalize_signal(values):
    values = values.astype(np.float32)
    peak = np.max(np.abs(values))
    if peak == 0:
        return values
    return values / peak


def parse_audio(audio_path):
    if not audio_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo '{audio_path}'.")

    fs, data = wavfile.read(audio_path)
    if len(data.shape) > 1:
        signal_data = data[:, 0]
    else:
        signal_data = data

    return fs, normalize_signal(signal_data)


def compute_metrics(signal_data, fs):
    num_samples = len(signal_data)
    duration = num_samples / fs
    return {
        'fs': fs,
        'Ts': 1 / fs,
        'num_samples': num_samples,
        'duration': duration,
        'peak': float(np.max(np.abs(signal_data))),
        'val_max': float(np.max(signal_data)),
        'val_min': float(np.min(signal_data)),
        'val_mean': float(np.mean(signal_data)),
        'rms': float(np.sqrt(np.mean(signal_data ** 2))),
    }


def design_filter(fs, low_cut=None, high_cut=None, order=4, filter_type='bandpass'):
    nyquist = fs / 2
    if filter_type == 'low':
        cutoff = low_cut / nyquist
        return signal.butter(order, cutoff, btype='low')
    if filter_type == 'high':
        cutoff = high_cut / nyquist
        return signal.butter(order, cutoff, btype='high')
    if filter_type == 'bandpass':
        low = low_cut / nyquist
        high = high_cut / nyquist
        return signal.butter(order, [low, high], btype='bandpass')
    raise ValueError(f'Filtro no soportado: {filter_type}')


def build_equalizer(signal_data, fs, low_gain_db, mid_gain_db, high_gain_db):
    low_b, low_a = design_filter(fs, low_cut=250, filter_type='low')
    mid_b, mid_a = design_filter(fs, low_cut=250, high_cut=4000, filter_type='bandpass')
    high_b, high_a = design_filter(fs, high_cut=4000, filter_type='high')

    low_band = signal.filtfilt(low_b, low_a, signal_data)
    mid_band = signal.filtfilt(mid_b, mid_a, signal_data)
    high_band = signal.filtfilt(high_b, high_a, signal_data)

    low_gain = 10 ** (low_gain_db / 20)
    mid_gain = 10 ** (mid_gain_db / 20)
    high_gain = 10 ** (high_gain_db / 20)

    equalized = low_band * low_gain + mid_band * mid_gain + high_band * high_gain
    equalized = normalize_signal(equalized)
    return equalized, {'low': low_band, 'mid': mid_band, 'high': high_band}


def change_playback_speed(signal_data, speed_factor):
    if speed_factor <= 0:
        raise ValueError('La velocidad debe ser mayor que cero.')
    if np.isclose(speed_factor, 1.0):
        return signal_data.copy()

    ratio = Fraction(speed_factor).limit_denominator(1000)
    return signal.resample_poly(signal_data, up=ratio.denominator, down=ratio.numerator)


def apply_master_gain(signal_data, gain_db):
    return (signal_data * (10 ** (gain_db / 20))).astype(np.float32)


def plot_time_domain(fs, original, processed, save_path):
    original_time = np.arange(len(original)) / fs
    processed_time = np.arange(len(processed)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(original_time, original, color='#1f77b4', alpha=0.7, label='Original')
    plt.plot(processed_time, processed, color='#d62728', alpha=0.7, label='Procesada')
    plt.title('Señal original vs. señal procesada', fontsize=12, fontweight='bold')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def plot_frequency_domain(fs, original, processed, save_path):
    def magnitude(signal_data):
        spectrum = np.fft.rfft(signal_data)
        freqs = np.fft.rfftfreq(len(signal_data), d=1 / fs)
        return freqs, np.abs(spectrum) / len(signal_data)

    freq_orig, mag_orig = magnitude(original)
    freq_proc, mag_proc = magnitude(processed)

    plt.figure(figsize=(10, 5))
    plt.plot(freq_orig, mag_orig, color='royalblue', lw=1.2, label='Original')
    plt.plot(freq_proc, mag_proc, color='crimson', lw=1.2, label='Procesada')

    plt.axvspan(0, 250, color='blue', alpha=0.08, label='Graves')
    plt.axvspan(250, 4000, color='green', alpha=0.08, label='Medios')
    plt.axvspan(4000, fs / 2, color='orange', alpha=0.08, label='Agudos')

    plt.axvline(250, color='black', linestyle=':', linewidth=1)
    plt.axvline(4000, color='black', linestyle=':', linewidth=1)

    plt.title('Comparación del espectro antes y después del ecualizador', fontsize=12, fontweight='bold')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud normalizada')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlim(0, min(fs / 2, 20000))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)


def run_equalizer(audio_path, low_gain_db, mid_gain_db, high_gain_db, show_plot):
    RESULTS_DIR.mkdir(exist_ok=True)

    fs, signal_data = parse_audio(audio_path)
    metrics = compute_metrics(signal_data, fs)

    print('==================================================')
    print('       PARÁMETROS DE TEORÍA DE SEÑALES')
    print('==================================================')
    print(f'-> Frecuencia de muestreo (fs): {fs} Hz')
    print(f'-> Período de muestreo (Ts): {metrics["Ts"]:.6f} s')
    print(f'-> Número total de muestras: {metrics["num_samples"]}')
    print(f'-> Duración de la señal: {metrics["duration"]:.2f} s')
    print(f'-> Amplitud máxima / mínima: {metrics["val_max"]:.4f} / {metrics["val_min"]:.4f}')
    print(f'-> Valor medio: {metrics["val_mean"]:.4f}')
    print(f'-> RMS: {metrics["rms"]:.4f}')
    print('==================================================')

    equalized, _ = build_equalizer(signal_data, fs, low_gain_db, mid_gain_db, high_gain_db)
    wavfile.write(str(RESULTS_DIR / 'audio_ecualizado.wav'), fs, equalized.astype(np.float32))

    plot_time_domain(fs, signal_data, equalized, RESULTS_DIR / 'dominio_temporal_comparacion.png')
    plot_frequency_domain(fs, signal_data, equalized, RESULTS_DIR / 'dominio_frecuencial_comparacion.png')

    print(f'\nEcualizador aplicado con ganancias: graves={low_gain_db} dB, medios={mid_gain_db} dB, agudos={high_gain_db} dB')
    print(f'Archivo de audio guardado en: {RESULTS_DIR / "audio_ecualizado.wav"}')
    print(f'Gráfica temporal guardada en: {RESULTS_DIR / "dominio_temporal_comparacion.png"}')
    print(f'Gráfica frecuencial guardada en: {RESULTS_DIR / "dominio_frecuencial_comparacion.png"}')

    if show_plot:
        plt.show()


def is_gui_available():
    return sys.platform == 'win32' or sys.platform == 'darwin' or bool(os.environ.get('DISPLAY'))


class EqualizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Ecualizador Digital - Proyecto 14')
        self.geometry('1000x820')
        self.minsize(800, 650)
        self.audio_path = Path(AUDIO_PATH)
        self.output_path = RESULTS_DIR / 'audio_ecualizado.wav'
        self.fs = None
        self.original_signal = None
        self.equalized_signal = None

        self.low_gain = tk.DoubleVar(value=6.0)
        self.mid_gain = tk.DoubleVar(value=0.0)
        self.high_gain = tk.DoubleVar(value=4.0)
        self.master_gain = tk.DoubleVar(value=0.0)
        self.speed_factor = tk.DoubleVar(value=1.0)

        self._build_ui()
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.apply_equalizer(show_message=False)

    def _build_ui(self):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        main_frame = ttk.Frame(self, padding=16)
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text='Ecualizador digital de audio', font=('Arial', 14, 'bold')).pack(anchor='w', pady=(0, 10))

        ttk.Label(main_frame, text='Archivo de entrada:').pack(anchor='w')
        file_row = ttk.Frame(main_frame)
        file_row.pack(fill='x', pady=(0, 10))
        self.audio_label = ttk.Label(file_row, text=str(self.audio_path), wraplength=760)
        self.audio_label.pack(side='left', fill='x', expand=True)
        ttk.Button(file_row, text='Cargar WAV', command=self.load_audio).pack(side='right', padx=(10, 0))

        controls = ttk.Frame(main_frame)
        controls.pack(fill='x', pady=(0, 12))

        for label, variable, min_val, max_val, suffix, decimals in [
            ('Graves (dB)', self.low_gain, -12.0, 12.0, ' dB', 1),
            ('Medios (dB)', self.mid_gain, -12.0, 12.0, ' dB', 1),
            ('Agudos (dB)', self.high_gain, -12.0, 12.0, ' dB', 1),
            ('Amplitud (dB)', self.master_gain, -12.0, 2.0, ' dB', 1),
            ('Velocidad', self.speed_factor, 0.5, 2.0, '×', 2),
        ]:
            row = ttk.Frame(controls)
            row.pack(fill='x', pady=4)
            ttk.Label(row, text=label, width=14).pack(side='left')
            scale = ttk.Scale(row, from_=min_val, to=max_val, variable=variable, orient='horizontal')
            scale.pack(side='left', fill='x', expand=True)
            value_label = ttk.Label(row, text=f'{variable.get():.{decimals}f}{suffix}', width=8)
            value_label.pack(side='left', padx=(8, 0))
            variable.trace_add(
                'write',
                lambda *_args, var=variable, label=value_label, digits=decimals, unit=suffix:
                    label.config(text=f'{var.get():.{digits}f}{unit}'),
            )

        ttk.Label(
            controls,
            text='0.5× alarga y baja el tono; 2× acorta y sube el tono. Se reserva margen para evitar saturación.',
        ).pack(anchor='w', pady=(4, 0))

        actions = ttk.Frame(main_frame)
        actions.pack(fill='x', pady=(8, 0))
        ttk.Button(actions, text='Aplicar ecualización', command=self.apply_equalizer).pack(side='left', padx=(0, 8))
        self.play_original_button = ttk.Button(actions, text='▶ Escuchar original', command=self.play_original)
        self.play_original_button.pack(side='left', padx=4)
        self.play_equalized_button = ttk.Button(actions, text='▶ Escuchar ecualizada', command=self.play_equalized)
        self.play_equalized_button.pack(side='left', padx=4)
        ttk.Button(actions, text='■ Detener audio', command=self.stop_audio).pack(side='left', padx=4)
        ttk.Button(actions, text='Salir', command=self.close).pack(side='right')
        if sd is None:
            self.play_original_button.state(['disabled'])
            self.play_equalized_button.state(['disabled'])

        self.status_var = tk.StringVar(value='Listo para procesar.')
        ttk.Label(main_frame, textvariable=self.status_var, foreground='darkgreen', wraplength=920).pack(anchor='w', pady=(10, 8))

        plots = ttk.Notebook(main_frame)
        plots.pack(fill='both', expand=True)

        time_tab = ttk.Frame(plots)
        frequency_tab = ttk.Frame(plots)
        plots.add(time_tab, text='Dominio temporal')
        plots.add(frequency_tab, text='Dominio frecuencial')

        self.time_figure = Figure(figsize=(9, 3.2), dpi=100)
        self.time_axes = self.time_figure.add_subplot(111)
        self.time_canvas = FigureCanvasTkAgg(self.time_figure, master=time_tab)
        self.time_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.frequency_figure = Figure(figsize=(9, 3.2), dpi=100)
        self.frequency_axes = self.frequency_figure.add_subplot(111)
        self.frequency_canvas = FigureCanvasTkAgg(self.frequency_figure, master=frequency_tab)
        self.frequency_canvas.get_tk_widget().pack(fill='both', expand=True)

    def load_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[('Archivos WAV', '*.wav'), ('Todos', '*.*')])
        if file_path:
            self.audio_path = Path(file_path)
            self.audio_label.config(text=str(self.audio_path))
            self.apply_equalizer(show_message=False)

    def update_plots(self):
        original_step = max(1, len(self.original_signal) // 10000)
        processed_step = max(1, len(self.equalized_signal) // 10000)
        original_time = np.arange(0, len(self.original_signal), original_step) / self.fs
        processed_time = np.arange(0, len(self.equalized_signal), processed_step) / self.fs

        self.time_axes.clear()
        self.time_axes.plot(original_time, self.original_signal[::original_step], color='royalblue', alpha=0.8, label='Original')
        self.time_axes.plot(processed_time, self.equalized_signal[::processed_step], color='crimson', alpha=0.8, label='Procesada')
        self.time_axes.set_title('Comparación en el dominio temporal')
        self.time_axes.set_xlabel('Tiempo (s)')
        self.time_axes.set_ylabel('Amplitud')
        self.time_axes.grid(True, linestyle='--', alpha=0.4)
        self.time_axes.legend(loc='upper right')
        self.time_figure.tight_layout()
        self.time_canvas.draw_idle()

        original_count = len(self.original_signal)
        processed_count = len(self.equalized_signal)
        original_freqs = np.fft.rfftfreq(original_count, d=1 / self.fs)
        processed_freqs = np.fft.rfftfreq(processed_count, d=1 / self.fs)
        original_magnitude = np.abs(np.fft.rfft(self.original_signal)) / original_count
        processed_magnitude = np.abs(np.fft.rfft(self.equalized_signal)) / processed_count

        self.frequency_axes.clear()
        self.frequency_axes.plot(original_freqs, original_magnitude, color='royalblue', label='Original')
        self.frequency_axes.plot(processed_freqs, processed_magnitude, color='crimson', label='Procesada')
        self.frequency_axes.axvspan(0, min(250, self.fs / 2), color='blue', alpha=0.08, label='Graves')
        self.frequency_axes.axvspan(250, min(4000, self.fs / 2), color='green', alpha=0.08, label='Medios')
        if self.fs / 2 > 4000:
            self.frequency_axes.axvspan(4000, self.fs / 2, color='orange', alpha=0.08, label='Agudos')
        self.frequency_axes.axvline(250, color='black', linestyle=':', linewidth=1)
        self.frequency_axes.axvline(4000, color='black', linestyle=':', linewidth=1)
        self.frequency_axes.set_title('Comparación en el dominio frecuencial')
        self.frequency_axes.set_xlabel('Frecuencia (Hz)')
        self.frequency_axes.set_ylabel('Magnitud')
        self.frequency_axes.set_xlim(0, min(self.fs / 2, 20000))
        self.frequency_axes.grid(True, linestyle='--', alpha=0.4)
        self.frequency_axes.legend(loc='upper right')
        self.frequency_figure.tight_layout()
        self.frequency_canvas.draw_idle()

    def apply_equalizer(self, show_message=True):
        try:
            if not self.audio_path.exists():
                raise FileNotFoundError(f'No existe el archivo: {self.audio_path}')

            RESULTS_DIR.mkdir(exist_ok=True)
            self.fs, self.original_signal = parse_audio(self.audio_path)
            equalized, _ = build_equalizer(
                self.original_signal,
                self.fs,
                float(self.low_gain.get()),
                float(self.mid_gain.get()),
                float(self.high_gain.get()),
            )
            time_scaled = normalize_signal(
                change_playback_speed(equalized, float(self.speed_factor.get()))
            ) * 0.75
            self.equalized_signal = apply_master_gain(time_scaled, float(self.master_gain.get()))
            wavfile.write(str(self.output_path), self.fs, self.equalized_signal)

            plot_time_domain(self.fs, self.original_signal, self.equalized_signal, RESULTS_DIR / 'dominio_temporal_comparacion.png')
            plot_frequency_domain(self.fs, self.original_signal, self.equalized_signal, RESULTS_DIR / 'dominio_frecuencial_comparacion.png')
            self.update_plots()

            self.status_var.set(
                f'Procesamiento listo: {self.output_path} | Volumen={self.master_gain.get():.1f} dB | '
                f'Velocidad={self.speed_factor.get():.2f}× | Duración={len(self.equalized_signal) / self.fs:.2f} s'
            )
            if show_message:
                messagebox.showinfo('Ecualización completada', 'Audio y gráficas actualizados. El WAV está guardado en results/.')
        except Exception as exc:  # pragma: no cover
            self.status_var.set(f'Error: {exc}')
            if show_message:
                messagebox.showerror('Error', str(exc))

    def play_audio(self, audio_data, label):
        if sd is None:
            messagebox.showerror('Audio no disponible', 'Instala las dependencias del proyecto para habilitar la reproducción.')
            return
        if audio_data is None or self.fs is None:
            messagebox.showwarning('Sin audio', 'Carga un archivo WAV antes de reproducir.')
            return
        try:
            sd.stop()
            sd.play(audio_data, self.fs)
            self.status_var.set(f'Reproduciendo: {label}')
        except Exception as exc:
            messagebox.showerror('Error de reproducción', str(exc))

    def play_original(self):
        self.play_audio(self.original_signal, 'señal original')

    def play_equalized(self):
        self.play_audio(self.equalized_signal, 'señal ecualizada')

    def stop_audio(self):
        if sd is not None:
            sd.stop()
        self.status_var.set('Reproducción detenida.')

    def close(self):
        if sd is not None:
            sd.stop()
        self.destroy()


def main():
    parser = argparse.ArgumentParser(description='Ecualizador digital de audio en Python.')
    parser.add_argument('--audio', type=str, default=str(AUDIO_PATH), help='Ruta del archivo WAV de entrada.')
    parser.add_argument('--low-gain-db', type=float, default=6.0, help='Ganancia para la banda de graves (dB).')
    parser.add_argument('--mid-gain-db', type=float, default=0.0, help='Ganancia para la banda de medios (dB).')
    parser.add_argument('--high-gain-db', type=float, default=4.0, help='Ganancia para la banda de agudos (dB).')
    parser.add_argument('--no-show', action='store_true', help='No abre ventanas gráficas; solo guarda resultados.')
    parser.add_argument('--gui', action='store_true', help='Abre la interfaz gráfica del ecualizador.')
    args = parser.parse_args()

    if args.gui or (not args.no_show and is_gui_available() and tk is not None):
        if tk is None:
            print('Tkinter no está disponible. Instala una versión de Python que incluya Tk 8.6 o posterior.')
            raise SystemExit(1)
        if tk.TkVersion < 8.6:
            print(
                f'Tk {tk.TkVersion} no es compatible con esta interfaz. '
                'Se requiere Tk 8.6 o posterior. En macOS, instala Python desde python.org, '
                'crea de nuevo el entorno virtual y vuelve a ejecutar el programa.'
            )
            raise SystemExit(1)
        app = EqualizerApp()
        app.mainloop()
        return

    try:
        run_equalizer(
            audio_path=Path(args.audio),
            low_gain_db=args.low_gain_db,
            mid_gain_db=args.mid_gain_db,
            high_gain_db=args.high_gain_db,
            show_plot=not args.no_show,
        )
    except FileNotFoundError as exc:
        print(f'⚠️ {exc}')
        print("Coloque un archivo .wav en la carpeta 'data/' para ejecutar el ecualizador.")
        raise SystemExit(1)


if __name__ == '__main__':
    main()