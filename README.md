Ecualizador Digital. Proyecto No 14.
Descripción: Diseñar e implementar un ecualizador que permita modificar diferentes
bandas de frecuencia de una señal de audio. El sistema deberá permitir observar y
comparar la señal antes y después del procesamiento tanto en el dominio temporal
como frecuencial, incorporando el diseño y evaluación de filtros digitales.

Andrés Felipe Gutiérrez
Johan Stiven Londoño

## Requisitos

- Python 3.9 o posterior.
- Las dependencias de Python indicadas en `docs/requirements.txt`: NumPy, SciPy, Matplotlib y SoundFile.
- El archivo de entrada `data/audio_prueba.wav`, incluido en este repositorio. El programa admite WAV mono o estéreo; si es estéreo, analiza el primer canal.

SoundFile está declarado como dependencia, aunque la versión actual del programa no lo importa directamente.

## Instalación y ejecución

Ejecuta estos comandos desde la carpeta raíz del repositorio. El entorno virtual mantiene las dependencias del proyecto separadas de las instalaciones globales de Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/requirements.txt
python src/main.py
```

En macOS y Linux, `source .venv/bin/activate` activa el entorno. En Windows PowerShell, usa `.venv\Scripts\Activate.ps1` en su lugar. Para salir del entorno virtual, ejecuta `deactivate`.

## Qué ocurre al ejecutarlo

El programa lee `data/audio_prueba.wav`, normaliza la señal y muestra en la terminal la frecuencia y el período de muestreo, el número de muestras, la duración, los valores máximo y mínimo, el valor medio y el RMS. Después crea y muestra estas gráficas:

- `results/dominio_temporal.png`: amplitud de la señal original a través del tiempo.
- `results/dominio_frecuencial.png`: espectro calculado con FFT y bandas de frecuencia indicadas sobre la gráfica.
- `results/comparacion_ruido.png`: comparación de la señal original con una versión a la que se añade ruido simulado con SNR nominal de 20 dB.

Las ventanas de Matplotlib se muestran una por una: cierra cada ventana para que el programa continúe. Las imágenes se guardan localmente en `results/` antes de mostrarse. Se vuelven a generar al ejecutar el programa y están excluidas del control de versiones por `.gitignore`; por tanto, no aparecerán automáticamente en el repositorio remoto.

Para ejecutarlo sin abrir ventanas gráficas (por ejemplo, en un servidor sin escritorio), usa:

```bash
MPLBACKEND=Agg python src/main.py
```

En Windows PowerShell, el equivalente es:

```powershell
$env:MPLBACKEND = "Agg"
python src/main.py
```

## Alcance actual

Esta versión caracteriza y visualiza la señal de entrada y genera una comparación con ruido simulado. Las bandas de graves, medios y agudos se señalan en el análisis, pero los filtros digitales Butterworth/FIR y el ajuste de ganancia por banda todavía no están implementados. El programa tampoco genera ni guarda un archivo de audio ecualizado. Aún falta una interfaz interactiva.

