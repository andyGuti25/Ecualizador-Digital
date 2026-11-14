Ecualizador Digital. Proyecto No 14.
Descripción: Diseñar e implementar un ecualizador que permita modificar diferentes
bandas de frecuencia de una señal de audio. El sistema deberá permitir observar y
comparar la señal antes y después del procesamiento tanto en el dominio temporal
como frecuencial, incorporando el diseño y evaluación de filtros digitales.

Andrés Felipe Gutiérrez
Johan Stiven Londoño

## Requisitos

- Python 3.9 o posterior.
- Las dependencias de Python indicadas en `docs/requirements.txt`: NumPy, SciPy, Matplotlib, SoundFile y SoundDevice.
- El archivo de entrada `data/audio_prueba.wav`, incluido en este repositorio. El programa admite WAV mono o estéreo; si es estéreo, analiza el primer canal.

SoundFile está declarado como dependencia, aunque la versión actual del programa no lo importa directamente.

## Instalación y ejecución

Ejecuta estos comandos desde la carpeta raíz del repositorio. El entorno virtual mantiene las dependencias del proyecto separadas de las instalaciones globales de Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r docs/requirements.txt
python src/main.py --gui
```

Si tu equipo tiene interfaz gráfica, esta orden abre una ventana con sliders para ajustar graves, medios y agudos. En macOS y Linux, `source .venv/bin/activate` activa el entorno. En Windows PowerShell, usa `.venv\Scripts\Activate.ps1` en su lugar. Para salir del entorno virtual, ejecuta `deactivate`.

En macOS, usa una versión reciente de Python instalada desde [python.org](https://www.python.org/downloads/macos/) para la interfaz gráfica. Algunas versiones de Python incluidas con Xcode traen Tk antiguo y no pueden abrir la ventana. Comprueba la versión de Tk con `python -c "import tkinter; print(tkinter.TkVersion)"`; se requiere Tk 8.6 o posterior. Si instalas otro Python, crea el entorno virtual nuevamente con ese intérprete.

## Qué ocurre al ejecutarlo

La versión gráfica permite cargar un archivo WAV, ajustar la ganancia de cada banda (graves-medios-agudos), modificar el volumen general en dB y cambiar la velocidad entre 0.5× y 2×. Esta velocidad cambia también el tono: 0.5× duplica la duración y baja el tono; 2× reduce la duración a la mitad y sube el tono. La interfaz permite escuchar el audio original o procesado y detener la reproducción. Sus pestañas muestran las comparaciones temporales y frecuenciales, y al aplicar los cambios guarda el resultado en `results/`. El programa también calcula e imprime en la terminal la frecuencia de muestreo, el período de muestreo, el número de muestras, la duración, los valores máximo y mínimo, el valor medio y el RMS. Además genera:

- `results/dominio_temporal_comparacion.png`: comparación temporal entre la señal original y la procesada.
- `results/dominio_frecuencial_comparacion.png`: comparación del espectro antes y después del ecualizador.
- `results/audio_ecualizado.wav`: archivo de audio procesado en formato WAV.

Las gráficas también se guardan en `results/`. Para ejecutar el programa sin abrir una interfaz gráfica, usa la versión en modo terminal:

```bash
MPLBACKEND=Agg python3 src/main.py --no-show
```

También es posible ajustar la ganancia por banda desde la línea de comandos:

```bash
python3 src/main.py --low-gain-db 6 --mid-gain-db 2 --high-gain-db 5 --no-show
```

## Alcance actual

La versión actual implementa un ecualizador digital simple con filtros Butterworth de orden 4 para separar graves, medios y agudos. Permite modificar la ganancia de cada banda, visualizar las señales dentro de la interfaz, escuchar la entrada y la salida, y guardar el audio procesado.

