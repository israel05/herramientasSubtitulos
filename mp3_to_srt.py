from faster_whisper import WhisperModel
import datetime
import argparse
import os
import time # Para medir el tiempo

def format_timestamp(seconds: float) -> str:
    """Convierte segundos a formato de timestamp SRT (HH:MM:SS,ms)."""
    assert seconds >= 0, "El tiempo no puede ser negativo"
    milliseconds = round(seconds * 1000.0)

    td = datetime.timedelta(milliseconds=milliseconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000

    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def create_srt_faster(audio_path: str, srt_path: str, model_name: str = "base", device: str = "cpu", compute_type: str = "int8"):
    """
    Transcribe un archivo de audio usando faster-whisper y guarda el resultado como un archivo SRT.

    Args:
        audio_path (str): Ruta al archivo de audio (MP3, WAV, etc.).
        srt_path (str): Ruta donde se guardará el archivo SRT resultante.
        model_name (str): Nombre del modelo Whisper a usar
                            (tiny, base, small, medium, large, large-v2, large-v3).
                            'base' es un buen punto de partida.
        device (str): Dispositivo a usar ('cpu' o 'cuda'). Requiere configuración especial para 'cuda'.
        compute_type (str): Tipo de cómputo a usar ('int8', 'float16', 'float32').
                            'int8' suele ser bueno para CPU.
                            'float16' es bueno para GPUs NVIDIA modernas.
                            Consulta la documentación de faster-whisper para compatibilidad.
    """
    print(f"Cargando el modelo faster-whisper '{model_name}' en '{device}' con compute_type '{compute_type}'...")
    model = None
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("Modelo cargado exitosamente.")
    except Exception as e:
        print(f"Error al cargar el modelo faster-whisper: {e}")
        print("Asegúrate de que las dependencias (como CTranslate2) se instalaron correctamente.")
        print(f"Verifica que el dispositivo '{device}' y el tipo de cómputo '{compute_type}' sean compatibles con tu sistema.")
        return

    print(f"Transcribiendo el archivo de audio: {audio_path}")
    start_time = time.time()
    try:
        segments, info = model.transcribe(audio_path, beam_size=5)

        print(f"Idioma detectado: {info.language} (probabilidad: {info.language_probability:.2f})")
        print(f"Duración del audio: {format_timestamp(info.duration)}")

    except Exception as e:
        print(f"Error durante la transcripción: {e}")
        print("Verifica que el archivo de audio sea válido y que ffmpeg esté funcionando.")
        return
    finally:
        if model is not None:
            del model
            print("Modelo descargado.")

    print(f"Generando archivo SRT en: {srt_path}")
    with open(srt_path, 'w', encoding='utf-8') as srt_file:
        segment_index = 1
        for segment in segments:
            seg_start_time = format_timestamp(segment.start)
            seg_end_time = format_timestamp(segment.end)
            text = segment.text.strip()

            srt_file.write(f"{segment_index}\n")
            srt_file.write(f"{seg_start_time} --> {seg_end_time}\n")
            srt_file.write(f"{text}\n\n")
            segment_index += 1
            print(f"[{seg_start_time} --> {seg_end_time}] {text}")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Archivo SRT '{srt_path}' creado exitosamente.")
    print(f"Tiempo total de procesamiento (transcripción + guardado): {processing_time:.2f} segundos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera archivos SRT desde archivos de audio MP3 en el directorio actual usando faster-Whisper.")
    parser.add_argument("-m", "--model", default="large-v3", help="Modelo Whisper a usar (tiny, base, small, medium, large, large-v2, large-v3). Default: base")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Dispositivo a usar (cpu o cuda). Default: cpu")
    parser.add_argument("--compute_type", default="float32", choices=["int8", "int16", "float16", "float32"], help="Tipo de cómputo (int8, int16, float16, float32). Default: int8")

    args = parser.parse_args()

    current_directory = os.getcwd()
    files_in_directory = os.listdir(current_directory)

    mp3_files = [f for f in files_in_directory if f.lower().endswith(".mp3")]

    if not mp3_files:
        print("No se encontraron archivos MP3 en el directorio actual.")
    else:
        print(f"Se encontraron {len(mp3_files)} archivos MP3 en el directorio actual:")
        for audio_file in mp3_files:
            print(f"- {audio_file}")

        for audio_file in mp3_files:
            audio_path = os.path.join(current_directory, audio_file)
            base_name = os.path.splitext(audio_file)[0]
            output_path = os.path.join(current_directory, base_name + ".srt")

            print(f"\nProcesando archivo: {audio_file}")
            create_srt_faster(audio_path, output_path, args.model, args.device, args.compute_type)

        print("\nProceso completado para todos los archivos MP3.")