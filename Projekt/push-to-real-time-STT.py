import whisper
import pyaudio
import wave
import tempfile
from ctypes import *
import numpy as np
import time
import keyboard

# Load Whisper model once
model = whisper.load_model("tiny.en")

def transcribe_directly():
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    # Audio settings
    sample_rate = 44100
    bits_per_sample = 16
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 2

    # Open the WAV file
    wav_file = wave.open(temp_file.name, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(bits_per_sample // 8)
    wav_file.setframerate(sample_rate)

    audio = pyaudio.PyAudio()
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Hold SPACE to record...")

    recording = []
    triggered = False
    time_started = 0.0

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            
            if keyboard.is_pressed('space'):
                recording.append(data)
                if not triggered:
                    print("Key pressed, recording...")
                    time_started = time.time()
                triggered = True

            if triggered and not keyboard.is_pressed('space'):
                print("Key released. Stopping recording.")
                break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Write to file only if we recorded something
    if recording:
        for frame in recording:
            wav_file.writeframes(frame)
        wav_file.close()
        time_recorded = time.time()
        print("Transcribing...")
        result = model.transcribe(temp_file.name, fp16=False)
        print("Transcription complete.")
        print(f"→ {result['text'].strip()}")
        time_transcribed = time.time()
        print(f"Record time: {time_recorded - time_started}")
        print(f"Transcription time: {time_transcribed - time_recorded}")
        return result["text"].strip()
    else:
        print("No audio detected.")
        return ""

if __name__ == "__main__":
    try:
        while True:
            transcribe_directly()
    except KeyboardInterrupt:
        print("Stopping.")
