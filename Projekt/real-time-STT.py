import whisper
import pyaudio
import wave
import tempfile
from ctypes import *
import numpy as np
import time

# Load Whisper model once
model = whisper.load_model("tiny.en")

def transcribe_directly():
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

    # Audio settings
    sample_rate = 16000
    bits_per_sample = 16
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    # Silence detection parameters
    threshold = 500     # Adjust based on your microphone sensitivity
    silence_limit = 1.0 # seconds of silence before stopping

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

    print("Listening for speech...")

    recording = []
    silence_start = None
    speaking = False

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            samples = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(np.square(samples)))

            if rms > threshold:
                # Detected speech
                if not speaking:
                    print("Speech detected, recording...")
                    speaking = True
                recording.append(data)
                silence_start = None  # reset silence timer
            else:
                # Detected silence
                if speaking:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_limit:
                        print("Silence detected. Stopping recording.")
                        break

    except KeyboardInterrupt:
        print("Stopped manually.")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    # Write to file only if we recorded something
    if recording:
        for frame in recording:
            wav_file.writeframes(frame)
        wav_file.close()

        print("Transcribing...")
        result = model.transcribe(temp_file.name, fp16=False)
        print("Transcription complete.")
        print(f"→ {result['text'].strip()}")
        return result["text"].strip()
    else:
        print("No audio detected.")
        return ""

if __name__ == "__main__":
    transcribe_directly()
