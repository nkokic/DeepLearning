import whisper
import time

start_time = time.time()

model = whisper.load_model("tiny")

load_time = time.time()
print(f"Load time: {load_time -  start_time} s")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("Projekt/audio-M.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

analysis_time = time.time()
print(f"Analysis time: {analysis_time -  load_time} s")

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

lang_time = time.time()
print(f"Language detection time: {lang_time -  analysis_time} s")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

decode_time = time.time()
print(f"Decoding time: {decode_time -  lang_time} s")
print(f"Total time: {decode_time -  load_time} s")
print(f"Total (loaded and language preset) time: {decode_time -  lang_time + analysis_time - load_time} s")

# print the recognized text
print(result.text)