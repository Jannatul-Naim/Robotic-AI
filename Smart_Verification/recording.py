import sounddevice as sd
from scipy.io.wavfile import write

sr = 16000
duration = 2  # seconds

print("Recording...")
audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
sd.wait()

write("myvoice.wav", sr, audio)
print("Saved to myvoice.wav")
