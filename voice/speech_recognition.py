
import vosk, sounddevice as sd, json, queue

q = queue.Queue()
model = vosk.Model("vosk-model-small-en-us-0.15")

def callback(indata, frames, time, status):
    q.put(bytes(indata))

rec = vosk.KaldiRecognizer(model, 16000)

with sd.RawInputStream(samplerate=16000, channels=1,
                       dtype='int16', callback=callback):
    while True:
        data = q.get()
        if rec.AcceptWaveform(data):
            text = json.loads(rec.Result())['text']
            print("User:", text)
            if text:
                break
