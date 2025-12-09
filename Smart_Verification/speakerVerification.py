# =========================================================
# INSTALLATION
# pip install onnxruntime sounddevice numpy scipy soundfile librosa torchaudio
# =========================================================

import numpy as np
import sounddevice as sd
import soundfile as sf
import scipy.signal
import onnxruntime as ort
import torchaudio
import torch
from collections import deque
import os
import time

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "ecapa.onnx"
VOICE_PATH = "myvoice.wav"
SR = 16000

BUFFER_SECONDS = 2.0
BLOCK_SIZE = 1024        # <<< FIXED (real-time safe)
THRESHOLD = 0.7

# =========================================================
# LOAD MODEL
# =========================================================
sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = sess.get_inputs()[0].name
print("Loaded ONNX model:", input_name)

# =========================================================
# FBANK PREPROCESSING
# =========================================================
def wav_to_fbank(wav, sr):
    # ensure mono
    if len(wav.shape) > 1:
        wav = wav[:, 0]

    # convert to float32 tensor
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)

    # normalize amplitude
    wav = wav / (wav.abs().max() + 1e-9)

    # resample to 16k
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)

    # EXACT ECAPA FBANK (80 bins, kaldi)
    fbank = torchaudio.compliance.kaldi.fbank(
        wav,
        sample_frequency=16000,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        low_freq=20,
        high_freq=-400,   # ← IMPORTANT: ECAPA uses high_freq=-400
        dither=0.0,
        energy_floor=0.0,
        window_type="hamming",
        use_energy=False
    )

    # result shape: [T, 80] → transpose to [80, T]
    print(fbank.shape)
    return fbank.numpy()   # shape: [T, 80]




# =========================================================
# EMBEDDING
# =========================================================
def get_embedding(wav, sr):
    fbank = wav_to_fbank(wav, sr)
    inp = fbank[np.newaxis, :, :].astype(np.float32)  # [1, 80, T]

    emb = sess.run(None, {input_name: inp})[0].squeeze()
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb


# =========================================================
# COSINE
# =========================================================
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-9))


# =========================================================
# LOAD REFERENCE VOICE
# =========================================================
if not os.path.exists(VOICE_PATH):
    raise FileNotFoundError(f"Reference file not found: {VOICE_PATH}")

ref_audio, ref_sr = sf.read(VOICE_PATH)
ref_audio = ref_audio.astype(np.float32)

ref_emb = get_embedding(ref_audio, ref_sr)
print("Reference voice loaded. Embedding dim:", ref_emb.shape)


# =========================================================
# REAL-TIME BUFFER
# =========================================================
buffer_size = int(SR * BUFFER_SECONDS)
audio_buffer = deque(maxlen=buffer_size)

def callback(indata, frames, time_info, status):
    audio_buffer.extend(indata[:, 0].tolist())


# =========================================================
# VERIFICATION
# =========================================================
def verify():
    if len(audio_buffer) < SR:
        return None, None  # need 1 sec minimum

    segment = np.array(audio_buffer, dtype=np.float32)
    emb = get_embedding(segment, SR)
    score = cosine(ref_emb, emb)

    return score, score > THRESHOLD


# =========================================================
# MAIN LOOP
# =========================================================
print("\n🎤 Listening... Press Ctrl+C to stop.\n")

with sd.InputStream(
    channels=1,
    samplerate=SR,
    blocksize=BLOCK_SIZE,
    callback=callback
):
    try:
        while True:
            time.sleep(1)

            score, match = verify()
            if score is None:
                continue

            if match:
                print(f"✔ MATCH — score={score:.3f}")
            else:
                print(f"✘ NO MATCH — score={score:.3f}")

    except KeyboardInterrupt:
        print("\nStopped.")
