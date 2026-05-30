#!/usr/bin/env python3
"""
Dog Bark Detector
-----------------
Continuously listens on the microphone. When a dog bark is detected via
YAMNet (TensorFlow Lite), it:
  - Logs the event to a daily log file  (logs/YYYY-MM-DD.log)
  - Saves a 5-second WAV clip          (clips/YYYY-MM-DD_HH-MM-SS.wav)

Requirements (install once):
    pip install sounddevice numpy scipy tflite-runtime

YAMNet model (download once):
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/audio_classification/rpi/lite-model_yamnet_classification_tflite_1.tflite \
         -O yamnet.tflite

    wget https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv \
         -O yamnet_class_map.csv
"""

import os
import csv
import time
import queue
import datetime
import threading
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# ── Configuration ─────────────────────────────────────────────────────────────

SAMPLE_RATE       = 16000          # Hz — YAMNet expects 16 kHz
CHANNELS          = 1
CHUNK_DURATION    = 0.975          # seconds — YAMNet input window (~15 600 samples)
CLIP_DURATION     = 5              # seconds to save around each bark event
CONFIDENCE_THRESH = 0.3            # 0–1; lower = more sensitive
COOLDOWN_SECS     = 4              # minimum gap between logged events (avoids spam)

MODEL_PATH        = "yamnet.tflite"
CLASS_MAP_PATH    = "yamnet_class_map.csv"

LOGS_DIR          = "logs"
CLIPS_DIR         = "clips"

# ── Bark-related class names in YAMNet's class map ────────────────────────────
BARK_LABELS = {
    "Dog",
    "Bark",
    "Bow-wow",
    "Growling",
    "Whimper (dog)",
}

# ── Setup ─────────────────────────────────────────────────────────────────────

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

# Load class labels
def load_class_names(path):
    names = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[int(row["index"])] = row["display_name"]
    return names

# Load TFLite model
def load_model(path):
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        # Fallback to full TensorFlow if tflite_runtime not available
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

# ── Inference ─────────────────────────────────────────────────────────────────

def run_yamnet(interpreter, audio_chunk, class_names):
    """
    Run YAMNet on a mono float32 audio chunk.
    Returns list of (label, score) tuples for bark-related classes.
    """
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # YAMNet expects shape [1, num_samples] float32 in [-1, 1]
    waveform = audio_chunk.astype(np.float32)
    if waveform.max() > 1.0:
        waveform = waveform / 32768.0          # convert int16 → float

    expected_len = input_details[0]["shape"][1]
    if len(waveform) < expected_len:
        waveform = np.pad(waveform, (0, expected_len - len(waveform)))
    else:
        waveform = waveform[:expected_len]

    interpreter.set_tensor(input_details[0]["index"], waveform[np.newaxis, :])
    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]["index"])[0]   # shape [521]
    hits = [
        (class_names[i], float(scores[i]))
        for i in range(len(scores))
        if class_names.get(i, "") in BARK_LABELS and scores[i] >= CONFIDENCE_THRESH
    ]
    return hits

# ── Logging ───────────────────────────────────────────────────────────────────

def log_event(label, score, clip_filename):
    today     = datetime.date.today().isoformat()
    log_path  = os.path.join(LOGS_DIR, f"{today}.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] BARK DETECTED — label: '{label}', confidence: {score:.2f}, clip: {clip_filename}\n"
    with open(log_path, "a") as f:
        f.write(line)
    print(line, end="")

# ── Clip saving ───────────────────────────────────────────────────────────────

def save_clip(ring_buffer, clip_filename):
    """Flatten ring buffer into a WAV file."""
    audio = np.concatenate(list(ring_buffer), axis=0)
    clip_path = os.path.join(CLIPS_DIR, clip_filename)
    # Convert float32 → int16 for WAV
    audio_int16 = (audio * 32767).astype(np.int16)
    wav.write(clip_path, SAMPLE_RATE, audio_int16)

# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("Loading YAMNet model …")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Download it with:\n"
            "  wget https://storage.googleapis.com/download.tensorflow.org/models/"
            "tflite/task_library/audio_classification/rpi/"
            "lite-model_yamnet_classification_tflite_1.tflite -O yamnet.tflite"
        )
    if not os.path.exists(CLASS_MAP_PATH):
        raise FileNotFoundError(
            f"Class map not found: {CLASS_MAP_PATH}\n"
            "Download it with:\n"
            "  wget https://raw.githubusercontent.com/tensorflow/models/master/"
            "research/audioset/yamnet/yamnet_class_map.csv -O yamnet_class_map.csv"
        )

    interpreter  = load_model(MODEL_PATH)
    class_names  = load_class_names(CLASS_MAP_PATH)
    chunk_frames = int(SAMPLE_RATE * CHUNK_DURATION)
    clip_chunks  = int(CLIP_DURATION / CHUNK_DURATION)   # how many chunks = 5 s

    # Ring buffer holds the last `clip_chunks` audio chunks (≈ 5 seconds)
    from collections import deque
    ring_buffer = deque(maxlen=clip_chunks)

    audio_queue     = queue.Queue()
    last_event_time = 0.0

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Audio status:", status)
        audio_queue.put(indata[:, 0].copy())   # keep mono

    print(f"Listening … (sample rate {SAMPLE_RATE} Hz, chunk {CHUNK_DURATION:.3f}s)")
    print(f"Logs → {LOGS_DIR}/   Clips → {CLIPS_DIR}/")
    print("Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=chunk_frames,
        callback=audio_callback,
    ):
        while True:
            chunk = audio_queue.get()
            ring_buffer.append(chunk)

            hits = run_yamnet(interpreter, chunk, class_names)
            if not hits:
                continue

            now = time.time()
            if now - last_event_time < COOLDOWN_SECS:
                continue                         # within cooldown window, skip

            last_event_time = now
            best_label, best_score = max(hits, key=lambda x: x[1])
            ts          = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            clip_name   = f"{ts}.wav"

            # Save clip in a background thread so we don't miss audio
            threading.Thread(
                target=save_clip,
                args=(ring_buffer.copy(), clip_name),
                daemon=True,
            ).start()

            log_event(best_label, best_score, clip_name)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")