import numpy as np
from openwakeword.model import Model
import argparse
import time
import wave
# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="wakeword_models/hey_zelda/hey_Zelda_8_15.onnx",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite')",
    type=str,
    default='onnx',
    required=False
)
parser.add_argument(
    "--audio_path",
    help="Path to the input audio file (.wav format) to be processed",
    type=str,
    default="/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/examples/Hey_Zelda.wav",
    required=False
)

args = parser.parse_args()

# Load the audio file
audio_file = wave.open(args.audio_path, 'rb')
RATE = audio_file.getframerate()
CHANNELS = audio_file.getnchannels()

# Ensure the audio file is mono
if CHANNELS != 1:
    raise ValueError("The input audio file must be mono.")

# Load pre-trained openwakeword models
owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework) if args.model_path else Model(inference_framework=args.inference_framework)
n_models = len(owwModel.models.keys())

# Initialize time marker for total length into audio stream
cumulative_audio_time = 0
CHUNK = args.chunk_size

patience = {}  # e.g., {"model_name": 3} if using patience, or leave as empty if not
threshold = {"hey_Zelda_8_15": 0.0637}  # Replace "hey_Zelda_model" with the actual model name in your setup

# Measure inference time
predictions = owwModel.predict_clip(args.audio_path, debounce_time=1.5,padding=1, threshold=threshold)
# predictions = owwModel.predict_clip(args.audio_path)
# Filter and print only positive detections above the threshold
for i, prediction in enumerate(predictions):
    for model_name, score in prediction.items():
        if score > 0.06:  # Print only scores above 0.1
            time_in_seconds = i * (args.chunk_size / RATE)
            print(f"Positive detection for {model_name} at {time_in_seconds:.2f} seconds with score: {score:.4f}")
