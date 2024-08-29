import numpy as np
import matplotlib.pyplot as plt
from openwakeword.model import Model
import scipy.io.wavfile
import argparse
import os
import time

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dirs",
    help="Comma-separated list of directories containing the WAV files to process",
    type=str,
    required=False
)
parser.add_argument(
    "--false_positive_dir",
    help="Directory containing the WAV files to calculate false positives per hour",
    type=str,
    required=False
)
parser.add_argument(
    "--threshold",
    help="The score threshold for an activation",
    type=float,
    default=0.5,
    required=False
)
parser.add_argument(
    "--vad_threshold",
    help="The threshold to use for voice activity detection (VAD) in the openWakeWord instance.",
    type=float,
    default=0.0,
    required=False
)
parser.add_argument(
    "--noise_suppression",
    help="Whether to enable speex noise suppression in the openWakeWord instance.",
    type=bool,
    default=False,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/wakeword_models/hey_zelda/hey_Zelda_8_15.onnx",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite')",
    type=str,
    default='onnx',
    required=False
)

args = parser.parse_args()

# Load pre-trained openwakeword model
owwModel = Model(
    wakeword_models=[args.model_path], 
    enable_speex_noise_suppression=args.noise_suppression,
    vad_threshold=args.vad_threshold,
    inference_framework=args.inference_framework
)

# Warm-up the model with a dummy input
dummy_audio = np.zeros((5 * 16000,), dtype=np.int16)  # 5 seconds of silence
for _ in range(10):  # Run 10 dummy inferences
    owwModel.predict(dummy_audio)

# Access the large single audio file
file_path = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/examples/audio/false_positive/Zelda_FAR_output_file_16k.wav"  # Provide the correct path
sample_rate, mic_audio = scipy.io.wavfile.read(file_path)

# Ensure audio is in the correct format (flatten and ensure int16 dtype)
mic_audio = mic_audio.flatten().astype(np.int16)

# Parameters for sliding window and cooldown
window_size_seconds = 2  # Sliding window size in seconds
window_size = window_size_seconds * sample_rate  # Number of samples per window
step_size_seconds = 0.1  # How much to slide the window by, in seconds (e.g., 100ms)
step_size = int(step_size_seconds * sample_rate)  # Convert step size to samples

cooldown_seconds = 5  # Cooldown period after an activation (in seconds)
cooldown_samples = cooldown_seconds * sample_rate  # Convert cooldown to samples
last_activation = -cooldown_samples  # Track the index of the last activation

# Variables for tracking activations
all_max_scores_fp = []
activation_times = []

# Iterate over the audio with a sliding window
for start_idx in range(0, len(mic_audio) - window_size + 1, step_size):
    # Check if we are still in cooldown period
    if start_idx < last_activation + cooldown_samples:
        continue

    end_idx = start_idx + window_size
    audio_window = mic_audio[start_idx:end_idx]

    # Ensure the audio window is flattened and in the correct format
    audio_window = audio_window.flatten().astype(np.int16)

    # Predict using the openWakeWord model
    prediction = owwModel.predict(audio_window)
    max_score = max(prediction.values())

    all_max_scores_fp.append(max_score)

    # Check if the activation threshold is reached
    if max_score >= args.threshold:
        activation_time_seconds = start_idx / sample_rate  # Store activation time in seconds
        activation_times.append(activation_time_seconds)
        print(f"Activation detected at {activation_time_seconds:.2f} seconds")

        # Set the last activation index to enforce cooldown
        last_activation = start_idx

# Calculate the number of activations
false_positive_count = len(activation_times)

# Print or log the number of activations
print(f"Total number of activations: {false_positive_count}")

# Optionally, save the activation times for further analysis
output_dir = "output_directory"  # Replace with your output directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(os.path.join(output_dir, "activation_times.txt"), "w") as f:
    for activation_time in activation_times:
        f.write(f"{activation_time:.2f}\n")
