import numpy as np
import matplotlib.pyplot as plt
from openwakeword.model import Model
import scipy.io.wavfile
import argparse
import os

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--false_positive_dir",
    help="Directory containing the WAV files to calculate false positives per hour",
    type=str,
    default="examples/audio/false_positive",
    required=False
)
parser.add_argument(
    "--threshold",
    help="The score threshold for an activation",
    type=float,
    default=0.1,
    required=False
)
parser.add_argument(
    "--vad_threshold",
    help="""The threshold to use for voice activity detection (VAD) in the openWakeWord instance.
            The default (0.0), disables VAD.""",
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
    default="/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/wakeword_models/voice_genie/voice_genie_v01.onnx",
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

# Define thresholds range
thresholds = np.logspace(-3.2, 0, num=100)

# Warm-up the model with a dummy input
dummy_audio = np.zeros((5 * 16000,), dtype=np.int16)  # 5 seconds of silence
for _ in range(10):  # Run 10 dummy inferences
    owwModel.predict(dummy_audio)

# Prepare to store false positive scores
all_max_scores_fp = []
count_clips = 0
# Process the false positive directory for raw false positive acceptance
for root, _, files in os.walk(args.false_positive_dir):
    for filename in files:
        if filename.endswith(".wav"):
            filepath = os.path.join(root, filename)
            sample_rate, mic_audio = scipy.io.wavfile.read(filepath)
            print(root, filename)
            # Slice the audio into 5-second segments (each test clip is 5 seconds long)
            five_second_length = 5 * sample_rate  # Number of samples in 5 seconds
            mic_audio = mic_audio[:five_second_length]

            # Feed to openWakeWord model
            prediction = owwModel.predict(mic_audio)

            # Record the highest prediction score for this clip
            max_score = max(prediction.values())
            all_max_scores_fp.append(max_score)
            count_clips += 1
            print(count_clips)

# Calculate false positive counts for each threshold
# Note: The calculation multiplies by (60/) because 5 seconds per clip means 12 clips make up 1 minute
false_positive_counts = [sum(1 for score in all_max_scores_fp if score >= threshold) * (720 / count_clips) for threshold in thresholds]

# Define a variable for the plot name
plot_name = 'False Accepts per Hour'

# Plot False Positive Accepts per Hour
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(thresholds, false_positive_counts, marker='o', label='False Positives')

# Set up axes
ax.set_xlabel('Threshold')
ax.set_ylabel('False Accepts per Hour')
ax.set_xscale('log')  # Logarithmic scale for thresholds

# Use the plot_name variable for the title
ax.set_title(plot_name)
ax.grid(True)

# Display the plot
plt.tight_layout()

# Print the plot name if needed
print(f"Generating plot: {plot_name}")

plt.show()
