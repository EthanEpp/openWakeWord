import numpy as np
import matplotlib.pyplot as plt
from openwakeword.model import Model
import scipy.io.wavfile
import argparse
import os
import mplcursors  # Import mplcursors for interactivity

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dirs",
    help="Comma-separated list of directories containing the WAV files to process",
    type=str,
    default="examples/audio/hey_zelda/no_mask/no_back/,examples/audio/hey_zelda/no_mask/med_back/,examples/audio/hey_zelda/no_mask/high_back/,examples/audio/hey_zelda/no_mask/max_back/",
    required=False
)
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
    default=0.5,
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
    default="/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/wakeword_models/hey_zelda/hey_Zelda_8_15.onnx",
    # default="/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/wakeword_models/hey_zelda/hey_Zelda_med_multi_phrase.onnx",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
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

thresholds = np.logspace(-3.2, 0, num=100)

# Warm-up the model with a dummy input
dummy_audio = np.zeros((5 * 16000, ), dtype=np.int16)  # 5 seconds of silence
for _ in range(10):  # Run 10 dummy inferences
    owwModel.predict(dummy_audio)

# Split the input directories
input_dirs = args.input_dirs.split(',')

# Prepare to store detection counts for each input directory
all_false_rejects = []
false_positive_counts = []

# Process each directory for false reject metrics
for input_dir in input_dirs:
    all_max_scores = []
    total_files = 0

    # Walk through each directory and its subdirectories
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith(".wav"):
                total_files += 1
                filepath = os.path.join(root, filename)
                sample_rate, mic_audio = scipy.io.wavfile.read(filepath)

                # Slice the audio to the first 5 seconds
                five_second_length = 5 * sample_rate  # Number of samples in 5 seconds
                mic_audio = mic_audio[:five_second_length]

                # Feed to openWakeWord model
                prediction = owwModel.predict(mic_audio)

                # Record the highest prediction score
                max_score = max(prediction.values())
                all_max_scores.append(max_score)

    # Calculate false rejects for each threshold
    total_activations = len(all_max_scores)
    # false_rejects = [total_activations - sum(1 for score in all_max_scores if score >= threshold) for threshold in thresholds]
    false_rejects = [(total_activations - sum(1 for score in all_max_scores if score >= threshold)) / total_activations * 100 for threshold in thresholds]
    all_false_rejects.append(false_rejects)

# Process the false positive directory for raw false positive acceptance
all_max_scores_fp = []

for root, _, files in os.walk(args.false_positive_dir):
    for filename in files:
        if filename.endswith(".wav"):
            filepath = os.path.join(root, filename)
            sample_rate, mic_audio = scipy.io.wavfile.read(filepath)

            # Slice the audio into two segments: 0-5 seconds and 5-10 seconds
            five_second_length = 10 * sample_rate  # Number of samples in 5 seconds
            first_segment = mic_audio[:five_second_length]

            # Process the first 5 seconds
            prediction_first = owwModel.predict(mic_audio)
            max_score_first = max(prediction_first.values())

            max_score = max_score_first
            all_max_scores_fp.append(max_score)

# Calculate false positive counts for each threshold
false_positive_counts = [sum(1 for score in all_max_scores_fp if score >= threshold) * (60/330) for threshold in thresholds]

# Define a variable for the plot name
plot_name = 'False Accepts vs False Rejects In House No mask'

# Plot False Accepts vs False Rejects
fig, ax = plt.subplots(figsize=(10, 6))
lines = []
fig.canvas.manager.set_window_title(plot_name)

for idx, false_rejects in enumerate(all_false_rejects):
    dir_name = os.path.basename(input_dirs[idx].rstrip('/'))
    line, = ax.plot(false_positive_counts, false_rejects, marker='o', label=dir_name)
    lines.append(line)

# Set up axes
ax.set_xlabel('False Accepts per Hour')
ax.set_ylabel('False Rejects Rate (%)')

# Use the plot_name variable for the title
ax.set_title(plot_name)
ax.grid(True)

# Legend handling
ax.legend(lines, [line.get_label() for line in lines], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout(rect=[0, 0, 1, 1])  # Adjust layout to make room for the legend

# Create a cursor for interactive hovering
cursor = mplcursors.cursor(lines, hover=True)

# Customize hover annotations to show all y-values at a hovered x-value
@cursor.connect("add")
def on_add(sel):
    x = sel.target[0]
    annotations = []
    for idx, false_rejects in enumerate(all_false_rejects):
        label = os.path.basename(input_dirs[idx].rstrip('/'))
        # Find the index of the closest false accepts value to x
        closest_idx = np.argmin(np.abs(false_positive_counts - x))
        y_value = false_rejects[closest_idx]
        corresponding_threshold = thresholds[closest_idx]
        annotations.append(f"{label}: {y_value:.0f} False Reject Rate (Threshold: {corresponding_threshold:.4f})")
    sel.annotation.set_text(f"False Accepts = {x:.0f}\n" + "\n".join(annotations))

# Print the plot name if needed
print(f"Generating plot: {plot_name}")

plt.show()
