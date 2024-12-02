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

# Read audio data in chunks and process
print("\n\n")
print("#"*100)
print("Processing audio file for wakewords...")
print("#"*100)
print("\n"*(n_models*3))

wakeword_detected = False

while True:
    # Read the next chunk of audio
    audio_data = audio_file.readframes(CHUNK)
    if len(audio_data) == 0:  # End of file
        break

    # Convert audio data to numpy array
    audio = np.frombuffer(audio_data, dtype=np.int16)
    cumulative_audio_time += CHUNK / RATE  # Update time in seconds
    patience = {"hey_Zelda_8_15": 3}  # e.g., {"model_name": 3} if using patience, or leave as empty if not
    threshold = {"hey_Zelda_8_15": 0.0637}  # Replace "hey_Zelda_model" with the actual model name in your setup

    # Measure inference time
    start_time = time.time()
    prediction = owwModel.predict(audio, patience=patience, threshold=threshold)
    # prediction = owwModel.predict(audio)
    inference_time = time.time() - start_time

    # Column titles
    n_spaces = 16
    output_string_header = f"""
        Model Name         | Score | Wakeword Status | Inference Time (s) | Time into Sample (s)
        ---------------------------------------------------------
        """

    for mdl in owwModel.prediction_buffer.keys():
        # Add scores in formatted table
        scores = list(owwModel.prediction_buffer[mdl])
        curr_score = format(scores[-1], '.20f').replace("-", "")

        status = "--" + " " * 20 if scores[-1] <= 0.05 else "Wakeword Detected!"
        output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {status} | {inference_time:.6f} | {cumulative_audio_time:.2f}
        """
        
        # Check for wakeword detection and output the time
        if scores[-1] > 0.0637:
            wakeword_detected = True
            print(f"\nWakeword detected! Detected at {cumulative_audio_time:.2f} seconds into the audio file.\n")

    # Print results table
    print("\033[F"*(4*n_models+1))
    print(output_string_header, "                             ", end='\r')

# Close the audio file
audio_file.close()

if not wakeword_detected:
    print("\nNo wakeword detected in the audio file.\n")
