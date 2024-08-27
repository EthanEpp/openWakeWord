import os
import collections
import time
import numpy as np
from openwakeword.model import Model
import scipy.io.wavfile
import datetime
import argparse
import subprocess

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_dir",
    help="Where to save the audio that resulted in an activation",
    type=str,
    default="positivedetections/meeting_test/orin_device/8_26_zelda",
    required=False
)
parser.add_argument(
    "--threshold",
    help="The score threshold for an activation",
    type=float,
    default=0.0015,
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
    "--chunk_size",
    help="How much audio (in number of 16khz samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    # default="wakeword_models/voice_genie/voice_genie_v01.onnx",
    default="/root/ai_test_eebs/EEww/Untitled/wakeword_models/hey_zelda/hey_Zelda_8_15.onnx",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='onnx',
    required=False
)
parser.add_argument(
    "--disable_activation_sound",
    help="Disables the activation sound, clips are silently captured",
    action='store_true',
    required=False
)

args = parser.parse_args()

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(
        wakeword_models=[args.model_path], 
        enable_speex_noise_suppression=args.noise_suppression,
        vad_threshold=args.vad_threshold,
        inference_framework=args.inference_framework
    )
else:
    owwModel = Model(inference_framework=args.inference_framework)

# Set waiting period after activation before saving clip (to get some audio context after the activation)
save_delay = 1  # seconds

# Set cooldown period before another clip can be saved
cooldown = 4  # seconds

# Create output directory if it does not already exist
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Initialize arecord subprocess
command = [
    'arecord',
    '-D', 'record_left_16k',  # Specify the device
    '-r', '16000',  # Sample rate
    '-c', '1',  # Number of channels
    '-f', 'S16_LE',  # Format
    '-t', 'raw'  # Output as raw audio
]

# Open a subprocess to read the audio stream
process = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1024)

# Run capture loop, checking for hotwords
if __name__ == "__main__":
    # Predict continuously on audio stream
    last_save = time.time()
    activation_times = collections.defaultdict(list)

    print("\n\nListening for wakewords...\n")

    while True:
        # Read raw audio data from arecord
        raw_data = process.stdout.read(args.chunk_size * 2)  # 2 bytes per sample for S16_LE

        # Convert raw data to numpy array
        mic_audio = np.frombuffer(raw_data, dtype=np.int16)

        # Feed to openWakeWord model
        prediction = owwModel.predict(mic_audio)

        # Display prediction scores in real time, updating the same line
        for mdl in prediction.keys():
            score = prediction[mdl]
            print(f"Prediction score for model \"{mdl}\": {score:.4f}", end='\r')

            # Check for model activations (score above threshold), and save clips
            if score >= args.threshold:
                activation_times[mdl].append(time.time())

            if activation_times.get(mdl) and (time.time() - last_save) >= cooldown \
               and (time.time() - activation_times.get(mdl)[0]) >= save_delay:
                last_save = time.time()
                activation_times[mdl] = []
                detect_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

                print(f'Detected activation from \"{mdl}\" model at time {detect_time}!', end='\r')

                # Capture total of 5 seconds, with the microphone audio associated with the
                # activation around the ~4 second point
                audio_context = np.array(list(owwModel.preprocessor.raw_data_buffer)[-16000*5:]).astype(np.int16)
                fname = detect_time + f"_{mdl}.wav"
                scipy.io.wavfile.write(os.path.join(os.path.abspath(args.output_dir), fname), 16000, audio_context)

                if not args.disable_activation_sound:
                    print("Detected", end='\r')
