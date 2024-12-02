import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import time

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
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='onnx',
    required=False
)


args = parser.parse_args()

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    # owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework, custom_verifier_models={"hey_stryker_dipco_1":"/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/notebooks/openwakeword/openwakeword/resources/models/hey_stryker_epp_verifier (1).pkl"}, custom_verifier_threshold=0.3)
    owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework)
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Run capture loop continuously, checking for wakewords
if __name__ == "__main__":
    # Generate output string header
    print("\n\n")
    print("#"*100)
    print("Listening for wakewords...")
    print("#"*100)
    print("\n"*(n_models*3))
    patience = {} 
    threshold = {"hey_Zelda_8_15": 0.0637}  


    while True:
        # Get audio
        audio = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        # Measure inference time
        start_time = time.time()
        prediction = owwModel.predict(audio)
        # prediction = owwModel.predict(audio, debounce_time=1.5, threshold=threshold)
        inference_time = time.time() - start_time

        # Column titles
        n_spaces = 16
        output_string_header = f"""
            Model Name         | Score | Wakeword Status | Inference Time (s)
            ---------------------------------------------------------
            """

        wakeword_detected = False
        for mdl in owwModel.prediction_buffer.keys():
            # Add scores in formatted table
            scores = list(owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], '.20f').replace("-", "")

            status = "--" + " " * 20 if scores[-1] <= 0.05 else "Wakeword Detected!"
            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {status} | {inference_time:.6f}
            """
            if scores[-1] > 0.0637:
                wakeword_detected = True

        # Print results table
        print("\033[F"*(4*n_models+1))
        print(output_string_header, "                             ", end='\r')

        # Print additional message if wakeword is detected
        if wakeword_detected:
            print("\nWakeword detected! Saving activation event...\n")
