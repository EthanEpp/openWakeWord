import pyaudio
import torch
import numpy as np
from resemblyzer import VoiceEncoder
from time import sleep, time
from collections import deque
from utils.convert_audio import convert_and_augment, convert_and_augment_tensor
from utils.speaker_embedding import generate_speaker_embedding_tensor
from utils.capture_detections import compute_similarity_single_slice


# Initialize the speaker encoder and set it to evaluation mode
speaker_encoder = VoiceEncoder()
speaker_encoder.eval()

# Load and generate the target speaker embedding
warmup_file1 = '/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/examples/audio/epp_long/epp_long_16k.wav'
audiotens = convert_and_augment(warmup_file1, "examples/audio/augments", "examples/audio/clean_augmentations/noise", "examples/audio/clean_augmentations/rir", augment=False, save=False)
speaker_embedding2 = generate_speaker_embedding_tensor(audiotens, speaker_encoder)

# Audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Sample rate
CHUNK = 1280 * 4  # Adjust based on desired buffer size for real-time processing
BUFFER_SECONDS = 1  # We want the last 1 second of audio

# Set up microphone stream
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize a rolling buffer for the last 1 second of audio data
from collections import deque

# Initialize deque buffer with a fixed length
# audio_buffer = deque(maxlen=RATE * BUFFER_SECONDS)
audio_buffer = deque(maxlen=int(RATE * BUFFER_SECONDS))

# Initialize deque to store the last 5 similarity scores
similarity_scores = deque(maxlen=5)

try:
    print("Starting audio stream for similarity comparison...")
    while True:
        # Read chunk of audio data
        audio_data = mic_stream.read(CHUNK)
        
        # Convert to NumPy array and normalize
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Append new audio chunk to the buffer
        audio_buffer.extend(audio_np)
        
        # Only process once we have 1 second of audio in the buffer
        if len(audio_buffer) == RATE * BUFFER_SECONDS:
            # Convert buffer to a numpy array and pass it as the slice
            buffer_np = np.array(audio_buffer)
            
            # Compute similarity score for this slice using the new function
            similarity_score = compute_similarity_single_slice(buffer_np, speaker_embedding2, speaker_encoder)
            
            # Store the similarity score in the deque
            similarity_scores.append(similarity_score.item())
            
            # Calculate the average of the last 5 similarity scores
            if len(similarity_scores) == 5:
                avg_similarity = sum(similarity_scores) / len(similarity_scores)
                print(f"Average similarity score (last 5): {avg_similarity}")
            else:
                print(f"Similarity score: {similarity_score.item()}")
        
        # Add a smaller sleep interval to help maintain real-time processing
        sleep(0.1)

except KeyboardInterrupt:
    print("Audio streaming stopped.")
finally:
    # Close the stream and terminate the PyAudio instance
    mic_stream.stop_stream()
    mic_stream.close()
    audio.terminate()

