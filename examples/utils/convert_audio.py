import os
import time
from pydub import AudioSegment
import numpy as np
import librosa
import random
import torch

# Define the path to the folder with audio samples and the output folder
input_folder = '/Users/SAI/Documents/Code/pvad/pvad-poc/on_target_samples/eepp_no_background/hey_zelda/single/epp_9_11_pvad5.wav'
output_folder = '/Users/SAI/Documents/Code/pvad/pvad-poc/on_target_samples/augmented_enrollment_sets/epp_9_12_temp'
background_noise_folder = '/Users/SAI/Documents/Code/pvad/pvad-poc/data/clean_augmentations/noise'
impulse_response_folder = '/Users/SAI/Documents/Code/pvad/pvad-poc/data/clean_augmentations/rir'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

def add_impulse_response(audio, ir_file, reverb_level=0.05):
    # start_time = time.time()  # Start timing

    # Load the impulse response with the same sample rate as the audio
    ir, sr = librosa.load(ir_file, sr=audio.frame_rate)

    # Convert audio to numpy array
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    if audio.sample_width == 2:
        audio_data = audio_data / 32768  # Convert 16-bit audio to float range (-1.0, 1.0)

    # Normalize and reduce the amplitude of the impulse response
    ir = librosa.util.normalize(ir) * reverb_level  # Lower scaling factor reduces reverb intensity

    # Perform convolution
    reverb_audio = np.convolve(audio_data, ir, mode='same')

    # Mix original (dry) and reverb (wet) audio
    mixed_audio = (1 - reverb_level) * audio_data + reverb_audio  # Adjust mixing ratio here

    # Ensure output is within 16-bit range
    mixed_audio = np.clip(mixed_audio, -1.0, 1.0) * 32767  # Scale back to 16-bit range

    # Convert the numpy array back to AudioSegment
    reverb_audio_segment = AudioSegment(
        data=mixed_audio.astype(np.int16).tobytes(),  # Convert the numpy array to 16-bit integer bytes
        sample_width=audio.sample_width,
        frame_rate=audio.frame_rate,
        channels=audio.channels
    )

    # duration = time.time() - start_time  # Calculate duration
    # print(f"Time taken for adding impulse response: {duration:.2f} seconds")
    return reverb_audio_segment

def add_background_noise(audio, noise_file, snr_db=10):
    # start_time = time.time()  # Start timing

    noise = AudioSegment.from_file(noise_file).set_frame_rate(16000).set_channels(1)
    if len(noise) > len(audio):
        noise = noise[:len(audio)]
    else:
        noise = noise + AudioSegment.silent(duration=(len(audio) - len(noise)))

    audio_power = audio.dBFS
    noise_power = noise.dBFS
    noise = noise + (audio_power - noise_power - snr_db)
    
    noisy_audio = audio.overlay(noise)

    # duration = time.time() - start_time  # Calculate duration
    # print(f"Time taken for adding background noise: {duration:.2f} seconds")
    return noisy_audio

def convert_and_augment_tensor(audio_tensor, output_folder = "/Users/SAI/Documents/Code/wakeWord/wakeWordForked/Untitled/examples/audio/augments", background_noise_folder = background_noise_folder, impulse_response_folder = impulse_response_folder, augment=True, save=False):
    load_start_time = time.time()  # Start timing for loading

    # Check or create the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Convert the input tensor to AudioSegment (assuming tensor is normalized and has shape [1, num_samples])
    audio_array = (audio_tensor.squeeze(0) * 32768).numpy().astype('int16')  # Denormalize back to int16
    audio = AudioSegment(
        audio_array.tobytes(), 
        frame_rate=16000, 
        sample_width=2,  # 2 bytes for int16
        channels=1
    )
    
    tensors = []  # List to hold tensors of original and augmented audio

    # Save the original 16k WAV version and convert to tensor
    if save:
        base_name = "original_tensor.wav"
        output_file = os.path.join(output_folder, base_name)
        save_start_time = time.time()
        audio.export(output_file, format='wav')
        save_duration = time.time() - save_start_time
        # print(f"Time taken for saving original audio: {save_duration:.2f} seconds")
    
    # Append the original tensor to the list
    tensors.append(audio_tensor)

    if augment:
        augment_start_time = time.time()

        # Add background noise
        if os.path.exists(background_noise_folder) and os.listdir(background_noise_folder):
            noise_file = os.path.join(background_noise_folder, random.choice(os.listdir(background_noise_folder)))
            noisy_audio = add_background_noise(audio, noise_file)
            noisy_output_file = os.path.join(output_folder, "noisy_tensor.wav")
            noisy_tensor = torch.tensor(noisy_audio.get_array_of_samples()).float().unsqueeze(0) / 32768.0  # Normalize
            tensors.append(noisy_tensor)
            if save:
                save_start_time = time.time()
                noisy_audio.export(noisy_output_file, format='wav')
                save_duration = time.time() - save_start_time
                print(f"Added background noise from {noise_file}, saving time: {save_duration:.2f} seconds")

        # Add reverb/impulse response
        if os.path.exists(impulse_response_folder) and os.listdir(impulse_response_folder):
            ir_file = os.path.join(impulse_response_folder, random.choice(os.listdir(impulse_response_folder)))
            reverb_audio = add_impulse_response(audio, ir_file)
            reverb_output_file = os.path.join(output_folder, "reverb_tensor.wav")
            reverb_tensor = torch.tensor(reverb_audio.get_array_of_samples()).float().unsqueeze(0) / 32768.0  # Normalize
            tensors.append(reverb_tensor)
            if save:
                save_start_time = time.time()
                reverb_audio.export(reverb_output_file, format='wav')
                save_duration = time.time() - save_start_time
                print(f"Added impulse response from {ir_file}, saving time: {save_duration:.2f} seconds")

        augment_duration = time.time() - augment_start_time
        print(f"Total augmentation time: {augment_duration:.2f} seconds")

    total_duration = time.time() - load_start_time  # Total duration from loading to save
    print(f"Total process time for input tensor: {total_duration:.2f} seconds")
    
    return tensors

def convert_and_augment(input_file, output_folder, background_noise_folder, impulse_response_folder, augment=False, save=False):
    # load_start_time = time.time()  # Start timing for loading

    # Check or create the output folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Construct the output file path from the input file name and the output folder
    base_name = os.path.basename(input_file)
    output_file = os.path.join(output_folder, base_name)

    # Load the audio file
    audio = AudioSegment.from_file(input_file).set_frame_rate(16000).set_channels(1)
    tensors = []  # List to hold tensors of original and augmented audio

    # Save the original 16k WAV version and convert to tensor
    if save:
        save_start_time = time.time()
        audio.export(output_file, format='wav')
        save_duration = time.time() - save_start_time
        # print(f"Time taken for saving original audio: {save_duration:.2f} seconds")
    
    # Convert original audio to tensor
    tensor = torch.tensor(audio.get_array_of_samples()).float().unsqueeze(0) / 32768.0  # Normalize audio data
    tensors.append(tensor)

    if augment:
        augment_start_time = time.time()

        # Add background noise
        if os.path.exists(background_noise_folder) and os.listdir(background_noise_folder):
            noise_file = os.path.join(background_noise_folder, random.choice(os.listdir(background_noise_folder)))
            noisy_audio = add_background_noise(audio, noise_file)
            noisy_output_file = os.path.splitext(output_file)[0] + '_noisy.wav'
            noisy_tensor = torch.tensor(noisy_audio.get_array_of_samples()).float().unsqueeze(0) / 32768.0  # Normalize
            tensors.append(noisy_tensor)
            if save:
                save_start_time = time.time()
                noisy_audio.export(noisy_output_file, format='wav')
                save_duration = time.time() - save_start_time
                # print(f"Added background noise from {noise_file} to {input_file}, saving time: {save_duration:.2f} seconds")

        # Add reverb/impulse response
        if os.path.exists(impulse_response_folder) and os.listdir(impulse_response_folder):
            ir_file = os.path.join(impulse_response_folder, random.choice(os.listdir(impulse_response_folder)))
            reverb_audio = add_impulse_response(audio, ir_file)
            reverb_output_file = os.path.splitext(output_file)[0] + '_reverb.wav'
            reverb_tensor = torch.tensor(reverb_audio.get_array_of_samples()).float().unsqueeze(0) / 32768.0  # Normalize
            tensors.append(reverb_tensor)
            if save:
                save_start_time = time.time()
                reverb_audio.export(reverb_output_file, format='wav')
                save_duration = time.time() - save_start_time
                # print(f"Added impulse response from {ir_file} to {input_file}, saving time: {save_duration:.2f} seconds")

        augment_duration = time.time() - augment_start_time
        # print(f"Total augmentation time: {augment_duration:.2f} seconds")

    # total_duration = time.time() - load_start_time  # Total duration from loading to save
    # print(f"Total process time for {input_file}: {total_duration:.2f} seconds")
    # print(tensors[0])
    return tensors