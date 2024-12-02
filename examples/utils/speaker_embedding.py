import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from resemblyzer import VoiceEncoder, preprocess_wav
from tqdm import tqdm
import librosa
from pathlib import Path
from typing import Union, Optional

int16_max = (2 ** 15) - 1
sampling_rate = 16000  # Assuming 16kHz is the target sample rate
audio_norm_target_dBFS = -30  # Target dBFS for normalization

def preprocess_wav_no_trim(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int]=None):
    """
    Applies preprocessing operations to a waveform either on disk or in memory.
    This version does not trim silences, only resamples and normalizes the waveform.

    :param fpath_or_wav: either a filepath to an audio file (many extensions are supported, not 
    just .wav), or a waveform as a numpy array of floats.
    :param source_sr: if passing a waveform, the sampling rate of the waveform before 
    preprocessing. After preprocessing, the waveform's sample rate will match the data 
    hyperparameters. If passing a filepath, the sampling rate will be automatically detected.
    """
    # Load the wav from disk if needed
    if isinstance(fpath_or_wav, str) or isinstance(fpath_or_wav, Path):
        wav, source_sr = librosa.load(str(fpath_or_wav), sr=None)
    else:
        wav = fpath_or_wav
    
    # Resample the wav
    if source_sr is not None:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)
        
    # Apply the preprocessing: normalize volume (no silence trimming)
    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    
    return wav


def normalize_volume(wav, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))


def generate_speaker_embedding_tensor(waveform_tensors, speaker_encoder):
    # Ensure waveform_tensors is a valid list of tensors
    if not isinstance(waveform_tensors, list) or not all(isinstance(waveform, torch.Tensor) for waveform in waveform_tensors):
        raise ValueError("waveform_tensors must be a list of torch.Tensor instances.")
    
    if not waveform_tensors:
        raise ValueError("No valid waveform tensors found for speaker enrollment.")
    # print(waveform_tensors)
    # Preprocess each tensor in the list
    waveforms_preprocessed = []
    for waveform in waveform_tensors:
        # Assuming preprocess_wav function accepts a numpy array
        # print(waveform)
        waveform_preprocessed = preprocess_wav(waveform[0].numpy())
        # print(waveform_preprocessed)
        waveforms_preprocessed.append(waveform_preprocessed)

    # Generate embedding using the preprocessed waveforms
    embedding = speaker_encoder.embed_speaker(waveforms_preprocessed)
    # print("embedding", embedding)
    return torch.from_numpy(embedding)

def generate_speaker_embedding_tensor_pretrained(waveform_tensors, speaker_encoder):
    # Ensure waveform_tensors is a valid list of tensors
    if not isinstance(waveform_tensors, list) or not all(isinstance(waveform, torch.Tensor) for waveform in waveform_tensors):
        raise ValueError("waveform_tensors must be a list of torch.Tensor instances.")
    
    if not waveform_tensors:
        raise ValueError("No valid waveform tensors found for speaker enrollment.")
    
    # Preprocess each tensor in the list
    waveforms_preprocessed = []
    for waveform in waveform_tensors:
        # Assuming preprocess_wav function accepts a numpy array
        waveform_preprocessed = preprocess_wav(waveform[0].numpy())
        waveforms_preprocessed.append(waveform_preprocessed)

    # Check if the speaker_encoder expects tensors or numpy arrays, assuming it processes them in batches
    embeddings = []
    for preprocessed_waveform in waveforms_preprocessed:
        # Convert the preprocessed waveform back to tensor if needed
        waveform_tensor = torch.tensor(preprocessed_waveform)
        
        # Assuming the new speaker_encoder processes tensors directly
        embedding = speaker_encoder(waveform_tensor.unsqueeze(0))  # Add batch dimension if necessary
        embeddings.append(embedding)

    # Stack all embeddings into a single tensor
    return torch.stack(embeddings)


def generate_speaker_embedding(speaker, speaker_encoder):
    # Ensure speaker is a valid directory
    if not os.path.isdir(speaker):
        raise ValueError(f"{speaker} is not a valid directory")
    
    with os.scandir(speaker) as sessions:
        files = [session.path for session in sessions if session.is_file() and session.name.endswith('.wav')]
        print(f"Found {len(files)} files in the directory.")  # Debugging line

        if not files:
            raise ValueError("No valid audio files found for speaker enrollment.")

        # Process each audio file in the directory
        waveforms_preprocessed = []
        for file in files:
            waveform, sample_rate = torchaudio.load(file)
            # print(waveform)
            waveform_preprocessed = preprocess_wav(waveform[0].numpy())
            waveforms_preprocessed.append(waveform_preprocessed)

        # Generate embedding using the preprocessed waveforms
        embedding = speaker_encoder.embed_speaker(waveforms_preprocessed)
        
    return torch.from_numpy(embedding)

def generate_embeddings(libri_root: str, split: str = "dev-clean", save_dir: str = "embeddings"):
    speaker_encoder = VoiceEncoder()
    Path(save_dir).mkdir(parents=True, exist_ok=False)
    with os.scandir(libri_root + split) as speakers:
        speakers_list = list(speakers)
        n_speakers = len(speakers_list)
        embeddings = {}
        for speaker in tqdm(speakers_list, unit=" embeddings", total=n_speakers):
            if os.path.isdir(speaker.path) and not speaker.name.startswith("."):
                embedding = generate_speaker_embedding(speaker, speaker_encoder)
                embeddings[speaker.name] = embedding

    torch.save(embeddings, f=save_dir + "/speaker_embeddings.pt")


def load_speaker_embedding(speaker_id: int, embeddings_dir: str, split: str):
    return torch.load(embeddings_dir + split + "/" + str(speaker_id) + ".pt")



def compute_similarity(path, speaker_embedding, speaker_encoder, threshold):
    waveform, sr = torchaudio.load(path)
    wav_preprocessed = preprocess_wav_no_trim(waveform[0].numpy())

    _, partial_embeddings, _ = speaker_encoder.embed_utterance(wav_preprocessed,
                                                               return_partials=True,
                                                               min_coverage=0.5,
                                                               rate=2.5)

    similarity = torch.nn.functional.cosine_similarity(speaker_embedding, torch.from_numpy(partial_embeddings),
                                                       dim=-1).numpy()
    if np.isnan(similarity).any():
        print("found NaN in similarity scores")
        
    # Count the number of windows/samples above or at the threshold
    count_above_threshold = np.sum(similarity >= threshold)
    # print(f"Number of windows/samples at or above the threshold: {count_above_threshold}")
    
    torch.cuda.empty_cache()
    return similarity, count_above_threshold



def generate_similarity_scores(libri_concat_root: str, split: str = "dev-clean", waveforms_dir: str = "Waveforms",
                               speaker_embedding_dir: str = "SpeakerEmbeddings", save_dir: str = "SpeakerEmbeddings"):
    speaker_encoder = VoiceEncoder(device="cpu")
    waveform_paths = sorted(str(p) for p in Path(libri_concat_root + split + "/" + waveforms_dir).glob("**/*.flac"))
    embeddings_path = libri_concat_root + split + "/" + speaker_embedding_dir + "/speaker_embeddings.pt"
    metadata = pd.read_csv(libri_concat_root + split + "/metadata.csv").set_index('identifier')
    speaker_embeddings = torch.load(embeddings_path)
    for path in tqdm(waveform_paths, unit="utterances", total=len(waveform_paths)):
        identifier = Path(path).stem
        out_name = "/" + identifier + "_similarity"
        # if Path(libri_concat_root + split + "/" + save_dir + out_name + ".pt").exists():
        #     continue

        target_speaker_id = int(metadata.loc[identifier]["target_speaker_id"])
        speaker_embedding = speaker_embeddings[str(target_speaker_id)]
        similarity = compute_similarity(path, speaker_embedding, speaker_encoder=speaker_encoder)
        
        torch.save(similarity, f=libri_concat_root + split + "/" + save_dir + out_name + ".pt")

