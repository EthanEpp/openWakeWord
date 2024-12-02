import torch
import numpy as np
from librosa.feature import melspectrogram

# def extract_features(waveform, sampling_rate, similarity, feature_extractor):
#     waveform, sampling_rate,  similarity,

#     waveform = (waveform - waveform.mean()) / (waveform.std() + 1.e-9)
#     features = feature_extractor(waveform)
#     sampling_rate = round(sampling_rate / feature_extractor.hop_length)

#     # Cutoff similarity score to feature length
#     similarity = similarity[:features.size(-1)]

#     return features, similarity


def compute_similarity_single_frame(waveform, speaker_embedding, speaker_encoder):
    # Ensure waveform is in a 1-D numpy array
    waveform_copy = waveform.numpy().copy() if isinstance(waveform, torch.Tensor) else waveform
    waveform_copy = waveform_copy.squeeze()
    
    # Verify that waveform contains data
    if waveform_copy.size == 0:
        raise ValueError("Empty waveform received. Ensure the audio data is correctly processed.")
    
    # Convert waveform to mel spectrogram for a single frame
    features = melspectrogram(y=waveform_copy, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype('float32').T
    features_frame = features[:160]  # Adjust this to process just one frame
    
    features_frame = np.expand_dims(features_frame, axis=0)  # Add batch dimension
    
    with torch.no_grad():
        # Pass through the encoder for embedding
        embedding_frame = speaker_encoder(torch.from_numpy(features_frame)).numpy()
    
    # Compute cosine similarity
    score = cos_sim(speaker_embedding, embedding_frame[0])
    
    return score

def compute_similarity_multiple_frames(waveform, speaker_embedding, speaker_encoder, num_frames=160):
    # Ensure waveform is in a 1-D numpy array
    waveform_copy = waveform.numpy().copy() if isinstance(waveform, torch.Tensor) else waveform
    waveform_copy = waveform_copy.squeeze()

    # Convert waveform to mel spectrogram
    features = melspectrogram(y=waveform_copy, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype('float32').T
    features_frames = features[:num_frames]  # Use multiple frames

    features_frames = np.expand_dims(features_frames, axis=0)  # Add batch dimension

    with torch.no_grad():
        # Pass through encoder for embedding
        embedding = speaker_encoder(torch.from_numpy(features_frames)).numpy()
    
    # Compute cosine similarity
    score = cos_sim(speaker_embedding, embedding[0])
    
    return score


def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_similarity_single_slice(slice_waveform, speaker_embedding, speaker_encoder):
    # Convert waveform to a numpy array if needed
    waveform_copy = slice_waveform.numpy().copy() if isinstance(slice_waveform, torch.Tensor) else slice_waveform
    
    # Compute melspectrogram features for the single slice
    features = melspectrogram(y=waveform_copy, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype('float32').T
    
    # Compute embedding for the slice with the speaker encoder
    with torch.no_grad():
        embedding_slice = speaker_encoder(torch.from_numpy(features[np.newaxis, :, :])).numpy()
    
    # Calculate the similarity score between the speaker embedding and the slice embedding
    similarity_score = cos_sim(speaker_embedding, embedding_slice[0])
    
    # Convert similarity score to tensor for consistency
    return torch.tensor([similarity_score], dtype=torch.float32)


def compute_similarity2(waveform, speaker_embedding, speaker_encoder):
    rate = 2.5
    min_coverage = 0.5
    waveform_copy = waveform[0].numpy().copy()
    
    # Duplicate the initial 160 frames for padding
    initial_160_frames = waveform_copy[:160 * 160]
    waveform_copy = np.concatenate((initial_160_frames, waveform_copy), axis=0)
    
    wav_slices, mel_slices = speaker_encoder.compute_partial_slices(waveform_copy.size,
                                                                    rate=rate,
                                                                    min_coverage=min_coverage)
    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= waveform_copy.size:
        waveform_copy = np.pad(waveform_copy, (0, max_wave_length - waveform_copy.size), "constant")
    
    features = melspectrogram(y=waveform_copy, sr=16000, n_fft=400, hop_length=160, n_mels=40).astype('float32').T
    features_sliced = np.array([features[s] for s in mel_slices])
    
    with torch.no_grad():
        embedding_slices = speaker_encoder(torch.from_numpy(features_sliced)).numpy()

    scores_slices = np.array([cos_sim(speaker_embedding, partial_embedding) for partial_embedding in embedding_slices])

    # Scores, linearly interpolated, starting from 0.5 every time
    frame_step = int(np.round((16000 / rate) / 160))
    
    # First 160 frames are score one (160 frames window)
    scores_linear_interpolated = np.kron(scores_slices[0], np.ones(160, dtype='float32'))
    
    # The rest are linearly interpolated
    for i, s in enumerate(scores_slices[1:]):
        scores_linear_interpolated = np.append(scores_linear_interpolated,
                                               np.linspace(scores_slices[i], s, frame_step, endpoint=False))

    # Remove the first 160 predictions (corresponding to the initial duplicated padding)
    similarity = torch.from_numpy(scores_linear_interpolated[160:])
    
    return similarity
