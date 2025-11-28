# preprocess.py
import numpy as np
import librosa

SAMPLE_RATE = 16000

def compute_mfcc_from_array(wav, sr=SAMPLE_RATE, n_mfcc=40, max_frames=200):
    mf = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=160)
    mf = mf.T  # frames x n_mfcc
    if mf.shape[0] >= max_frames:
        mf = mf[:max_frames, :]
    else:
        pad = np.zeros((max_frames - mf.shape[0], mf.shape[1]), dtype=np.float32)
        mf = np.vstack([mf, pad])
    # return shape (n_mfcc, max_frames)
    return mf.T.astype(np.float32)
