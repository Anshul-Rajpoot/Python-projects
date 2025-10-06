import numpy as np
import librosa
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa.display


class MFCCProcessor:
    def __init__(self, sr=16000, frame_size=25, frame_stride=10,
                 n_fft=512, n_mels=20, n_mfcc=13, pre_emphasis=0.97):
        """
        Initialize MFCC processor with default parameters
        """
        self.sr = sr
        self.frame_size_ms = frame_size      # in ms
        self.frame_stride_ms = frame_stride  # in ms
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.pre_emphasis_coeff = pre_emphasis

    # --------------------------------------------------
    #               SIGNAL PROCESSING
    # --------------------------------------------------
    def pre_emphasize(self, signal):
        """Apply pre-emphasis to boost high frequencies"""
        return np.append(signal[0], signal[1:] - self.pre_emphasis_coeff * signal[:-1])

    def frame_signal(self, signal, sr):
        """Split signal into overlapping frames"""
        frame_length = int(round(self.frame_size_ms * sr / 1000))
        frame_step = int(round(self.frame_stride_ms * sr / 1000))

        signal_length = len(signal)
        # +1 to include the last frame
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros(pad_signal_length - signal_length)
        pad_signal = np.append(signal, z)

        indices = (np.tile(np.arange(0, frame_length), (num_frames, 1))
                   + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T)

        frames = pad_signal[indices.astype(np.int32, copy=False)]
        return frames, frame_length, frame_step

    def apply_window(self, frames):
        """Apply Hamming window to each frame"""
        frame_length = frames.shape[1]
        return frames * np.hamming(frame_length)

    def compute_power_spectrum(self, frames):
        """Compute power spectrum for each frame"""
        mag_frames = np.absolute(np.fft.rfft(frames, self.n_fft))
        return (1.0 / self.n_fft) * (mag_frames ** 2)

    def create_mel_filterbank(self, sr):
        """Create Mel filter bank"""
        return librosa.filters.mel(sr=sr, n_fft=self.n_fft, n_mels=self.n_mels)

    def apply_mel_filterbank(self, pow_frames, mel_fbanks):
        """Apply Mel filter bank to power spectrum"""
        mel_energy = np.dot(pow_frames, mel_fbanks.T)
        return np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)

    def compute_mfcc_coefficients(self, log_mel_energy):
        """Compute MFCC coefficients from log Mel energies"""
        # Keeps c0…cN by default for consistency with your app
        mfcc_coeffs = dct(log_mel_energy, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        return mfcc_coeffs

    def full_pipeline(self, signal, sr):
        """Complete MFCC processing pipeline"""
        emphasized_signal = self.pre_emphasize(signal)
        frames, _, _ = self.frame_signal(emphasized_signal, sr)
        windowed_frames = self.apply_window(frames)
        pow_frames = self.compute_power_spectrum(windowed_frames)
        mel_fbanks = self.create_mel_filterbank(sr)
        mel_energy = self.apply_mel_filterbank(pow_frames, mel_fbanks)
        log_mel_energy = np.log(mel_energy)
        mfcc = self.compute_mfcc_coefficients(log_mel_energy)

        return {
            'signal': signal,
            'emphasized_signal': emphasized_signal,
            'frames': frames,
            'windowed_frames': windowed_frames,
            'power_spectrum': pow_frames,
            'mel_fbanks': mel_fbanks,
            'mel_energy': mel_energy,
            'log_mel_energy': log_mel_energy,
            'mfcc': mfcc
        }

    # --------------------------------------------------
    #               PLOTTING UTILITIES
    # --------------------------------------------------
    @staticmethod
    def plot_time_domain(signal, sr, title="Time Domain Signal"):
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(signal, sr=sr, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        return fig

    @staticmethod
    def plot_spectrogram(signal, sr, title="Spectrogram"):
        fig, ax = plt.subplots(figsize=(10, 4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set_title(title)
        return fig

    @staticmethod
    def plot_mel_filterbank(mel_fbanks, title="Mel Filter Bank"):
        fig, ax = plt.subplots(figsize=(10, 4))
        for i in range(mel_fbanks.shape[0]):
            ax.plot(mel_fbanks[i])
        ax.set_title(title)
        ax.set_xlabel('Frequency Bin')
        ax.set_ylabel('Amplitude')
        return fig

    @staticmethod
    def plot_mfcc(mfcc, sr, title="MFCC Coefficients"):
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(mfcc.T, sr=sr, x_axis='time', ax=ax)
        fig.colorbar(img, ax=ax)
        ax.set_title(title)
        ax.set_ylabel("MFCC Coefficients")
        return fig
