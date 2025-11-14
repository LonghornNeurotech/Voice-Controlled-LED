"""
Master preprocessing script for voice-controlled LED project.
All team members should collaborate on this single file using Git branches and Pull Requests.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset


class AudioPreprocessor:
    """Base class for audio preprocessing."""
    
    def __init__(self, sample_rate: int = 16000):
        """
        Initialize preprocessor.
        
        Args:
            sample_rate: Target sample rate for audio (default: 16000 Hz)
        """
        self.sample_rate = sample_rate
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load audio file and resample if necessary.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Audio signal as numpy array
        """
        audio, sr = librosa.load(file_path, sr=self.sample_rate)
        return audio
    
    def normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio signal to [-1, 1] range.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Normalized audio signal
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    def trim_silence(self, audio: np.ndarray, top_db: int = 20) -> np.ndarray:
        """
        Trim silence from beginning and end of audio.
        
        Args:
            audio: Input audio signal
            top_db: Threshold in dB below reference to consider as silence
            
        Returns:
            Trimmed audio signal
        """
        trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return trimmed


class AudioDataset(Dataset):
    """Base dataset class for loading audio files."""
    
    def __init__(self, data_dir: str, preprocessor: AudioPreprocessor, transform=None):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing class subdirectories
            preprocessor: AudioPreprocessor instance
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.transform = transform
        self.samples = []
        self.labels = []
        self._load_samples()
    
    def _load_samples(self):
        """Load all audio samples and their labels."""
        classes = ['pi_on', 'pi_off', 'background']
        label_map = {cls: idx for idx, cls in enumerate(classes)}
        
        for class_name in classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for audio_file in class_dir.glob('*.wav'):
                    self.samples.append(str(audio_file))
                    self.labels.append(label_map[class_name])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio = self.preprocessor.load_audio(self.samples[idx])
        label = self.labels[idx]
        
        if self.transform:
            audio = self.transform(audio)
        
        return audio, label


class VoicePreprocessor(AudioPreprocessor):
    """
    Master preprocessing pipeline for voice-controlled LED project.
    Team members should extend this class with their preprocessing contributions.
    """
    
    def __init__(self, sample_rate: int = 16000):
        super().__init__(sample_rate)
        # Add initialization parameters here as needed
    
    def preprocess(self, audio: np.ndarray) -> np.ndarray:
        """
        Main preprocessing function.
        Implement the preprocessing pipeline here.
        
        Args:
            audio: Raw audio signal
            
        Returns:
            Preprocessed audio/features
        """
        # Example: normalize and trim silence
        audio = self.normalize(audio)
        audio = self.trim_silence(audio)
        
        # ========================================================================
        # SIGNAL PREPROCESSING IDEAS - Reference: Week 2 of Onboarding Files
        # ========================================================================
        # 
        # FILTERING TECHNIQUES (inspired by real-time signal processing):
        # 
        # 1. BANDPASS FILTERING:
        #    - Remove frequencies outside speech range (typically 80-8000 Hz for speech)
        #    - Use scipy.signal.butter() or librosa for bandpass filtering
        #    - Example: Filter 80-8000 Hz to focus on speech frequencies
        #    - Why: Removes low-frequency noise (e.g., 60 Hz powerline interference) 
        #      and high-frequency noise that doesn't contain speech information
        #    - Reference: Similar to EEG bandpass filtering (1-50 Hz) in real-time processing
        #
        # 2. NOTCH FILTERING:
        #    - Remove specific frequency bands (e.g., 50 Hz or 60 Hz powerline noise)
        #    - Use scipy.signal.iirnotch() for notch filtering
        #    - Example: Remove 60 Hz hum from electrical interference
        #    - Why: Powerline noise can contaminate audio signals, especially in indoor environments
        #    - Reference: Similar to EEG notch filtering at 60 Hz in real-time processing
        #
        # 3. HIGH-PASS FILTERING:
        #    - Remove low-frequency noise and DC offset
        #    - Use scipy.signal.butter() with highpass configuration
        #    - Example: High-pass filter at 80 Hz to remove rumble and wind noise
        #    - Why: Low frequencies often contain non-speech noise
        #
        # 4. LOW-PASS FILTERING:
        #    - Remove high-frequency noise above speech range
        #    - Use scipy.signal.butter() with lowpass configuration
        #    - Example: Low-pass filter at 8000 Hz (Nyquist for 16 kHz sample rate)
        #    - Why: Reduces aliasing and removes ultrasonic noise
        #
        # NOISE REDUCTION TECHNIQUES:
        #
        # 5. SPECTRAL SUBTRACTION:
        #    - Estimate noise spectrum and subtract from signal spectrum
        #    - Use librosa or custom implementation
        #    - Example: Estimate noise from silent segments, subtract from full signal
        #    - Why: Reduces background noise while preserving speech content
        #

        return audio
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from audio (e.g., MFCC, mel-spectrogram).
        Customize this method with your feature extraction approach.
        
        Args:
            audio: Preprocessed audio signal
            
        Returns:
            Feature array
        """
        # Example: Extract MFCC features
        # TODO: Team members can contribute different feature extraction methods
        
        # ========================================================================
        # FEATURE EXTRACTION IDEAS - Reference: Week 2 of Onboarding Files
        # ========================================================================
        #
        # CURRENT IMPLEMENTATION: MFCC (Mel-Frequency Cepstral Coefficients)
        # - Good baseline for speech recognition
        # - Consider experimenting with n_mfcc values: 13, 20, 40
        # - Adjust n_fft and hop_length based on desired time/frequency resolution
        #
        # ALTERNATIVE FEATURE EXTRACTION METHODS:
        #
        # 1. MEL-SPECTROGRAM (2D feature, good for CNNs):
        #    mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, 
        #                                              n_mels=128, n_fft=2048, hop_length=512)
        #    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        #    - Returns 2D array: (n_mels, time_frames)
        #    - Good for CNN-based models
        #    - Can flatten or use as-is for 2D convolutions
        #
        # 2. SPECTROGRAM (linear frequency scale):
        #    stft = librosa.stft(audio, n_fft=2048, hop_length=512)
        #    magnitude = np.abs(stft)
        #    spectrogram_db = librosa.amplitude_to_db(magnitude)
        #    - Returns 2D array: (freq_bins, time_frames)
        #    - More frequency resolution than mel-spectrogram
        #
        # 3. COMBINED FEATURES (stack multiple feature types):
        #    - Stack MFCC + Mel-spectrogram + Chroma + ZCR
        #    - Use np.concatenate() or np.hstack() to combine
        #    - Example: Combine 13 MFCCs + 128 mel bins + 12 chroma = 153 features
        #    - Why: Different features capture different aspects of speech
        #

        
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=13,
            n_fft=2048,
            hop_length=512
        )
        return mfccs


class VoiceDataset(AudioDataset):
    """
    Master dataset class for voice-controlled LED project.
    Team members should extend this to add custom data loading logic.
    """
    
    def __init__(self, data_dir: str, preprocessor: VoicePreprocessor):
        super().__init__(data_dir, preprocessor)
        self.preprocessor = preprocessor
    
    def __getitem__(self, idx):
        audio, label = super().__getitem__(idx)
        
        # Apply custom preprocessing
        features = self.preprocessor.preprocess(audio)
        features = self.preprocessor.extract_features(features)
        
        # Convert to tensor
        features = torch.FloatTensor(features)
        
        return features, label


# Example usage
if __name__ == "__main__":
    preprocessor = VoicePreprocessor(sample_rate=16000)
    dataset = VoiceDataset(data_dir="../data", preprocessor=preprocessor)
    
    print(f"Dataset size: {len(dataset)}")
    features, label = dataset[0]
    print(f"Features shape: {features.shape}, Label: {label}")
