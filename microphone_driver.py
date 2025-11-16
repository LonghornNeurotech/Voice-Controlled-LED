"""
INMP441 I2S microphone driver for Raspberry Pi.
Handles audio acquisition from the INMP441 MEMS I2S microphone module.
"""

import numpy as np
import time
from typing import Optional


class INMP441Driver:
    """
    Driver for INMP441 I2S microphone module.
    
    The INMP441 uses I2S protocol with the following connections:
    - VDD: 3.3V
    - GND: GND
    - WS (LRCL): GPIO 19 (Word Select)
    - SCK (BCLK): GPIO 18 (Bit Clock)
    - SD: GPIO 20 (Serial Data)
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_size: int = 1024
    ):
        """
        Initialize INMP441 driver.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            buffer_size: Size of audio buffer (default: 1024)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.buffer_size = buffer_size
        
        # Try to import I2S library
        # Note: This requires I2S support, which may need additional setup
        try:
            import pyaudio
            self.pyaudio_available = True
            self.pyaudio = pyaudio
            self.audio = pyaudio.PyAudio()
        except ImportError:
            self.pyaudio_available = False
            self.pyaudio = None
            print("Warning: PyAudio not available. I2S functionality may be limited.")
            print("For full I2S support, you may need to use a C library or hardware-specific driver.")
    
    def start_stream(self):
        """Start audio stream from INMP441."""
        if not self.pyaudio_available:
            raise RuntimeError("PyAudio not available. Cannot start stream.")
        
        
        self.stream = self.audio.open(
            format=self.pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.buffer_size,
            # device_index=...  # Specify I2S device index if needed
        )
    
    def read_audio(self, num_frames: Optional[int] = None) -> np.ndarray:
        """
        Read audio data from INMP441.
        
        Args:
            num_frames: Number of frames to read (default: buffer_size)
            
        Returns:
            Audio data as numpy array
        """
        if not hasattr(self, 'stream'):
            raise RuntimeError("Stream not started. Call start_stream() first.")
        
        if num_frames is None:
            num_frames = self.buffer_size
        
        # Read raw audio data
        raw_data = self.stream.read(num_frames, exception_on_overflow=False)
        
        # Convert to numpy array
        audio = np.frombuffer(raw_data, dtype=np.int16)
        
        # Normalize to [-1, 1] range
        audio = audio.astype(np.float32) / 32768.0
        
        return audio
    
    def stop_stream(self):
        """Stop audio stream."""
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
    
    def cleanup(self):
        """Cleanup resources."""
        self.stop_stream()
        if self.pyaudio_available:
            self.audio.terminate()
    
    def test_recording(self, duration: float = 2.0) -> np.ndarray:
        """
        Test recording for specified duration.
        
        Args:
            duration: Recording duration in seconds
            
        Returns:
            Recorded audio as numpy array
        """
        self.start_stream()
        
        num_samples = int(duration * self.sample_rate)
        audio_data = []
        
        print(f"Recording for {duration} seconds...")
        
        while len(audio_data) < num_samples:
            chunk = self.read_audio()
            audio_data.extend(chunk)
        
        self.stop_stream()
        
        audio_array = np.array(audio_data[:num_samples])
        print(f"Recorded {len(audio_array)} samples at {self.sample_rate} Hz")
        
        return audio_array


def test_microphone():
    """Test function for INMP441 microphone."""
    print("=" * 60)
    print("INMP441 MICROPHONE TEST")
    print("=" * 60)
    print()
    print("Note: This is a basic test. For full I2S functionality,")
    print("you may need to configure I2S on your Raspberry Pi and")
    print("use hardware-specific drivers.")
    print()
    
    try:
        driver = INMP441Driver(sample_rate=16000)
        
        # Test recording
        audio = driver.test_recording(duration=2.0)
        
        print(f"Audio shape: {audio.shape}")
        print(f"Audio range: [{audio.min():.4f}, {audio.max():.4f}]")
        print(f"Audio mean: {audio.mean():.4f}")
        print()
        print("✅ Test completed successfully!")
        print("   You can save the audio to verify it was recorded correctly:")
        print("   import soundfile as sf")
        print("   sf.write('test_recording.wav', audio, 16000)")
        
        driver.cleanup()
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        print()
        print("Troubleshooting:")
        print("1. Ensure I2S is enabled: sudo raspi-config -> Interface Options -> I2S")
        print("2. Check I2S device is recognized")
        print("3. Verify hardware connections")
        print("4. You may need to use a C library for full I2S support")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test INMP441 I2S microphone')
    parser.add_argument('--test', action='store_true',
                        help='Run microphone test')
    
    args = parser.parse_args()
    
    if args.test:
        test_microphone()
    else:
        print("Use --test flag to run microphone test")
        print("Example: python microphone_driver.py --test")

