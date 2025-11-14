"""
Real-time inference script for voice-controlled LED.
Runs on Raspberry Pi 4B to detect voice commands and control LED.
"""

import argparse
import torch
import numpy as np
import time
from collections import deque
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.preprocessing import VoicePreprocessor
from model.architecture import VoiceClassifier
from led_control_inmp441 import setup, turn_on, turn_off, cleanup
from microphone_driver import INMP441Driver


class PredictionSmoother:
    """Smooth predictions to prevent LED flickering."""
    
    def __init__(self, window_size: int = 5, confidence_threshold: float = 0.7):
        """
        Initialize prediction smoother.
        
        Args:
            window_size: Number of predictions to average
            confidence_threshold: Minimum confidence to trigger action
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=window_size)
        self.current_state = None  # 0: off, 1: on, 2: background
    
    def add_prediction(self, prediction: int, confidence: float):
        """
        Add a new prediction to the history.
        
        Args:
            prediction: Predicted class (0: pi_on, 1: pi_off, 2: background)
            confidence: Prediction confidence
        """
        if confidence >= self.confidence_threshold:
            self.prediction_history.append(prediction)
        else:
            # Low confidence predictions are treated as background
            self.prediction_history.append(2)
    
    def get_smoothed_prediction(self) -> Optional[int]:
        """
        Get smoothed prediction based on history.
        
        Returns:
            Smoothed prediction or None if uncertain
        """
        if len(self.prediction_history) < self.window_size:
            return None
        
        # Get most common prediction in window
        predictions = list(self.prediction_history)
        most_common = max(set(predictions), key=predictions.count)
        
        # Only return if it appears in majority of window
        count = predictions.count(most_common)
        if count >= self.window_size * 0.6:  # 60% threshold
            return most_common
        
        return None


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load trained model.
    
    Args:
        model_path: Path to model weights file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Initialize model (adjust input_size based on your feature extraction)
    # TODO: Update input_size to match your trained model
    model = VoiceClassifier(input_size=128, num_classes=3)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    
    return model


def preprocess_audio(audio: np.ndarray, preprocessor: VoicePreprocessor) -> torch.Tensor:
    """
    Preprocess audio for inference.
    
    Args:
        audio: Raw audio signal
        preprocessor: VoicePreprocessor instance
        
    Returns:
        Preprocessed features as tensor
    """
    # Apply preprocessing pipeline
    processed = preprocessor.preprocess(audio)
    features = preprocessor.extract_features(processed)
    
    # Convert to tensor and add batch dimension
    features_tensor = torch.FloatTensor(features).unsqueeze(0)
    
    return features_tensor


def run_inference(
    model: torch.nn.Module,
    audio: np.ndarray,
    preprocessor: VoicePreprocessor,
    device: torch.device
) -> Tuple[int, float]:
    """
    Run inference on audio.
    
    Args:
        model: Trained model
        audio: Audio signal
        preprocessor: VoicePreprocessor instance
        device: Device to run inference on
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Preprocess audio
    features = preprocess_audio(audio, preprocessor)
    features = features.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(features)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    return predicted.item(), confidence.item()


def main():
    parser = argparse.ArgumentParser(description='Real-time voice command inference')
    parser.add_argument('--model_path', type=str, default='model/weights/best_model.pt',
                        help='Path to trained model weights')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (default: 16000)')
    parser.add_argument('--buffer_duration', type=float, default=2.0,
                        help='Audio buffer duration in seconds (default: 2.0)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7,
                        help='Minimum confidence to trigger action (default: 0.7)')
    parser.add_argument('--smoothing_window', type=int, default=5,
                        help='Number of predictions to average (default: 5)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    try:
        model = load_model(args.model_path, device)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize preprocessing
    preprocessor = VoicePreprocessor(sample_rate=args.sample_rate)
    
    # Initialize microphone driver
    print("Initializing microphone...")
    try:
        mic_driver = INMP441Driver(sample_rate=args.sample_rate)
        mic_driver.start_stream()
        print("✅ Microphone initialized")
    except Exception as e:
        print(f"❌ Error initializing microphone: {e}")
        print("Make sure I2S is enabled and microphone is connected")
        return
    
    # Initialize LED control
    print("Initializing LED...")
    try:
        setup()
        print("✅ LED initialized")
    except Exception as e:
        print(f"⚠️  Warning: LED initialization failed: {e}")
        print("Continuing without LED control...")
    
    # Initialize prediction smoother
    smoother = PredictionSmoother(
        window_size=args.smoothing_window,
        confidence_threshold=args.confidence_threshold
    )
    
    print("\n" + "=" * 60)
    print("VOICE COMMAND INFERENCE - RUNNING")
    print("=" * 60)
    print("Say 'Pi on' to turn LED on")
    print("Say 'Pi off' to turn LED off")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    print()
    
    # Audio buffer
    buffer_size = int(args.buffer_duration * args.sample_rate)
    audio_buffer = deque(maxlen=buffer_size)
    
    try:
        while True:
            # Read audio chunk
            audio_chunk = mic_driver.read_audio()
            audio_buffer.extend(audio_chunk)
            
            # When buffer is full, run inference
            if len(audio_buffer) >= buffer_size:
                audio_array = np.array(list(audio_buffer))
                
                # Run inference
                prediction, confidence = run_inference(
                    model, audio_array, preprocessor, device
                )
                
                # Add to smoother
                smoother.add_prediction(prediction, confidence)
                
                # Get smoothed prediction
                smoothed = smoother.get_smoothed_prediction()
                
                if smoothed is not None and smoothed != smoother.current_state:
                    smoother.current_state = smoothed
                    
                    if smoothed == 0:  # pi_on
                        print("🔵 Detected: 'Pi on' (confidence: {:.2f})".format(confidence))
                        try:
                            turn_on()
                        except:
                            pass
                    elif smoothed == 1:  # pi_off
                        print("🔴 Detected: 'Pi off' (confidence: {:.2f})".format(confidence))
                        try:
                            turn_off()
                        except:
                            pass
                    # smoothed == 2 is background, no action
            
            time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
    except KeyboardInterrupt:
        print("\n\nStopping inference...")
    finally:
        mic_driver.cleanup()
        try:
            cleanup()
        except:
            pass
        print("✅ Cleanup complete")


if __name__ == '__main__':
    main()

