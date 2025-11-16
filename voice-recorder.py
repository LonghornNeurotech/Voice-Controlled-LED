import sounddevice as sd
import soundfile as sf
import numpy as np
import os
from datetime import datetime
from pynput import keyboard
import time

# Configuration
SAMPLE_RATE = 16000  # 16kHz is standard for speech
DURATION = 2.0  # seconds per recording
OUTPUT_DIR = "voice_data"

# Create output directories
os.makedirs(f"{OUTPUT_DIR}/pi_on", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/pi_off", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/background", exist_ok=True)

# Counters
counters = {
    'pi_on': 0,
    'pi_off': 0,
    'background': 0
}

# Recording state
is_recording = False
current_label = None

def record_audio():
    """Record audio for DURATION seconds"""
    print(f"🔴 Recording {current_label}...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE), 
                   samplerate=SAMPLE_RATE, 
                   channels=1, 
                   dtype='float32')
    sd.wait()
    return audio

def save_recording(audio, label):
    """Save the recording with timestamp"""
    counters[label] += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_DIR}/{label}/{label}_{counters[label]:04d}_{timestamp}.wav"
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"✅ Saved: {filename}")
    print(f"   Total {label}: {counters[label]}")
    print()

def on_press(key):
    """Handle key presses"""
    global current_label, is_recording
    
    try:
        if key.char == '1':
            current_label = 'pi_on'
            audio = record_audio()
            save_recording(audio, current_label)
            
        elif key.char == '2':
            current_label = 'pi_off'
            audio = record_audio()
            save_recording(audio, current_label)
            
        elif key.char == '3':
            current_label = 'background'
            audio = record_audio()
            save_recording(audio, current_label)
            
        elif key.char == 'q':
            print("\n👋 Stopping recorder...")
            return False  # Stop listener
            
    except AttributeError:
        pass

def print_instructions():
    """Print usage instructions"""
    print("=" * 60)
    print("🎤 VOICE COMMAND RECORDER")
    print("=" * 60)
    print("\nInstructions:")
    print("  Press '1' → Record 'Pi on'")
    print("  Press '2' → Record 'Pi off'")
    print("  Press '3' → Record background noise/silence")
    print("  Press 'q' → Quit")
    print("\nTips:")
    print("  - Vary your tone, speed, and volume")
    print("  - Try whispering, shouting, casual speech")
    print("  - Record in different locations/positions")
    print("  - For background: capture room noise, music, talking, etc.")
    print("\nReady! Press a key to start recording...")
    print("=" * 60)
    print()

def main():
    print_instructions()
    
    # Start listening for keypresses
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RECORDING SUMMARY")
    print("=" * 60)
    for label, count in counters.items():
        print(f"  {label}: {count} recordings")
    print("=" * 60)
    print("\nFiles saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()