"""
LED control script for USB microphone recorder.
Controls LED 1 based on voice recording status.
"""

import RPi.GPIO as GPIO
import time

# GPIO pin for LED 1 (USB mic recorder)
LED_PIN = 18  # Change this to match your circuit

def setup():
    """Initialize GPIO pins."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    GPIO.output(LED_PIN, GPIO.LOW)
    print(f"LED initialized on GPIO {LED_PIN}")

def turn_on():
    """Turn LED on."""
    GPIO.output(LED_PIN, GPIO.HIGH)
    print("LED ON")

def turn_off():
    """Turn LED off."""
    GPIO.output(LED_PIN, GPIO.LOW)
    print("LED OFF")

def toggle():
    """Toggle LED state."""
    current_state = GPIO.input(LED_PIN)
    GPIO.output(LED_PIN, not current_state)
    print(f"LED {'ON' if not current_state else 'OFF'}")

def cleanup():
    """Cleanup GPIO pins."""
    GPIO.cleanup()
    print("GPIO cleaned up")

# Test function
if __name__ == "__main__":
    try:
        setup()
        print("Testing LED...")
        for i in range(5):
            turn_on()
            time.sleep(0.5)
            turn_off()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cleanup()

