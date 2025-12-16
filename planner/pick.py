from voice.tts import speak
from communication.serial_comms import move
from control import inverse_kinematics
from vision import detect
import time

def pick_object(intent):
    speak("Searching for object")

    detected = detect()
    if not detected:
        speak("Object not found")
        return

    label, cx, cy = detected
    speak(f"{label} detected")

    # TEMP mapping (replace with calibration later)
    x, y = 15, 5

    angles = inverse_kinematics(x, y)
    if not angles:
        speak("Object unreachable")
        return

    base, elbow = angles

    move(0, base)
    move(1, elbow)
    time.sleep(1)

    move(2, 40)   # Gripper close
    speak("Object picked")
