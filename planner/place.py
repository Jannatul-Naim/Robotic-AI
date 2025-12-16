from voice.tts import speak
from communication.serial_comms import move
from control.kinematics import inverse_kinematics
import time

def place_object(intent):
    speak("Placing object")

    # Default place location
    x, y = 10, -5

    angles = inverse_kinematics(x, y)
    if not angles:
        speak("Cannot place object")
        return

    base, elbow = angles

    move(0, base)
    move(1, elbow)
    time.sleep(1)

    move(2, 90)   # Gripper open
    speak("Object placed")
