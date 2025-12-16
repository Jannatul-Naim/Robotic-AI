from voice.tts import speak
from communication.serial_comms import move
from control import inverse_kinematics

def move_arm(intent):
    try:
        x = float(intent.get("x"))
        y = float(intent.get("y"))
    except:
        speak("Invalid coordinates")
        return

    angles = inverse_kinematics(x, y)
    if not angles:
        speak("Target out of reach")
        return

    base, elbow = angles

    move(0, base)
    move(1, elbow)

    speak("Arm moved")
