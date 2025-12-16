from planner.pick import pick_object
from planner.place import place_object
from planner.move import move_arm
from voice.tts import speak

def execute_plan(intent: dict):
    if not intent:
        speak("Invalid command")
        return

    action = intent.get("action")

    if action == "pick":
        pick_object(intent)

    elif action == "place":
        place_object(intent)

    elif action == "move":
        move_arm(intent)

    elif action == "stop":
        speak("Stopping all motion")

    else:
        speak("Unknown action")
