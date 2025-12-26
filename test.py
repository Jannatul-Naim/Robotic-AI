import time
import threading
import queue
import difflib
import pyttsx3

# ===================== GLOBAL STATE =====================

system_running = True
speaking_now = False

command_queue = queue.Queue()
plan_queue = queue.Queue()

last_objects = []
last_object_ref = None
last_place_ref = None

# ===================== TTS =====================

engine = pyttsx3.init()
engine.setProperty("rate", 165)

def speak(text):
    global speaking_now
    speaking_now = True

    print(f"[TTS] Speaking: {text}")

    engine.say(text)
    engine.runAndWait()

    speaking_now = False
    print("[TTS] Done speaking")

# ===================== DYNAMIC WORD MATCH =====================

def best_match(word, candidates):
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None

# ===================== VOICE INPUT (SIMULATED STT HOOK) =====================

def on_voice_input(text, confidence):
    if speaking_now:
        print("[STT] Ignored input (system is speaking)")
        return

    print(f"[STT] Heard: '{text}' | confidence={confidence:.2f}")

    if confidence < 0.35:
        print("[STT] Confidence too low, ignoring")
        return

    command_queue.put((text.lower(), confidence))

# ===================== COMMAND PARSER =====================

def parse_command(text):
    global last_object_ref, last_place_ref

    words = text.split()
    action = None
    target = None

    for w in words:
        if w in ["pick", "pickup", "grab", "take"]:
            action = "pick"
        elif w in ["place", "put", "drop"]:
            action = "place"

    for w in words:
        match = best_match(w, last_objects)
        if match:
            target = match
            last_object_ref = match

    if "it" in words and last_object_ref:
        target = last_object_ref

    if not action:
        return None

    return {"action": action, "target": target}

# ===================== PLANNER =====================

def planner_loop():
    while system_running:
        if not command_queue.empty():
            text, conf = command_queue.get()
            plan = parse_command(text)

            if plan:
                plan_queue.put((plan, conf))
                print(f"[PLANNER] Plan created: {plan}")
            else:
                speak("I did not understand the command")

        time.sleep(0.05)

# ===================== ROBOT EXECUTOR =====================

def executor_loop():
    while system_running:
        if not plan_queue.empty():
            plan, conf = plan_queue.get()

            print(f"[ROBOT] Executing: {plan}")

            if plan["action"] == "pick":
                speak(f"Picking up the {plan['target']}")
            elif plan["action"] == "place":
                speak("Placing the object")

        time.sleep(0.05)

# ===================== VISION UPDATE =====================

def update_vision(objects):
    global last_objects
    last_objects = objects

    print(f"[VISION] Detected objects: {objects}")

# ===================== UI DEBUG =====================

def debug_ui():
    print("\n╔══ ROBOTIC MULTIMODAL CONTROL ══╗")
    print(f" Detected Objects : {', '.join(last_objects) if last_objects else 'None'}")
    print(f" CmdQ / PlanQ     : {command_queue.qsize()} / {plan_queue.qsize()}")
    print("╚══════════════════════════════╝\n")

# ===================== MAIN =====================

if __name__ == "__main__":
    update_vision(["laptop", "bottle", "book"])

    threading.Thread(target=planner_loop, daemon=True).start()
    threading.Thread(target=executor_loop, daemon=True).start()

    # ---- SIMULATED VOICE INPUT ----
    on_voice_input("pick up the laptop", 0.82)

    while True:
        debug_ui()
        time.sleep(1)
