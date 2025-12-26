# ===================== ENV SAFETY =====================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ===================== IMPORTS =====================
import time, json, threading, queue, difflib, copy
import cv2, torch
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== GLOBAL EVENTS =====================
stop_event = threading.Event()
speaking_event = threading.Event()   # 🔒 MIC MUTE LOCK

# ===================== CONFIG =====================
VOICE_TIMEOUT = 6
VISION_STALE_TIME = 1.5
CONF_THRESHOLD = 0.45
PLAN_CONF_MIN = 0.6
VISION_COOLDOWN = 0.4
LLM_TIMEOUT = 3.0

ALLOWED_ACTIONS = {"pick", "place", "stop"}
STOPWORDS = {"pick", "up", "the", "a", "an", "can", "you", "please", "to", "give", "me"}

# ===================== TERMINAL =====================
class AdvancedTerminal:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {
            "vision": "-",
            "vision_age": "-",
            "command": "-",
            "voice_conf": "-",
            "objects": "-",
            "plan": "-",
            "plan_conf": "-",
            "robot": "IDLE",
            "cmd_q": 0,
            "plan_q": 0,
        }

    def update(self, k, v):
        with self.lock:
            self.state[k] = v

    def render(self):
        with self.lock:
            s = dict(self.state)
        os.system("cls" if os.name == "nt" else "clear")
        print("╔══ ROBOTIC MULTIMODAL CONTROL ══╗")
        print(f" Vision      : {s['vision']}")
        print(f" Vision Age  : {s['vision_age']}")
        print("───────────────────────────────")
        print(f" Command     : {s['command']}")
        print(f" Voice Conf  : {s['voice_conf']}")
        print("───────────────────────────────")
        print(f" Objects     : {s['objects']}")
        print("───────────────────────────────")
        print(f" Plan        : {s['plan']}")
        print(f" Plan Conf   : {s['plan_conf']}")
        print("───────────────────────────────")
        print(f" Robot       : {s['robot']}")
        print(f" CmdQ / PlanQ: {s['cmd_q']} / {s['plan_q']}")
        print("╚══════════════════════════════╝")

class TerminalThread(threading.Thread):
    def __init__(self, t):
        super().__init__(daemon=True)
        self.t = t

    def run(self):
        while not stop_event.is_set():
            self.t.render()
            time.sleep(0.25)

# ===================== VOICE =====================
class VoiceIO:
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.rate = 16000
        self.rec = KaldiRecognizer(Model(model_path), self.rate)
        print("[AUDIO] STT ready")

    def listen(self):
        if speaking_event.is_set():
            return "", 0.0

        with sd.InputStream(samplerate=self.rate, channels=1, dtype="int16") as stream:
            start = time.time()
            while time.time() - start < VOICE_TIMEOUT and not stop_event.is_set():
                data, _ = stream.read(4000)
                if self.rec.AcceptWaveform(data.tobytes()):
                    res = json.loads(self.rec.Result())
                    self.rec.Reset()
                    txt = res.get("text", "").strip()
                    confs = [w.get("conf", 0.6) for w in res.get("result", [])]
                    conf = sum(confs) / len(confs) if confs else 0.6
                    return txt, round(conf, 2)
        self.rec.Reset()
        return "", 0.0

    def speak(self, text):
        if not text:
            return
        speaking_event.set()
        print("[TTS]", text)

        engine = pyttsx3.init("sapi5")
        engine.setProperty("rate", 170)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

        time.sleep(0.2)
        speaking_event.clear()

# ===================== VISION =====================
class Vision:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(0)
        self.last = {"items": [], "time": 0}

    def read(self):
        if time.time() - self.last["time"] < VISION_COOLDOWN:
            return self.last

        ok, frame = self.cap.read()
        if not ok:
            return self.last

        h, w, _ = frame.shape
        r = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        items = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            items.append({
                "name": self.model.names[int(b.cls[0])],
                "confidence": round(float(b.conf[0]), 2),
                "x": round(cx, 3),
                "y": round(cy, 3)
            })

        self.last = {"items": items, "time": time.time()}
        return self.last

    def snap(self):
        return copy.deepcopy(self.last)

    def release(self):
        self.cap.release()

# ===================== LLM =====================
class LLM:
    def __init__(self):
        self.name = "microsoft/phi-3-mini-4k-instruct"
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        self.sys = (
            "You are a robot planner.\n"
            "Reply ONLY valid JSON.\n"
            "Allowed actions: pick, place, stop.\n"
            "Only use objects in VISION.\n"
            "{speech:str, steps:[{action, object?}]}"
        )

    def plan(self, items, cmd):
        names = [i["name"] for i in items]
        msgs = [
            {"role": "system", "content": self.sys},
            {"role": "user", "content": f"VISION:{items}\nCMD:{cmd}"}
        ]
        x = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        y = self.model.generate(x, max_new_tokens=256, do_sample=False)
        out = self.tok.decode(y[0][x.shape[-1]:], skip_special_tokens=True)

        a, b = out.find("{"), out.rfind("}") + 1
        if a == -1:
            return {"speech": "I did not understand.", "steps": []}

        raw = json.loads(out[a:b])
        steps = []
        for s in raw.get("steps", []):
            if s.get("action") not in ALLOWED_ACTIONS:
                continue
            if s["action"] == "pick" and s.get("object") not in names:
                return {"speech": "I do not see that object.", "steps": []}
            steps.append(s)

        return {"speech": raw.get("speech", "Okay."), "steps": steps}

# ===================== ROBOT =====================
class Robot:
    def pick(self, x, y): print(f"[ROBOT] PICK @ {x},{y}")
    def place(self): print("[ROBOT] PLACE")
    def stop(self): print("[ROBOT] STOP")

# ===================== THREADS =====================
class VisionThread(threading.Thread):
    def __init__(self, v, t):
        super().__init__(daemon=True)
        self.v = v
        self.t = t

    def run(self):
        while not stop_event.is_set():
            s = self.v.read()
            self.t.update("vision", "OK" if s["items"] else "EMPTY")
            self.t.update("vision_age", round(time.time() - s["time"], 2))
            self.t.update("objects", [i["name"] for i in s["items"]])
            time.sleep(0.1)

class VoiceThread(threading.Thread):
    def __init__(self, v, q, t):
        super().__init__(daemon=True)
        self.v = v
        self.q = q
        self.t = t

    def run(self):
        while not stop_event.is_set():
            if speaking_event.is_set():
                time.sleep(0.1)
                continue
            txt, conf = self.v.listen()
            if txt:
                self.t.update("command", txt)
                self.t.update("voice_conf", conf)
                self.q.put({"text": txt, "conf": conf})

class PlannerThread(threading.Thread):
    def __init__(self, llm, vision, cmd_q, plan_q, tts_q, t):
        super().__init__(daemon=True)
        self.llm = llm
        self.vision = vision
        self.cmd_q = cmd_q
        self.plan_q = plan_q
        self.tts_q = tts_q
        self.t = t

    def run(self):
        while not stop_event.is_set():
            try:
                c = self.cmd_q.get(timeout=0.2)
            except queue.Empty:
                continue

            v = self.vision.snap()
            if time.time() - v["time"] > VISION_STALE_TIME:
                self.tts_q.put("I cannot see clearly.")
                continue

            self.t.update("plan", "THINKING")
            p = self.llm.plan(v["items"], c["text"])

            if not p["steps"]:
                self.tts_q.put(p["speech"])
                continue

            conf = sum(i["confidence"] for i in v["items"]) / max(1, len(v["items"]))
            self.t.update("plan", p["steps"])
            self.t.update("plan_conf", round(conf, 2))

            if conf >= PLAN_CONF_MIN:
                self.plan_q.put(p)
            else:
                self.tts_q.put("Not confident enough.")

class ExecutorThread(threading.Thread):
    def __init__(self, r, vision, plan_q, tts_q, t):
        super().__init__(daemon=True)
        self.r = r
        self.v = vision
        self.plan_q = plan_q
        self.tts_q = tts_q
        self.t = t

    def run(self):
        while not stop_event.is_set():
            try:
                p = self.plan_q.get(timeout=0.2)
            except queue.Empty:
                continue

            self.t.update("robot", "EXECUTING")
            self.tts_q.put(p["speech"])

            for s in p["steps"]:
                if s["action"] == "pick":
                    o = s["object"]
                    d = next(i for i in self.v.last["items"] if i["name"] == o)
                    self.r.pick(d["x"], d["y"])
                elif s["action"] == "place":
                    self.r.place()
                elif s["action"] == "stop":
                    self.r.stop()
                time.sleep(0.4)

            self.t.update("robot", "IDLE")

class TTSThread(threading.Thread):
    def __init__(self, v, q):
        super().__init__(daemon=True)
        self.v = v
        self.q = q

    def run(self):
        while not stop_event.is_set():
            try:
                self.v.speak(self.q.get(timeout=0.2))
            except queue.Empty:
                pass

# ===================== MAIN =====================
def main():
    voice = VoiceIO()
    vision = Vision()
    llm = LLM()
    robot = Robot()

    t = AdvancedTerminal()
    TerminalThread(t).start()

    cmd_q = queue.Queue()
    plan_q = queue.Queue()
    tts_q = queue.Queue()

    VisionThread(vision, t).start()
    VoiceThread(voice, cmd_q, t).start()
    PlannerThread(llm, vision, cmd_q, plan_q, tts_q, t).start()
    ExecutorThread(robot, vision, plan_q, tts_q, t).start()
    TTSThread(voice, tts_q).start()

    try:
        while True:
            t.update("cmd_q", cmd_q.qsize())
            t.update("plan_q", plan_q.qsize())
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        vision.release()
        print("\nShutdown complete")

if __name__ == "__main__":
    main()
