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

# ===================== CONFIG =====================
VOICE_TIMEOUT = 6
VISION_STALE_TIME = 1.5
CONF_THRESHOLD = 0.45
PLAN_CONF_MIN = 0.6
VISION_COOLDOWN = 0.4
LLM_TIMEOUT = 3.0 

ALLOWED_ACTIONS = {"pick", "place", "stop"}
STOPWORDS = {"pick", "up", "the", "a", "an", "can", "you", "please", "to", "give", "me"}

stop_event = threading.Event()

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

    def clear(self):
        os.system("cls" if os.name == "nt" else "clear")

    def render(self):
        with self.lock:
            s = dict(self.state)
        self.clear()
        print("╔══ ROBOTIC MULTIMODAL CONTROL ══╗")
        print(f" vision      : {s['vision']}")
        print(f" vision_age  : {s['vision_age']}")
        print("───────────────────────────────")
        print(f" command     : {s['command']}")
        print(f" voice_conf  : {s['voice_conf']}")
        print("───────────────────────────────")
        print(f" objects     : {s['objects']}")
        print("───────────────────────────────")
        print(f" plan        : {s['plan']}")
        print(f" plan_conf   : {s['plan_conf']}")
        print("───────────────────────────────")
        print(f" robot       : {s['robot']}")
        print(f" cmd_q       : {s['cmd_q']}")
        print(f" plan_q      : {s['plan_q']}")
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
        print("[AUDIO] Speech recognition ready")

    def listen(self):
        with sd.InputStream(
            samplerate=self.rate,
            channels=1,
            dtype="int16"
        ) as stream:
            start = time.time()
            while time.time() - start < VOICE_TIMEOUT and not stop_event.is_set():
                data, _ = stream.read(4000)
                if self.rec.AcceptWaveform(data.tobytes()):
                    res = json.loads(self.rec.Result())
                    self.rec.Reset()
                    txt = res.get("text", "")
                    confs = [w.get("conf", 0.5) for w in res.get("result", [])]
                    conf = sum(confs) / len(confs) if confs else 0.5
                    return txt, round(conf, 2)

        self.rec.Reset()
        return "", 0.0

    def speak(self, text):
        if not text:
            return
        print("[TTS]", text)
        engine = pyttsx3.init("sapi5")
        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

# ===================== VISION =====================
class Vision:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(0)
        self.last = {"items": [], "time": 0}

        # camera tuning (adjust once)
        self.min_box_area = 0.01   # very far
        self.max_box_area = 0.25   # very close

    def _estimate_z(self, box_area_norm):
        """
        Monocular depth proxy:
        larger box => closer => higher z0
        """
        z = (box_area_norm - self.min_box_area) / (
            self.max_box_area - self.min_box_area
        )
        return round(float(max(0.0, min(1.0, z))), 2)

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
            box_area = ((x2 - x1) * (y2 - y1)) / (w * h)

            items.append({
                "name": self.model.names[int(b.cls[0])],
                "confidence": round(float(b.conf[0]), 2),
                "x": round(cx, 3),
                "y": round(cy, 3),
                "z": self._estimate_z(box_area)
            })

        self.last = {"items": items, "time": time.time()}
        return self.last

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
            "You MUST only use objects present in VISION.\n"
            "If object is not visible, return empty steps.\n"
            "Format:\n"
            "{speech:str, steps:[{action, object?}]}"
        )

    def fuzzy(self, w, names):
        m = difflib.get_close_matches(w, names, n=1, cutoff=0.4)
        return m[0] if m else None

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
        if a == -1 or b == -1:
            return {"speech": "I did not understand.", "steps": []}
        raw = json.loads(out[a:b])
        steps = []
        for s in raw.get("steps", []):
            if s.get("action") not in ALLOWED_ACTIONS:
                continue
            if s["action"] == "pick":
                match = self.fuzzy(s.get("object", ""), names)
                if not match:
                    return {"speech": "I do not see that object.", "steps": []}
                s["object"] = match
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
        self.s = {}

    def run(self):
        while not stop_event.is_set():
            self.s = self.v.read()
            self.t.update("vision", "OK" if self.s["items"] else "EMPTY")
            self.t.update("vision_age", round(time.time() - self.s["time"], 2))
            self.t.update("objects", [i["name"] for i in self.s["items"]])
            time.sleep(0.1)

    def snap(self):
        return copy.deepcopy(self.s)

class VoiceThread(threading.Thread):
    def __init__(self, v, q, t):
        super().__init__(daemon=True)
        self.v = v
        self.q = q
        self.t = t

    def run(self):
        while not stop_event.is_set():
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

            visible = [i["name"].lower() for i in v["items"]]
            words = [w for w in c["text"].lower().split() if w not in STOPWORDS]
            invalid = [w for w in words if w not in visible]

            if invalid:
                self.tts_q.put(f"I do not see {', '.join(invalid)}.")
                self.t.update("plan", "REJECTED")
                self.t.update("plan_conf", "0.0")
                continue

            self.t.update("plan", "THINKING")
            p = None

            def call():
                nonlocal p
                p = self.llm.plan(v["items"], c["text"])

            th = threading.Thread(target=call)
            th.start()
            th.join(timeout=LLM_TIMEOUT)

            if not p or not p["steps"]:
                self.tts_q.put("Command rejected.")
                continue

            conf = sum(
                next(i["confidence"] for i in v["items"] if i["name"] == s["object"])
                for s in p["steps"] if s["action"] == "pick"
            ) / len(p["steps"])

            self.t.update("plan", p["steps"])
            self.t.update("plan_conf", round(conf, 2))

            if conf >= PLAN_CONF_MIN:
                self.plan_q.put(p)
            else:
                self.tts_q.put("Not confident enough.")

class ExecutorThread(threading.Thread):
    def __init__(self, r, plan_q, tts_q, t):
        super().__init__(daemon=True)
        self.r = r
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
            self.tts_q.put(p.get("speech", ""))

            for s in p["steps"]:
                if s["action"] == "pick":
                    o = s.get("object")
                    d = next((i for i in vision.last["items"] if i["name"] == o), None)
                    if not d:
                        self.tts_q.put("Unsafe action blocked.")
                        break
                    self.r.pick(d["x"], d["y"])
                elif s["action"] == "place":
                    self.r.place()
                elif s["action"] == "stop":
                    self.r.stop()
                time.sleep(0.4)

            self.t.update("robot", "IDLE")

class TTSThread(threading.Thread):
    def __init__(self, voice, q):
        super().__init__(daemon=True)
        self.voice = voice
        self.q = q

    def run(self):
        print("[TTS] Worker started")
        while not stop_event.is_set():
            try:
                text = self.q.get(timeout=0.2)
                self.voice.speak(text)   
            except queue.Empty:
                pass



# ===================== MAIN =====================
def main():
    global vision
    global tts_active 

    voice = VoiceIO()
    vision = Vision()
    llm = LLM()
    robot = Robot()

    t = AdvancedTerminal()
    TerminalThread(t).start()

    cmd_q = queue.Queue()
    plan_q = queue.Queue()
    tts_q = queue.Queue()



    vt = VisionThread(vision, t)
    vt.start()

    VoiceThread(voice, cmd_q, t).start()
    PlannerThread(llm, vt, cmd_q, plan_q, tts_q, t).start()
    ExecutorThread(robot, plan_q, tts_q, t).start()
    TTSThread(voice, tts_q).start()

    try:
        while True:
            t.update("cmd_q", cmd_q.qsize())
            t.update("plan_q", plan_q.qsize())
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        vision.release()
        print("\nShutdown clean")

if __name__ == "__main__":
    main()
