# ===================== ENV SAFETY =====================
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

# ===================== IMPORTS =====================
import time, json, threading, queue, re
import cv2, torch
import sounddevice as sd
import pyttsx3
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

# ===================== CONFIG =====================
VOICE_TIMEOUT = 6
VISION_STALE_TIME = 1.5
CONF_THRESHOLD = 0.45
VOICE_CONF_MIN = 0.25
PLAN_CONF_MIN = 0.55
VISION_COOLDOWN = 0.2

# ===================== OBJECT KNOWLEDGE =====================
OBJECT_SYNONYMS = {
    "tv": ["television", "screen", "monitor"],
    "bottle": ["water bottle", "flask"],
    "cup": ["mug", "glass"],
    "phone": ["mobile", "cellphone"],
}

# ===================== SMART OBJECT MATCHER =====================
class ObjectMatcher:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def normalize(self, t):
        return t.lower().strip()

    def match(self, query, candidates):
        query = self.normalize(query)
        candidates_norm = [self.normalize(c) for c in candidates]

        if query in candidates_norm:
            return candidates[candidates_norm.index(query)], 1.0

        for k, v in OBJECT_SYNONYMS.items():
            if query == k or query in v:
                for c in candidates:
                    if c == k or c in v:
                        return c, 0.95

        scores = [(c, fuzz.ratio(query, c)) for c in candidates]
        best, score = max(scores, key=lambda x: x[1])
        if score > 80:
            return best, score / 100.0

        q_emb = self.model.encode(query, convert_to_tensor=True)
        c_emb = self.model.encode(candidates, convert_to_tensor=True)
        sims = util.cos_sim(q_emb, c_emb)[0]
        idx = int(sims.argmax())
        if sims[idx] > 0.55:
            return candidates[idx], float(sims[idx])

        return None, 0.0

# ===================== TERMINAL UI =====================
class Terminal:
    def __init__(self):
        self.state = {}
        self.lock = threading.Lock()

    def update(self, k, v):
        with self.lock:
            self.state[k] = v

    def render(self):
        os.system("cls" if os.name == "nt" else "clear")
        s = self.state
        print("╔══ ROBOT MULTIMODAL SYSTEM ══╗")
        print(f" Vision     : {s.get('Vision','-')}")
        print(f" VisionAge  : {s.get('VisionAge','-')}s")
        print(f" Command    : {s.get('Command','-')}")
        print(f" VoiceConf  : {s.get('VoiceConf','-')}")
        print(f" Plan       : {s.get('Plan','-')}")
        print(f" PlanConf   : {s.get('PlanConf','-')}")
        print(f" Robot      : {s.get('Robot','IDLE')}")
        print("╚════════════════════════════╝")

# ===================== VOICE =====================
class VoiceIO:
    def __init__(self):
        self.rate = 16000
        self.rec = KaldiRecognizer(Model("vosk-model-small-en-us-0.15"), self.rate)
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", 170)
        self.lock = threading.Lock()

    def listen(self):
        self.rec.Reset()
        with sd.InputStream(samplerate=self.rate, channels=1, dtype="int16") as stream:
            start = time.time()
            while time.time() - start < VOICE_TIMEOUT:
                data, _ = stream.read(4000)
                if self.rec.AcceptWaveform(data.tobytes()):
                    r = json.loads(self.rec.Result())
                    text = r.get("text", "")
                    words = r.get("result", [])

                    if words:
                        conf = sum(w.get("conf", 0.5) for w in words) / len(words)
                    else:
                        # heuristic fallback
                        conf = min(0.6, 0.15 + 0.1 * len(text.split()))

                    return text, round(conf, 2)
        return "", 0.0

    def speak(self, text):
        if not text:
            return
        with self.lock:
            self.tts.say(text)
            self.tts.runAndWait()

# ===================== VISION =====================
class Vision:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.cap = cv2.VideoCapture(0)
        self.last = {"items": [], "time": 0}
        self.lock = threading.Lock()

    def read(self):
        with self.lock:
            ok, frame = self.cap.read()
            if not ok:
                return self.last

            res = self.model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
            h, w, _ = frame.shape
            items = []

            for b in res.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                name = self.model.names[int(b.cls[0])]
                conf = float(b.conf[0])

                items.append({
                    "name": name,
                    "x": (x1 + x2) / 2 / w,
                    "y": (y1 + y2) / 2 / h,
                    "confidence": conf
                })

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"{name} {conf:.2f}",
                            (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,0), 1)

            cv2.imshow("Robot Vision", frame)
            cv2.waitKey(1)

            self.last = {"items": items, "time": time.time()}
            return self.last

# ===================== PLANNER =====================
class Planner:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-3-mini-4k-instruct",
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct")
        self.matcher = ObjectMatcher()

    def plan(self, items, text):
        names = [i["name"] for i in items]

        prompt = f"""
You are a robot planner.
Objects visible: {names}
User command: {text}
Return JSON only:
{{"speech":"...", "steps":[{{"action":"pick","object":"name"}}]}}
"""

        ids = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=200)
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None, 0.0, "I did not understand"

        data = json.loads(m.group())
        confidences = []

        for s in data.get("steps", []):
            obj, score = self.matcher.match(s["object"], names)
            if not obj:
                return None, 0.0, "I cannot see that object"
            s["object"] = obj
            confidences.append(score)

        plan_conf = round(min(confidences) if confidences else 0.0, 2)
        return data, plan_conf, data.get("speech","")

# ===================== ROBOT =====================
class Robot:
    def pick(self, x, y):
        print(f"[ROBOT] PICK at ({x:.2f}, {y:.2f})")

    def stop(self):
        print("[ROBOT] STOP")

# ===================== MAIN =====================
def main():
    term = Terminal()
    voice = VoiceIO()
    vision = Vision()
    planner = Planner()
    robot = Robot()

    cmd_q = queue.Queue()

    def vision_loop():
        while True:
            v = vision.read()
            term.update("Vision", "OK" if v["items"] else "NONE")
            term.update("VisionAge", round(time.time() - v["time"], 2))

    def voice_loop():
        while True:
            text, conf = voice.listen()
            if text:
                term.update("Command", text)
                term.update("VoiceConf", conf)
                cmd_q.put((text, conf))

    def planner_loop():
        while True:
            text, conf = cmd_q.get()
            if conf < VOICE_CONF_MIN:
                voice.speak("Please repeat")
                continue

            with vision.lock:
                v = vision.last

            plan, plan_conf, speech = planner.plan(v["items"], text)
            term.update("PlanConf", plan_conf)

            if plan_conf < PLAN_CONF_MIN:
                voice.speak("I am not confident enough")
                continue

            term.update("Plan", plan["steps"])

            for s in plan["steps"]:
                if s["action"] == "pick":
                    obj = next(i for i in v["items"] if i["name"] == s["object"])
                    robot.pick(obj["x"], obj["y"])

            voice.speak(speech)

    threading.Thread(target=vision_loop, daemon=True).start()
    threading.Thread(target=voice_loop, daemon=True).start()
    threading.Thread(target=planner_loop, daemon=True).start()

    while True:
        term.render()
        time.sleep(0.5)

# ===================== RUN =====================
if __name__ == "__main__":
    main()
