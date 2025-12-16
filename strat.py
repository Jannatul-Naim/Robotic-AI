# import threading

# def listen():
#     while True:
#         data = q.get()
#         if rec.AcceptWaveform(data):
#             return json.loads(rec.Result())["text"]

# voice_thread = threading.Thread(target=listen)
# voice_thread.start()

import time
import json

from voice.stt import listen
from voice.tts import speak

from ai.ollama_llm import ask_llm
from planner.executor import execute_plan


SYSTEM_PROMPT = """
You are an AI assistant controlling a robotic arm.
Return ONLY valid JSON in this format:

{
  "action": "pick|place|move|stop",
  "object": "string or null",
  "x": number or null,
  "y": number or null
}
"""


def parse_llm_output(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main():
    speak("System online. Ready for command.")

    while True:
        try:
            # 🎙️ Listen
            user_text = listen()
            if not user_text:
                continue

            print(f"[USER] {user_text}")

            if "exit" in user_text or "shutdown" in user_text:
                speak("Shutting down system")
                break

            # 🧠 LLM
            prompt = SYSTEM_PROMPT + "\nUser: " + user_text
            llm_response = ask_llm(prompt)

            print(f"[LLM]\n{llm_response}")

            intent = parse_llm_output(llm_response)

            if not intent:
                speak("I did not understand that")
                continue

            # 🤖 Execute
            execute_plan(intent)

        except KeyboardInterrupt:
            speak("Emergency stop")
            break

        except Exception as e:
            print("ERROR:", e)
            speak("An error occurred")
            time.sleep(1)


if __name__ == "__main__":
    main()
