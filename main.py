from voice.tts import speak
from ai.intent_parser import parse_llm_output
from planner.robot_planner import execute_plan
from ai.ollama_llm import ask_llm

speak("Ready")

user_text = "Pick up the bottle"

llm_output = ask_llm(user_text)
intent = parse_llm_output(llm_output)
execute_plan(intent)
print("Intent:", intent)
