from ai.ollama_llm import ask_llm
import json

SYSTEM_PROMPT = """
You are a robotic control system.
Return ONLY valid JSON.

Format:
{
  "action": "pick|place|move|stop",
  "object": "string or null",
  "x": number or null,
  "y": number or null
}
"""


def parse_command(user_text):
    prompt = SYSTEM_PROMPT + "\nUser: " + user_text
    return ask_llm(prompt)


def parse_llm_output(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

