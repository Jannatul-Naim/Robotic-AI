
import pyautogui
import random
import time

keys = ['w', 'a', 's', 'd']

while True:
    key = random.choice(keys)
    duration = random.uniform(2, 4)

    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

    time.sleep(random.uniform(0.2, 0.8))
