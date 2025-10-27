from evdev import InputDevice, categorize, ecodes, list_devices
import threading

import numpy as np


class KeyboardIO:
    def __init__(self):
        # List all input devices
        devices = [InputDevice(path) for path in list_devices()]
        for dev in devices:
            print(dev.path, dev.name, dev.phys)

        # Pick your keyboard (usually /dev/input/eventX)
        keyboard = InputDevice('/dev/input/event3')
        self.scale = 1.
        self.last_key = None

        # keyboard-reference mapper
        self.key_map = {
            "KEY_W": np.array([1, 0, 0]),
            "KEY_S": np.array([-1, 0, 0]),
            "KEY_A": np.array([0, 1, 0]),
            "KEY_D": np.array([0, -1, 0]),
            "KEY_Q": np.array([1, 0, 1]),
            "KEY_E": np.array([1, 0, -1]),
            "KEY_Z": np.array([-1, 0, -1]),
            "KEY_C": np.array([-1, 0, 1]),
            "KEY_P": np.array([0, 0, -1]),
            "KEY_O": np.array([0, 0, 1]), 
        }
        self.ref = np.array([0, 0, 0])
        print(f"Listening to {keyboard.path} ({keyboard.name})")

        def loop():
            for event in keyboard.read_loop():
                if event.type == ecodes.EV_KEY:
                    self.key_event = categorize(event)
                    if self.key_event.keystate == self.key_event.key_down:
                        if self.key_event.keycode in self.key_map:
                            self.last_key = self.key_event.keycode
                            self.ref = self.scale * self.key_map[self.key_event.keycode]  
                        elif self.key_event.keycode == "KEY_M":
                            self.scale += 0.1
                            self.ref = self.scale * self.key_map[self.last_key]  
                        elif self.key_event.keycode == "KEY_N":
                            self.scale -= 0.1
                            self.ref = self.scale * self.key_map[self.last_key]  
                        else:
                            self.ref = np.array([0, 0, 0])
                    # elif self.key_event.keystate == self.key_event.key_up:
                        # self.ref = [0, 0, 0]

        # Start the thread
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
    
    def get_key(self) -> list:
        return self.ref.tolist()



