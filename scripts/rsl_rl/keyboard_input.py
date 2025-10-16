from evdev import InputDevice, categorize, ecodes, list_devices
import threading


class KeyboardIO:
    def __init__(self):
        # List all input devices
        devices = [InputDevice(path) for path in list_devices()]
        for dev in devices:
            print(dev.path, dev.name, dev.phys)

        # Pick your keyboard (usually /dev/input/eventX)
        keyboard = InputDevice('/dev/input/event3')

        # keyboard-reference mapper
        self.key_map = {
            "KEY_W": [1, 0, 0],
            "KEY_S": [-1, 0, 0],
            "KEY_A": [0, 1, 0],
            "KEY_D": [0, -1, 0],
            "KEY_Q": [1, 0, 1],
            "KEY_E": [1, 0, -1],
            "KEY_Z": [-1, 0, -1],
            "KEY_C": [-1, 0, 1],
            "KEY_P": [0, 0, -1],
            "KEY_O": [0, 0, 1], 
        }
        self.ref = [0, 0, 0]
        print(f"Listening to {keyboard.path} ({keyboard.name})")

        def loop():
            for event in keyboard.read_loop():
                if event.type == ecodes.EV_KEY:
                    self.key_event = categorize(event)
                    if self.key_event.keystate == self.key_event.key_down:
                        self.ref = self.key_map[self.key_event.keycode] if self.key_event.keycode in self.key_map else [0, 0, 0]
                        print(f"Key pressed: {self.key_event.keycode}")
                    elif self.key_event.keystate == self.key_event.key_up:
                        self.ref = [0, 0, 0]
                        print(f"Key released: {self.key_event.keycode}")

        # Start the thread
        thread = threading.Thread(target=loop, daemon=True)
        thread.start()
    
    def get_key(self) -> list:
        return self.ref



