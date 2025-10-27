import pygame
import os
import sys
import time
import zmq

# generate python files from proto
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)
os.makedirs(f'{script_dir}/proto', exist_ok=True)
os.chdir(f'{script_dir}/proto')
os.system(f'protoc *.proto --python_out=.')
sys.path.insert(0, f'{script_dir}/proto')

from proto import joy_msg_pb2

REMOTE_IP = 'localhost'
context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.connect(f"tcp://{REMOTE_IP}:5050")

os.environ["SDL_JOYSTICK_DEVICE"] = "/dev/input/js0"

pygame.init()
pygame.joystick.init()

# Check for joysticks
if pygame.joystick.get_count() == 0:
    print("No joystick connected.")
    exit()

joystick = pygame.joystick.Joystick(0)
joystick.init()

print(f"Joystick detected: {joystick.get_name()}")

DEADZONE = 0.1
def apply_deadzone(value, deadzone=DEADZONE):
    if abs(value) < deadzone:
        return 0.0
    return value

msg = joy_msg_pb2.JoyMsg()
seq = int(0)
rate = 10.0

def main():
    global seq
    while True:
        try:
            msg.axes.clear()
            msg.buttons.clear()
            pygame.event.pump()
            for i in range(joystick.get_numaxes()):
                msg.seq = seq
                msg.stamp = int(time.time() * 1e9)
                msg.axes.append(joystick.get_axis(i))
                msg.buttons.append(joystick.get_button(i))
            seq += 1
            print([msg.axes[1], msg.axes[0], msg.axes[3]])
            msg_str = msg.SerializeToString()
            socket.send(msg_str)
            time.sleep(1./rate)
        except KeyboardInterrupt:
            sys.exit(0)

if __name__ == "__main__":
    main()
