import zmq
import yaml 
import os
import sys 
import time
import numpy as np

# generate python files from proto
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(f'{script_dir}/proto')
os.makedirs(f'{script_dir}/proto', exist_ok=True)
os.system(f'protoc *.proto --python_out={script_dir}/proto')
sys.path.append(f'{script_dir}/proto')

from proto import generic_rx_msg_pb2, jointstate_pb2, jointcmd_pb2

class ZmqRobot:
    def __init__(self):
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://10.240.23.65:5557")

        # send joint_names request
        request = {"type": "joint_names"}
        self.socket.send_string(yaml.dump(request))
        response_str = self.socket.recv_string()
        response = yaml.safe_load(response_str)
        self.joint_names = response["data"][1:]
        print(self.joint_names)

        self.js_socket = context.socket(zmq.SUB)
        self.js_socket.connect("tcp://10.240.23.65:5556")
        self.js_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

        self.cmd_socket = context.socket(zmq.PUB)
        self.cmd_socket.connect("tcp://10.240.23.65:5558")

        self.joint_cmd = jointcmd_pb2.JointCommand()
        self.js_msg = jointstate_pb2.JointState()
        self.seq_msg = int()

    def sense(self):
        try:
            msg = self.js_socket.recv()
            rx_msg = generic_rx_msg_pb2.GenericRxMsg()
            rx_msg.ParseFromString(msg)

            assert rx_msg.seq >= 0
            assert rx_msg.js is not None

            self.seq_msg = rx_msg.seq

            if rx_msg.has_js:
                self.js_msg = rx_msg.js

            print(f"Received JointState message with seq: {rx_msg.seq}")
            print(f"linkPos: {js_msg.linkPos}")
            assert len(js_msg.linkPos) == len(js_msg.motVel)

        except zmq.Again:
            print("No message received yet.")

    def move(self):
        # Serialize and send the message
        msg_str = self.joint_cmd.SerializeToString()
        
        self.cmd_socket.send(msg_str)
        time.sleep(1)

        print("Sent JointCommand message.")

    def enableJoints(self, jnames: list):
        self.joint_cmd.name.extend(jnames)

    def setPositionReference(self, pos_ref: np.ndarray):
        self.joint_cmd.posRef.extend(pos_ref)

    def setVelocityReference(self, vel_ref: np.ndarray):
        self.joint_cmd.velRef.extend(vel_ref)

    def setEffortReference(self, tor_ref: np.ndarray):
        self.joint_cmd.torRef.extend(tor_ref)

    def setStiffness(self, K: np.ndarray):
        self.joint_cmd.k.extend(K)

    def setDamping(self, D: np.ndarray):
        self.joint_cmd.d.extend(D)

    def setCtrlMode(self, ctrl_mode: np.ndarray):
        self.joint_cmd.ctrl.extend(ctrl_mode)

    def getJointPosition(self):
        return self.js_msg.linkPos()

    def getMotorPosition(self):
        return self.js_msg.motPos()

    def getPositionReference(self):
        return self.js_msg.posRef()

    def getVelocityReference(self):
        return self.js_msg.velRef()

    def getEffortReference(self):
        return self.js_msg.torRef()

    def getJointVelocities(self):
        return self.js_msg.linkVel()

    def getMotorVelocities(self):
        return self.js_msg.motVel()

    def getJointEffort(self):
        return self.js_msg.tor

    def getStiffness(self):
        return self.js_msg.k

    def getDamping(self):
        return self.js_msg.d

def main():
    print("Start")

    zmq_robot = ZmqRobot()
    print('ZmqRobot created')
    home = np.array([0, 0.7, -1.4, 0, -0.7, 1.4, 0, 0.7, -1.4, 0, -0.7, 1.4,])
    zmq_robot.setPositionReference(home)
    zmq_robot.setVelocityReference(np.zeros_like(home))
    zmq_robot.setEffortReference(np.zeros_like(home))
    zmq_robot.setStiffness(np.ones_like(home) * 1500)
    zmq_robot.setDamping(np.ones_like(home) * 20)
    zmq_robot.enableJoints(zmq_robot.joint_names)
    zmq_robot.setCtrlMode(np.ones_like(home, dtype=int) * 25)
    while True:
        zmq_robot.move()

if __name__ == "__main__":
    main()



