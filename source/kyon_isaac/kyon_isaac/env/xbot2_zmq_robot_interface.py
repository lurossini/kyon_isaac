import zmq
import yaml 
import os
import sys 
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# generate python files from proto
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f'{script_dir}/proto', exist_ok=True)
os.chdir(f'{script_dir}/proto')
os.system(f'protoc *.proto --python_out=.')
sys.path.insert(0, f'{script_dir}/proto')

from .proto import generic_rx_msg_pb2, jointstate_pb2, jointcmd_pb2

class ZmqRobot:
    def __init__(self):

        REMOTE_IP = 'localhost'
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{REMOTE_IP}:5557")

        # send joint_names request
        request = {"type": "joint_names"}
        self.socket.send_string(yaml.dump(request))
        response_str = self.socket.recv_string()
        response = yaml.safe_load(response_str)
        self.joint_names = response["data"][1:]

        self.js_socket = context.socket(zmq.SUB)
        self.js_socket.connect(f"tcp://{REMOTE_IP}:5556")
        self.js_socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

        self.cmd_socket = context.socket(zmq.PUB)
        self.cmd_socket.connect(f"tcp://{REMOTE_IP}:5558")

        self.joint_cmd = jointcmd_pb2.JointCommand()
        self.js_msg = jointstate_pb2.JointState()
        self.seq_msg = int()

    def sense(self):
        msg = None
        while msg is None:
            while True:
                try:
                    msg = self.js_socket.recv(flags=zmq.NOBLOCK)
                    rx_msg = generic_rx_msg_pb2.GenericRxMsg()
                    rx_msg.ParseFromString(msg)

                    self.seq_msg = rx_msg.seq

                    if rx_msg.HasField('js'):
                        self.js_msg = rx_msg.js

                    if rx_msg.HasField('imu'):
                        self.imu_msg = rx_msg.imu

                except zmq.Again:
                    break        

    def set_filter_frequency_hz(self, cutoff_freq, enabled=True):
        # send joint_names request
        request = {"type": "set_filter_frequency_hz", "enabled": enabled, "cutoff_hz": cutoff_freq}
        self.socket.send_string(yaml.dump(request))
        response_str = self.socket.recv_string()
        print(response_str)

    def move(self):
        # Serialize and send the message
        cmd_msg = generic_rx_msg_pb2.GenericRxMsg() 
        cmd_msg.stamp = int(time.time() * 1e9)
        cmd_msg.cmd.CopyFrom(self.joint_cmd)
        msg_str = cmd_msg.SerializeToString()
        self.cmd_socket.send(msg_str)

    def enableJoints(self, jnames: list):
        self.joint_cmd.name.extend(jnames)

    def setPositionReference(self, pos_ref: np.ndarray):
        self.joint_cmd.posRef.clear()
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
        return self.js_msg.linkPos[6:]

    def getMotorPosition(self):
        return self.js_msg.motPos[6:]

    def getPositionReference(self):
        return self.js_msg.posRef[6:]

    def getVelocityReference(self):
        return self.js_msg.velRef[6:]

    def getEffortReference(self):
        return self.js_msg.torRef[6:]

    def getJointVelocities(self):
        return self.js_msg.linkVel[6:]

    def getMotorVelocities(self):
        return self.js_msg.motVel[6:]

    def getJointEffort(self):
        return self.js_msg.tor[6:]

    def getStiffness(self):
        return self.js_msg.k[6:]

    def getDamping(self):
        return self.js_msg.d[6:]
    
    def getImuAngularVelocity(self):
        return [self.imu_msg.angular_velocity_x, self.imu_msg.angular_velocity_y, self.imu_msg.angular_velocity_z]
    
    def getImuOrientation(self):
        w_T_imu = R.from_quat([self.imu_msg.orientation_x, self.imu_msg.orientation_y, self.imu_msg.orientation_z, self.imu_msg.orientation_w])
        return R.as_matrix(w_T_imu)
    
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



