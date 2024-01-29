import numpy as np
from gymnasium import spaces
import math
from panda_gym.envs.core import PyBulletRobot


class Fr5(PyBulletRobot):
    """My robot"""

    def __init__(self, sim):
        action_dim = 7 # = number of joints; here, 1 joint, so dimension = 1
        self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        
        super().__init__(
            sim,
            body_name="Fr5",  # choose the name you want
            file_name="F:\\Pycharm_project\\RL\\fr5_description\\urdf\\fr5_robot.urdf",  # the path of the URDF file
            base_position=np.zeros(3),  # the position of the base
            action_space=self.action_space,
            joint_indices=np.array([0,0,0,0,0,0,0]),  # list of the indices, as defined in the URDF
            joint_forces=np.array([5.0,5.0,5.0,5.0,5.0,5.0,5.0]),  # force applied when robot is controled (Nm)
        )

    def set_action(self, action):
        self.control_joints(target_angles=action)

    def get_obs(self):
        j0 = self.get_joint_angle(joint=0)
        j1 = self.get_joint_angle(joint=1)
        j2 = self.get_joint_angle(joint=2)
        j3 = self.get_joint_angle(joint=3)
        j4 = self.get_joint_angle(joint=4)
        j5 = self.get_joint_angle(joint=5)
        j6 = self.get_joint_angle(joint=6)
        # j7 = self.get_joint_angle(joint=7)
        # print(j0,j1,j2,j3,j4,j5,j6)
        obs = np.array([j0,j1,j2,j3,j4,j5,j6])
        return obs

    def reset(self):
        neutral_angle = np.array([0.0,0.0,-math.pi/2,0.0,-math.pi/2,0.0,0.0])
        self.set_joint_angles(angles=neutral_angle)
