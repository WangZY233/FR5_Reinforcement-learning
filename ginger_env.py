import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pybullet as p
import pybullet_data
import math
import time
from pybullet_utils import bullet_client
from scipy.spatial.transform import Rotation as R
from loguru import logger
import random

class Ginger_Env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self,gui = False):
        super(Ginger_Env).__init__()
        self.step_num = 0

        # 设置最小的关节变化量
        low_action = np.array([-1.0,-1.0,-1.0,-1.0,-1.0,-1.0])
        high_action = np.array([1.0,1.0,1.0,1.0,1.0,1.0])
        self.action_space = spaces.Box(low=low_action, high=high_action, dtype=np.float32)

        low = np.zeros((1,12),dtype=np.float32)
        high = np.ones((1,12),dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else :
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)
        self.p.setTimeStep(1/240)
        # print(self.p)
        self.p.setGravity(0, 0, -9.81)
        self.p.resetSimulation()
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # boxId = self.p.loadURDF("plane.urdf")
        # 创建机器人
        self.fr5 = self.p.loadURDF("F:\\Pycharm_project\\RL\\ginger_description\\urdf\\ginger_description.urdf")


        
                                                              

    def step(self, action):

        return self.observation, self.reward, self.terminated, self.truncated, info

    def reset(self, seed=None, options=None):

        return self.observation,info

    def render(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=1.7, cameraYaw=0, cameraPitch=-20, cameraTargetPosition=[0.45, 0, 0.8])
    
    def close(self):
        self.p.disconnect()

    def add_noise(self, angle, range, gaussian=False):
        if gaussian:
            angle += np.clip(np.random.normal(0, 1) * range, -1, 1)
        else:
            angle += random.uniform(-5, 5)
        return angle

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env 
    Env = Ginger_Env(gui=True)
    # Env.reset()
    # check_env(Env, warn=True)
    # for i in range(100):
    #         p.stepSimulation()
    #         time.sleep(1./240.)
    Env.render()
    print("test going")
    time.sleep(10)
    # observation, reward, terminated, truncated, info = Env.step([0,0,0,0,0,20])
    # print(reward)
    time.sleep(100)
